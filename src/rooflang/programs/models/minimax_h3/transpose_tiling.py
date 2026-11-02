"""Uncalibrated phase-tiled ConvTranspose1d cost, not a CANN reconstruction.

Output tiles own their accumulators; K panels gather only valid input payload.
There is no cross-tile cache reuse and no global scatter/col2im workspace.
The fixed template is a model choice, not an asserted hardware instruction.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
import math

MT, NT, KT = 16, 16, 8


def _ceil(n, d):
    return -(-n // d)


@dataclass(frozen=True)
class TransposeGeometry:
    batch: int
    cin: int
    cout: int
    length: int
    kernel: int
    stride: int = 1
    padding: int = 0
    dilation: int = 1
    groups: int = 1
    bias: bool = True

    def __post_init__(self):
        for name in ('batch', 'cin', 'cout', 'length', 'kernel', 'stride', 'dilation', 'groups'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.padding) is not int or self.padding < 0:
            raise ValueError('padding must be a nonnegative integer')
        if type(self.bias) is not bool:
            raise ValueError('bias must be boolean')
        if self.cin % self.groups or self.cout % self.groups:
            raise ValueError('groups must divide both channel counts')
        if self.output_length <= 0:
            raise ValueError('output length must be positive')

    @property
    def output_length(self):
        return ((self.length - 1) * self.stride - 2 * self.padding
                + self.dilation * (self.kernel - 1) + 1)


@dataclass(frozen=True)
class PhaseCost:
    phase: int
    rows: int
    columns: int
    reduction: int
    taps: tuple[int, ...]
    tasks: int
    waves: int
    tile_cube_flops: int


@dataclass(frozen=True)
class TransposeCost:
    core_count: int
    phases: tuple[PhaseCost, ...]
    reference_flops: int
    valid_pairs: int
    valid_flops: int
    executed_cube_flops: int
    executed_vector_flops: int
    critical_core_cube_flops: int
    critical_core_vector_flops: int
    input_read_bytes: int
    filter_read_bytes: int
    bias_read_bytes: int
    output_write_bytes: int
    workspace_bytes: int = 0
    onchip_buffer_bytes: int = (MT * KT + KT * NT + MT * NT) * 4

    @property
    def activation_bytes(self):
        return self.input_read_bytes + self.filter_read_bytes + self.output_write_bytes + 16

    def compute_time_us(self, tflops):
        time = 0.0
        for dtype, work in (('fp32:cube', self.critical_core_cube_flops),
                            ('fp32:vector', self.critical_core_vector_flops)):
            if not work:
                continue
            peak = tflops.get(dtype, 0)
            if not math.isfinite(peak) or peak <= 0:
                raise ValueError(f'positive finite throughput required for {dtype}')
            time += self.core_count * work / (peak * 1e6)
        return time

    def to_dict(self):
        return {**asdict(self), 'tile': [MT, NT, KT], 'activation_bytes': self.activation_bytes,
                'model': 'generic_phase_tiled_transpose', 'calibration': 'none',
                'implementation': 'not_cann_kernel_reconstruction',
                'reduction_order': 'tap_major_channel_minor',
                'task_order': 'phase_batch_group_m_n', 'cross_tile_cache_reuse': False,
                'traffic_scope': 'fp32_payload_not_bus_transactions',
                'overlap': 'kernel_level_roofline', 'output_size_read_bytes': 16}


def _merged(intervals):
    """Union intervals for one channel within a K panel."""
    merged = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else:
            merged.append((lo, hi))
    return merged


def _positive_sum(a, step, count):
    """sum(max(a - t*step, 0), t=0..count-1), without enumerating t."""
    n = min(count, max(0, _ceil(a, step)))
    return n * a - step * n * (n - 1) // 2


def _clipped_interval_sum(lo, hi, length, count):
    """Total valid payload of [lo+t*MT,hi+t*MT) across full M tiles."""
    width = hi - lo
    def clipped(a):
        return _positive_sum(a, MT, count) - _positive_sum(a - width, MT, count)
    return clipped(length - lo) - clipped(-lo)


def _input_elements(g, q, taps, rows):
    ci = g.cin // g.groups
    full, tail = divmod(rows, MT)
    elements = 0
    for k in range(0, len(taps) * ci, KT):
        offsets = {}
        for index in range(k, min(k + KT, len(taps) * ci)):
            r, c = taps[index // ci], index % ci
            offsets.setdefault(c, []).append((q + g.padding - r * g.dilation) // g.stride)
        for channel_offsets in offsets.values():
            for lo, hi in _merged([(a, a + MT) for a in channel_offsets]):
                elements += _clipped_interval_sum(lo, hi, g.length, full)
            if tail:
                for lo, hi in _merged([(a + full * MT, a + full * MT + tail)
                                       for a in channel_offsets]):
                    elements += max(0, min(g.length, hi) - max(0, lo))
    return elements


class _WaveMax:
    """Stream run lengths and repeated patterns, retaining one partial wave."""
    def __init__(self, cores):
        self.cores, self.fill, self.maximum, self.total = cores, 0, 0, 0

    def run(self, value, count):
        if self.fill and count:
            take = min(count, self.cores - self.fill)
            self.fill += take
            self.maximum = max(self.maximum, value)
            count -= take
            if self.fill == self.cores:
                self.total += self.maximum
                self.fill = self.maximum = 0
        waves, tail = divmod(count, self.cores)
        self.total += waves * value
        if tail:
            self.fill, self.maximum = tail, value

    def repeat(self, action, count):
        seen = {}
        i = 0
        while i < count:
            state = (self.fill, self.maximum)
            if state in seen:
                previous_i, previous_total = seen[state]
                cycles = (count - i) // (i - previous_i)
                if cycles:
                    self.total += cycles * (self.total - previous_total)
                    i += cycles * (i - previous_i)
                    seen.clear()
                    continue
            seen[state] = (i, self.total)
            action()
            i += 1


def _vector_critical(rows, columns, repeats, cores):
    wave = _WaveMax(cores)
    mf, mt = divmod(rows, MT)
    nf, nt = divmod(columns, NT)
    def row(height):
        wave.run(height * NT, nf)
        if nt:
            wave.run(height * nt, 1)
    def group():
        wave.repeat(lambda: row(MT), mf)
        if mt:
            row(mt)
    wave.repeat(group, repeats)
    return wave.total + wave.maximum


def transpose_cost(g: TransposeGeometry, core_count: int) -> TransposeCost:
    # Validate before cache lookup: bool and int can compare equal as cache keys.
    if type(core_count) is not int or core_count <= 0:
        raise ValueError('core_count must be a positive integer')
    return _transpose_cost(g, core_count)


@lru_cache(maxsize=512)
def _transpose_cost(g, cores):
    ci, co = g.cin // g.groups, g.cout // g.groups
    out, repeats = g.output_length, g.batch * g.groups
    phases = []
    cube = cube_critical = vector_critical = inputs = filters = biases = 0
    pairs = sum(max(0, min(g.length, (out - 1 + g.padding - r * g.dilation) // g.stride + 1)
                    - max(0, _ceil(g.padding - r * g.dilation, g.stride)))
                for r in range(g.kernel))
    for q in range(min(g.stride, out)):
        rows = _ceil(out - q, g.stride)
        taps = tuple(r for r in range(g.kernel) if (q + g.padding - r * g.dilation) % g.stride == 0)
        reduction = ci * len(taps)
        tasks = repeats * _ceil(rows, MT) * _ceil(co, NT)
        waves = _ceil(tasks, cores)
        work = 2 * MT * NT * _ceil(reduction, KT) * KT
        phases.append(PhaseCost(q, rows, co, reduction, taps, tasks, waves, work))
        cube += tasks * work
        cube_critical += waves * work
        inputs += 4 * repeats * _ceil(co, NT) * _input_elements(g, q, taps, rows)
        filters += 4 * repeats * _ceil(rows, MT) * co * reduction
        if g.bias:
            biases += 4 * repeats * _ceil(rows, MT) * co
            vector_critical += _vector_critical(rows, co, repeats, cores)
    vector = g.batch * g.cout * out if g.bias else 0
    return TransposeCost(
        cores, tuple(phases), 2 * g.batch * g.length * g.cin * co * g.kernel + vector,
        pairs, 2 * g.batch * g.cin * co * pairs + vector, cube, vector,
        cube_critical, vector_critical, inputs, filters, biases,
        4 * g.batch * _ceil(g.cout, 8) * 8 * out)
