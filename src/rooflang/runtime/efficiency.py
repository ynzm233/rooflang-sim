"""Shared model efficiency configuration and per-run hardware preparation."""
from copy import deepcopy
import math
import re

from rooflang.language.hardware.component import Compute


_SCALARS = (
    'cube_efficiency', 'vector_efficiency',
    'cube_memory_efficiency', 'vector_memory_efficiency',
    'intra_node_efficiency',
)


def _validate_value(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not 0 < value <= 1 or not math.isfinite(value)):
        raise ValueError(f'efficiency.{name} must be finite and in (0, 1]')
    return float(value)


def resolve_efficiency(value=None):
    """Resolve omitted coefficients to one; an empty collective table means one."""
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError('efficiency must be an object')
    unknown = set(value) - {*_SCALARS, 'collective_efficiencies'}
    if unknown:
        raise ValueError(f'Unknown efficiency fields: {sorted(unknown, key=str)}')
    resolved = {name: _validate_value(value.get(name, 1.0), name) for name in _SCALARS}
    table = value.get('collective_efficiencies', {})
    if not isinstance(table, dict):
        raise ValueError('efficiency.collective_efficiencies must be an object')
    normalized = {}
    for key, coefficient in table.items():
        if type(key) is int and key > 0:
            count = key
        elif isinstance(key, str) and re.fullmatch(r'[1-9][0-9]*', key):
            count = int(key)
        else:
            raise ValueError('efficiency.collective_efficiencies keys must be positive integers')
        if count in normalized:
            raise ValueError(f'Duplicate collective participant count: {count}')
        normalized[count] = _validate_value(coefficient, f'collective_efficiencies.{count}')
    resolved['collective_efficiencies'] = normalized
    return resolved


def prepare_hardware(hardware, efficiency):
    """Copy hardware and apply model compute efficiencies without shared mutation.

    Unit-less peaks supply both unit aliases. Explicit Cube/Vector peaks retain
    their existing distinctions; communication is applied by the simulator.
    """
    resolved = resolve_efficiency(efficiency)
    prepared = deepcopy(hardware)
    prepared.metadata = deepcopy(getattr(prepared, 'metadata', {}))
    prepared.metadata['efficiency'] = resolved
    for device in prepared.nodes:
        if not isinstance(device, Compute) or not device.tflops:
            continue
        base = dict(getattr(device, '_efficiency_base_tflops', device.tflops))
        device._efficiency_base_tflops = base
        peaks = dict(base)
        unit_peaks = any(key.endswith((':cube', ':vector')) for key in base)
        for dtype, peak in base.items():
            if ':' not in dtype:
                peaks[dtype] = peak * resolved['cube_efficiency']
                if not unit_peaks:
                    for unit in ('cube', 'vector'):
                        peaks[f'{dtype}:{unit}'] = peak * resolved[f'{unit}_efficiency']
            elif dtype.endswith((':cube', ':vector')):
                unit = dtype.rsplit(':', 1)[1]
                peaks[dtype] = peak * resolved[f'{unit}_efficiency']
        device.tflops = peaks
    return prepared


def apply_memory_efficiency(kernel, hardware):
    """Scale local memory time using operator units, without changing traffic."""
    efficiency = getattr(hardware, 'metadata', {}).get('efficiency')
    if efficiency is None:
        return
    op = kernel.operator
    component = op.get_compute_component()
    split = getattr(op, 'get_memory_bytes_by_unit', None)
    if split is not None:
        by_unit = split()
        basis = 'operator_memory_bytes_by_unit'
    else:
        unit = ('vector' if component is None or component.unit_type in ('vector', 'storage')
                else 'cube')
        by_unit = {unit: op.get_memory_bytes()}
        basis = 'operator_unit_or_cube_for_generic_mixed_boundary'
    coefficients = {unit: efficiency[f'{unit}_memory_efficiency'] for unit in by_unit}
    total = sum(by_unit.values())
    weighted = sum(size / coefficients[unit] for unit, size in by_unit.items())
    kernel.local_memory_efficiency = total / weighted if weighted else 1.0
    kernel.csv_metadata.update(
        memory_bytes_by_unit=by_unit, memory_efficiency_by_unit=coefficients,
        local_memory_efficiency=kernel.local_memory_efficiency,
        memory_efficiency_basis=basis,
    )


def _node_id(hardware, device):
    if hasattr(device, 'node_id'):
        return device.node_id
    machines = getattr(hardware, '_machine_by_gpu', {})
    if device in machines:
        return machines[device]
    # Existing GPU presets use the same nN prefix in their bandwidth formulas.
    prefix = device.name.split('-', 1)[0]
    return prefix if re.fullmatch(r'n[0-9]+', prefix) else 0


def communication_efficiency(hardware, devices, *, point_to_point=False):
    """Return the bandwidth factor for a communication's actual participants.

    Intra-node efficiency applies only when all participants share a node.
    Collective tables retain nearest-count lookup and insertion-order ties;
    native point-to-point recipes retain their existing table bypass.
    """
    efficiency = getattr(hardware, 'metadata', {}).get('efficiency')
    if efficiency is None or len(devices) < 2:
        return 1.0
    same_node = len({_node_id(hardware, device) for device in devices}) == 1
    factor = efficiency['intra_node_efficiency'] if same_node else 1.0
    table = efficiency['collective_efficiencies']
    if not point_to_point and table:
        key = min(table, key=lambda count: abs(count - len(devices)))
        factor *= table[key]
    return factor


def remote_memory_efficiency(hardware, device, memory):
    """Apply peer-network efficiency to remote accelerator buffers, not DMA."""
    efficiency = getattr(hardware, 'metadata', {}).get('efficiency')
    if (efficiency is None or efficiency['intra_node_efficiency'] == 1
            or device.kind != 'gpu'):
        return 1.0
    owner = hardware.find_local_device(memory)
    if owner.kind != 'gpu':
        return 1.0
    return communication_efficiency(hardware, (device, owner), point_to_point=True)
