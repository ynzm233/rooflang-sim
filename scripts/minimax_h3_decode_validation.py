"""Compare analytical H3 decode kernels with local profiler CSVs.

Run from the checkout with ``python scripts/minimax_h3_decode_validation.py``.
The model never reads these measurements; they are validation inputs only.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
STAGES = ('video_vae_decode', 'audio_vae_decode')
OP_FIELDS = {
    'stage', 'dit_step', 'kernel_name', 'op_type', 'input_shapes', 'input_dtypes',
    'output_shapes', 'output_dtypes', 'calls', 'valid_calls',
    'invalid_duration_count', 'total_ms', 'avg_ms', 'min_ms', 'max_ms',
}
SUMMARY_FIELDS = {
    'stage', 'dit_step', 'marker_calls', 'host_total_ms', 'kernel_count',
    'invalid_duration_count', 'kernel_total_ms', 'device_busy_ms', 'device_span_ms',
}


def _rows(path, required):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: missing columns {sorted(required - set(reader.fieldnames or []))}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'{path}: empty measurement file')
    return rows


def _number(row, key, location, *, integer=False):
    raw = row.get(key)
    if raw is None or not str(raw).strip():
        raise ValueError(f'{location}: missing {key}')
    try:
        if integer and not str(raw).isdigit():
            raise ValueError
        value = int(raw) if integer else float(raw)
    except (TypeError, ValueError):
        raise ValueError(f'{location}: invalid {key}={raw!r}') from None
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{location}: {key} must be finite and nonnegative')
    return value


def _digest(path):
    return {'path': str(path.resolve()), 'sha256': sha256(path.read_bytes()).hexdigest()}


def load_measurements(directory):
    """Validate aggregated leaf kernels against their independent stage totals."""
    directory = Path(directory)
    summary_path = directory / 'stage_summary.csv'
    summaries = _rows(summary_path, SUMMARY_FIELDS)
    result = {'files': {'stage_summary.csv': _digest(summary_path)}, 'stages': {}}
    for stage in STAGES:
        path = directory / f'{stage}.csv'
        groups = _rows(path, OP_FIELDS)
        expected = f'h3/{stage}'
        by_operator = defaultdict(lambda: {'calls': 0, 'kernel_total_ms': 0.0})
        signatures = set()
        for line, row in enumerate(groups, 2):
            location = f'{path}:{line}'
            if row.get('stage') != expected or row.get('dit_step') != '':
                raise ValueError(f'{location}: expected unstepped stage {expected}')
            if not row.get('op_type') or not row.get('kernel_name'):
                raise ValueError(f'{location}: empty kernel_name or op_type')
            signature = tuple(row.get(key) for key in (
                'kernel_name', 'op_type', 'input_shapes', 'input_dtypes',
                'output_shapes', 'output_dtypes'))
            if signature in signatures:
                raise ValueError(f'{location}: duplicate aggregated kernel signature')
            signatures.add(signature)
            calls = _number(row, 'calls', location, integer=True)
            valid = _number(row, 'valid_calls', location, integer=True)
            invalid = _number(row, 'invalid_duration_count', location, integer=True)
            if calls == 0 or valid != calls or invalid:
                raise ValueError(f'{location}: every call must have a valid duration')
            total, average, low, high = (
                _number(row, key, location) for key in ('total_ms', 'avg_ms', 'min_ms', 'max_ms'))
            if (not math.isclose(total, average * calls, rel_tol=1e-7, abs_tol=1e-6 * calls)
                    or average < low - 1e-6 or average > high + 1e-6):
                raise ValueError(f'{location}: inconsistent total/average/min/max duration')
            by_operator[row['op_type']]['calls'] += calls
            by_operator[row['op_type']]['kernel_total_ms'] += total
        matches = [row for row in summaries if row.get('stage') == expected and row.get('dit_step') == '']
        if len(matches) != 1:
            raise ValueError(f'{summary_path}: expected exactly one summary for {expected}')
        summary = matches[0]
        location = f'{summary_path}:{expected}'
        if _number(summary, 'invalid_duration_count', location, integer=True):
            raise ValueError(f'{location}: invalid kernel durations')
        count = _number(summary, 'kernel_count', location, integer=True)
        total = _number(summary, 'kernel_total_ms', location)
        if (count != sum(group['calls'] for group in by_operator.values())
                or not math.isclose(total, sum(group['kernel_total_ms'] for group in by_operator.values()),
                                    rel_tol=1e-8, abs_tol=1e-5)):
            raise ValueError(f'{location}: operator CSV does not reconcile with stage summary')
        detail = {key: _number(summary, key, location) for key in (
            'host_total_ms', 'device_busy_ms', 'device_span_ms')}
        if detail['device_span_ms'] + 1e-5 < detail['device_busy_ms']:
            raise ValueError(f'{location}: device span is shorter than device busy time')
        result['files'][path.name] = _digest(path)
        result['stages'][stage] = {
            'kernel_count': count, 'kernel_total_ms': total,
            'signature_count': len(groups), 'by_operator': dict(sorted(by_operator.items())),
            'marker_calls': _number(summary, 'marker_calls', location, integer=True), **detail,
        }
    return result


def summarize_simulation(stage):
    """Retain graph repetitions; deduplicate only repeated trace entries."""
    by_operator = defaultdict(lambda: {'calls': 0, 'kernel_total_ms': 0.0})
    for graph in stage.graphs:
        unique = {entry.kernel: entry for entry in graph.result.trace}
        for kernel, entry in unique.items():
            op_type = kernel.csv_metadata['operator']
            if op_type in ('WeightLoad', 'LayerRelease'):
                raise ValueError('Kernel comparison requires layerwise_offload=False')
            by_operator[op_type]['calls'] += graph.multiplier
            by_operator[op_type]['kernel_total_ms'] += (
                (entry.end_us - entry.start_us) / 1000 * graph.multiplier)
    return {
        'kernel_count': sum(group['calls'] for group in by_operator.values()),
        'kernel_total_ms': sum(group['kernel_total_ms'] for group in by_operator.values()),
        'device_span_ms': sum(graph.result.total_time_us / 1000 * graph.multiplier for graph in stage.graphs),
        'by_operator': dict(sorted(by_operator.items())),
        'graph_multipliers': {graph.label: graph.multiplier for graph in stage.graphs if hasattr(graph, 'label')},
    }


def signed_error_pct(simulated, measured):
    return (simulated - measured) / measured * 100 if measured else None


def _markdown(report):
    lines = [
        '# MiniMax-H3 decode：解析模型与实测核对', '',
        '比较口径为设备 kernel 时长之和。关闭逐层卸载，使模拟结果不含 H2D；'
        'host_total 与 device_span 单独列出，不加到 kernel_total 上。', '',
        '时延误差 = (模拟 kernel 总时间 − 实测 kernel 总时间) / 实测 kernel 总时间 × 100%。'
        '各时间单位为 ms；误差为有符号百分比，负值表示低估。不设置通过阈值。', '',
        '实测 CSV 只参与验证，模型运行不读取实测时延。硬件采用当前 Ascend910B3 单卡预设；'
        'profiler CSV 本身不含硬件身份，该硬件与请求对应关系是本次比较的假设。', '',
        '| 阶段 | 实测 kernel ms | 模拟 kernel ms | 误差 | 实测调用数 | 模拟调用数 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for stage, values in report['stages'].items():
        m, s = values['measured'], values['simulated']
        error = values['kernel_time_error_pct']
        error_text = f'{error:+.2f}%' if error is not None else '未定义（实测为零）'
        lines.append(f"| {stage} | {m['kernel_total_ms']:.6f} | {s['kernel_total_ms']:.6f} | "
                     f"{error_text} | {m['kernel_count']} | {s['kernel_count']} |")
    lines += ['', '| 阶段 | 实测 host_total ms | 实测 device_busy ms | 实测 device_span ms |',
              '|---|---:|---:|---:|']
    for stage, values in report['stages'].items():
        m = values['measured']
        lines.append(f"| {stage} | {m['host_total_ms']:.6f} | {m['device_busy_ms']:.6f} | {m['device_span_ms']:.6f} |")
    lines += ['', '模型仍不包含主机调度、kernel launch 固定开销、内部 workspace/L2 流量、'
              '框架同步和真实 kernel 效率校准。各算子类型的调用差额在下表逐项列出；'
              '音频分组转置卷积的真实低利用率不能从 FLOPs 或逻辑字节数直接恢复。', '']
    for stage, values in report['stages'].items():
        lines += [f'## {stage}', '', '| 算子类型 | 实测次数 | 模拟次数 | 差额 | 实测 ms | 模拟 ms |',
                  '|---|---:|---:|---:|---:|---:|']
        for row in values['operators']:
            lines.append(f"| {row['operator']} | {row['measured_calls']} | {row['simulated_calls']} | "
                         f"{row['call_delta']:+d} | {row['measured_ms']:.6f} | {row['simulated_ms']:.6f} |")
        lines += ['', f"[模拟算子 CSV](simulation/{stage}/{stage}.csv) · "
                  f"[配置快照](simulation/{stage}/config.snapshot.json)", '']
    lines += ['## 复现来源', '', '[完整数值与 SHA-256](decode_validation.json)', '',
              f"请求配置：`{report['configuration_source']['path']}`；只覆盖 `layerwise_offload=false`。", '']
    return '\n'.join(lines)


def build_report(measured_dir, output_dir):
    from rooflang.programs.experiments.run_and_export import run_and_export
    from rooflang.programs.models.minimax_h3 import create_workflow
    from rooflang.programs.presets.ascend910b import Ascend910B3

    measured = load_measurements(measured_dir)
    output_dir = Path(output_dir)
    config_path = ROOT / 'examples/minimax_h3_ref2va_910b_sp1.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    config['layerwise_offload'] = False
    report = {
        'schema_version': 1, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'analytical_device_kernel_sum_without_H2D_or_host_gaps',
        'measured_files': measured['files'], 'configuration_source': _digest(config_path),
        'configuration_overrides': {'layerwise_offload': False},
        'hardware': 'Ascend910B3_1Card', 'kernel_efficiency_calibrated': False,
        'comparison_assumptions': ['Profiler capture matches the bundled Ref2VA 124-frame 768x1344 request',
                                   'Profiler device is Ascend910B3; CSV does not encode device identity'],
        'stages': {},
    }
    for stage in STAGES:
        hardware = Ascend910B3(1)
        workflow = create_workflow(hardware, config)
        results, _ = run_and_export(workflow, model_name='minimax_h3',
            hardware_name='Ascend910B3_1Card', stage=stage, config=config, hardware=hardware,
            output_dir=output_dir / 'simulation' / stage)
        simulated = summarize_simulation(results[stage])
        reference = measured['stages'][stage]
        rows = []
        for op_type in sorted(reference['by_operator'].keys() | simulated['by_operator'].keys()):
            m = reference['by_operator'].get(op_type, {'calls': 0, 'kernel_total_ms': 0})
            s = simulated['by_operator'].get(op_type, {'calls': 0, 'kernel_total_ms': 0})
            rows.append({'operator': op_type, 'measured_calls': m['calls'], 'simulated_calls': s['calls'],
                         'call_delta': s['calls'] - m['calls'], 'measured_ms': m['kernel_total_ms'],
                         'simulated_ms': s['kernel_total_ms'],
                         'time_error_pct': signed_error_pct(s['kernel_total_ms'], m['kernel_total_ms'])})
        report['stages'][stage] = {
            'measured': reference, 'simulated': simulated, 'operators': rows,
            'kernel_time_error_pct': signed_error_pct(simulated['kernel_total_ms'], reference['kernel_total_ms']),
            'call_delta': simulated['kernel_count'] - reference['kernel_count'],
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'decode_validation.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    path = output_dir / 'decode_validation.md'
    path.write_text(_markdown(report), encoding='utf-8')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measured-dir', type=Path, default=ROOT / 'measured')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/minimax_h3_decode_validation')
    args = parser.parse_args()
    print(build_report(args.measured_dir, args.output_dir).resolve())


if __name__ == '__main__':
    main()
