"""Reproduce the report's audio scenario with uncalibrated transpose tiling.

Prediction finishes before any baseline or measured latency is loaded. Old
artifacts are read-only; output must be a new directory. No preset is modified.
"""
import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from rooflang.programs.models.minimax_h3.bridge import build_component
from rooflang.programs.presets.ascend910b import Ascend910B3
from rooflang.runtime.metrics_export import export_metrics
from rooflang.runtime.simulator import Simulator

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'outputs/minimax_h3_ref2va_910b3_sp1_fia_compute065_mem060_experiment'


def rows(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(shape, measured=False):
    """Match exact physical boundary multisets; never match on latency."""
    if measured:
        def shapes(text):
            return [tuple(map(int, s.split(','))) if s else ()
                    for s in text.strip('"').split(';')]
        inputs = list(zip(shapes(shape['input_shapes']), shape['input_dtypes'].split(';')))
        outputs = list(zip(shapes(shape['output_shapes']), shape['output_dtypes'].split(';')))
    else:
        dtype = {'fp32': 'FLOAT', 'int32': 'INT32', 'int64': 'INT64'}
        inputs = [(tuple(s), dtype[d]) for s, d in zip(
            shape['input_shapes'] + shape['weight_shapes'],
            shape['input_dtypes'] + shape['weight_dtypes'])]
        outputs = [(tuple(s), dtype[d]) for s, d in zip(shape['output_shapes'], shape['output_dtypes'])]
    return tuple(sorted(inputs)), tuple(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error('output directory must be new; existing artifacts are preserved')

    # Original report configuration, not today's 0.70 compute preset default.
    hardware = Ascend910B3(1)
    assumptions = hardware.metadata['simulation_assumptions']
    assumptions.update(cube_efficiency=0.65, vector_efficiency=0.65,
                       cube_memory_efficiency=0.60, vector_memory_efficiency=0.60,
                       calibration='Fixed report configuration: Cube/Vector compute 0.65, '
                                   'Cube/Vector HBM 0.60; communication/DMA unchanged. '
                                   'Generic transpose algorithm is uncalibrated.')
    for device in hardware.nodes:
        if getattr(device, 'kind', '') != 'gpu':
            continue
        for dtype, peak in hardware.metadata['derived_specification']['cube_tflops'].items():
            device.tflops[dtype] = device.tflops[f'{dtype}:cube'] = peak * 0.65
        for dtype, peak in hardware.metadata['derived_specification']['vector_tflops'].items():
            device.tflops[f'{dtype}:vector'] = peak * 0.65
    mm = {'target_audio_latents_per_channel': 207, 'execution_profile': 'sglang_v0_5_21'}
    graph, placement = build_component('audio_vae_decoder', {}, mm, 207, hardware)
    result = Simulator(graph, placement, hardware).run()
    export_metrics({'audio_vae_decode': result}, output, model_name='minimax_h3',
                   hardware_name='Ascend910B3', metadata={'scope': 'audio_only_fixed_report_configuration'})

    # Evaluation only: these files cannot influence the prediction above.
    report_manifest = json.loads((ROOT / 'reports/manifest.json').read_text())
    baseline_path = BASELINE / 'audio_vae_decode.csv'
    assert sha256(baseline_path) == report_manifest['source_sha256'][baseline_path.name]
    baseline, predicted = rows(baseline_path), rows(output / 'audio_vae_decode.csv')
    assert len(baseline) == len(predicted) == len(result.trace) == 4590
    measured_path = ROOT / 'measured/audio_vae_decode.csv'
    measured = defaultdict(lambda: {'calls': 0, 'total_ms': 0.0})
    for row in rows(measured_path):
        if row['op_type'] == 'Conv2DTranspose':
            group = measured[signature(row, measured=True)]
            group['calls'] += int(row['calls'])
            group['total_ms'] += float(row['total_ms'])
    groups = {}
    transpose_old = transpose_new = other_old = other_new = 0.0
    checked = 0
    numeric = ('compute_ms', 'memory_ms', 'execution_time_ms', 'memory_traffic_bytes',
               'flops', 'weight_read', 'weight_saved', 'activation_peak_bytes_proxy')
    for old, new, entry in zip(baseline, predicted, result.trace):
        assert old['node_name'] == new['node_name'] == entry.kernel.csv_metadata['node_name']
        assert old['operator'] == new['operator']
        assert json.loads(old['shape']) == json.loads(new['shape'])
        assert old['compute_dtype'] == new['compute_dtype']
        assert float(old['graph_multiplier']) == float(new['graph_multiplier']) == 1
        assert float(old['flops']) == float(new['flops'])
        before, after = float(old['execution_time_ms']), float(new['execution_time_ms'])
        if new['operator'] != 'Conv2DTranspose':
            for field in numeric:
                assert math.isclose(float(old[field]), float(new[field]), rel_tol=1e-12, abs_tol=1e-10), (new['node_name'], field)
            checked += 1
            other_old += before
            other_new += after
            continue
        transpose_old += before
        transpose_new += after
        op = entry.kernel.operator
        key = signature(json.loads(new['shape']))
        if key not in groups:
            cost = op.tile_cost
            groups[key] = {'geometry': json.dumps(asdict(op.geometry)), 'calls': 0,
                           'old_ms': 0.0, 'new_ms': 0.0,
                           'reference_flops_per_call': cost.reference_flops,
                           'valid_flops_per_call': cost.valid_flops,
                           'executed_cube_flops_per_call': cost.executed_cube_flops,
                           'executed_vector_flops_per_call': cost.executed_vector_flops,
                           'old_traffic_bytes_per_call': float(old['memory_traffic_bytes']),
                           'new_traffic_bytes_per_call': cost.activation_bytes + cost.bias_read_bytes,
                           'compute_ms_per_call': float(new['compute_ms']),
                           'memory_ms_per_call': float(new['memory_ms'])}
        item = groups[key]
        item['calls'] += 1
        item['old_ms'] += before
        item['new_ms'] += after
    assert checked == 4456
    assert sum(item['calls'] for item in groups.values()) == 134
    assert set(groups) == set(measured)
    for key, item in groups.items():
        assert item['calls'] == measured[key]['calls']
        item['measured_ms'] = measured[key]['total_ms']
        item['relative_error'] = item['new_ms'] / item['measured_ms'] - 1
    stage_path = ROOT / 'measured/stage_summary.csv'
    stage_measured = next(float(r['device_span_ms']) for r in rows(stage_path)
                          if r['stage'] == 'h3/audio_vae_decode')
    summary = {
        'scope': 'audio_only_fixed_report_configuration', 'calibration': 'none',
        'hardware': hardware.metadata, 'multimodal_config': mm,
        'old_audio_ms': other_old + transpose_old, 'new_audio_ms': result.total_time_us / 1000,
        'measured_audio_device_span_ms': stage_measured,
        'old_transpose_ms': transpose_old, 'new_transpose_ms': transpose_new,
        'measured_transpose_kernel_sum_ms': sum(m['total_ms'] for m in measured.values()),
        'unchanged_non_transpose_rows': checked, 'non_transpose_ms': other_new,
        'leaf_count': len(result.trace), 'transpose_count': 134,
        'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in
                          (baseline_path, measured_path, stage_path, ROOT / 'reports/manifest.json')},
    }
    (output / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    shape_path = output / 'transpose_shapes.csv'
    with shape_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(next(iter(groups.values()))))
        writer.writeheader()
        writer.writerows(groups.values())
    longest = max((item for item in groups.values() if json.loads(item['geometry'])['groups'] > 1),
                  key=lambda item: json.loads(item['geometry'])['length'])
    dense_first = next(iter(groups.values()))
    text = f'''# 音频转置卷积分块模型对比

固定原报告配置：910B3 单卡，FP32 音频，latent 长度 207，计算效率 0.65，Cube/Vector HBM 效率 0.60。preset 默认值未修改。预测先执行，之后才读取实测作评估；无拟合系数。

| 范围 | 原仿真 ms | 分块仿真 ms | 实测 ms |
|---|---:|---:|---:|
| 音频阶段 | {summary['old_audio_ms']:.6f} | {summary['new_audio_ms']:.6f} | {stage_measured:.6f} |
| 134 个转置卷积 | {transpose_old:.6f} | {transpose_new:.6f} | {summary['measured_transpose_kernel_sum_ms']:.6f} |
| 4456 个其他算子 | {other_old:.6f} | {other_new:.6f} | — |

音频阶段实测采用 device_span_ms，转置卷积实测采用 kernel 总和，两种口径不能直接混用。

模型修正了补齐执行工作、panel 重读、边界有效 payload、bias Vector 工作和核心尾波。空间维允许并行，固定模板为 16×16×8，不复刻未知的 CANN 实现。剩余偏差不作为调整 tile 或效率的依据。

音频阶段相对误差仍为 {(summary['new_audio_ms'] / stage_measured - 1) * 100:.2f}%；转置卷积总和相对误差仍为 {(transpose_new / summary['measured_transpose_kernel_sum_ms'] - 1) * 100:.2f}%。本次修正了通用算法的成本表达，尚未解释实测慢路径。

最长 Snake UpFilter（Cin=Cout=groups=8，Lin=165610）补齐 Cube 工作约为 reference FLOPs 的 {longest['executed_cube_flops_per_call'] / longest['reference_flops_per_call']:.2f} 倍；流量由 {longest['old_traffic_bytes_per_call'] / 1e6:.3f} MB 增至 {longest['new_traffic_bytes_per_call'] / 1e6:.3f} MB，但单次预测仅 {longest['new_ms'] / longest['calls']:.6f} ms，实测 {longest['measured_ms'] / longest['calls']:.6f} ms。通用模板允许长序列的空间 tile 并行，无法据此推断实际内核是否串行或有哪些指令、地址生成和片上带宽开销。

无跨 tile 缓存复用也会高估某些密集卷积：首个 Upsample 新预测 {dense_first['new_ms']:.6f} ms，实测 {dense_first['measured_ms']:.6f} ms。该误差按原样保留，不据此调整模板。

4590 个节点的顺序、形状、dtype、调用次数和原 FLOPs 口径一致；4456 个非转置算子的计算、流量、容量与时间均与原 CSV 一致（浮点容差 1e-10 ms / 相对 1e-12）。旧 reports、measured 和基线目录只读。

逐形状工作、流量与时间见 [transpose_shapes.csv](transpose_shapes.csv)，逐算子相位任务和波次见 [audio_vae_decode.csv](audio_vae_decode.csv) 的 transpose_schedule JSON。配置和输入 SHA256 见 [comparison.json](comparison.json)。这是本地解析模型验证，未运行 NPU。

复现（输出目录必须不存在）：

```sh
.venv/bin/python scripts/minimax_h3_audio_transpose_comparison.py --output-dir outputs/audio_transpose_tiling_recheck
```
'''
    (output / 'comparison.md').write_text(text, encoding='utf-8')
    print(json.dumps({key: value for key, value in summary.items()
                      if key.endswith('_ms') or key.endswith('_count') or key == 'unchanged_non_transpose_rows'}))


if __name__ == '__main__':
    main()
