"""Rerun the seven-stage accuracy-report scenario without changing presets."""
import argparse
import csv
import json
from pathlib import Path

from rooflang.programs.models.minimax_h3.workflow import create_workflow
from rooflang.programs.presets.ascend910b import Ascend910B3
from rooflang.programs.experiments.run_and_export import run_and_export

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error('choose a new output directory to preserve previous results')
    requests = ROOT / 'src/rooflang/programs/models/minimax_h3/requests'
    request = json.loads((requests / 'ref2va_official_d21241f.json').read_text())
    evidence = json.loads((requests / 'ref2va_official_d21241f.provenance.json').read_text())
    shape = evidence['resolved_workload']
    config = {
        'execution_profile': 'sglang_v0_5_21', 'workflow': 'ref2va', 'sp': 1,
        'num_inference_steps': shape['denoising_updates'], 'fps': shape['fps'],
        'prompt': request['prompt'], 'text_condition_rows': shape['text_condition_rows'],
        'target': {'frames': shape['target_frames'], 'height': shape['target_height'],
                   'width': shape['target_width']},
        'references': [
            {'kind': 'video', 'frames': shape['target_frames'], 'height': shape['target_height'],
             'width': shape['target_width'], 'audio_seconds': shape['target_frames'] / shape['fps']},
            {'kind': 'audio', 'audio_seconds': shape['target_frames'] / shape['fps']},
        ],
        'request_source': evidence['request_source'],
        'layerwise_offload': {'prefetch_size': 1, 'resident_layers': 0},
    }
    hardware = Ascend910B3(1)
    hardware.metadata['simulation_assumptions'].update(
        cube_efficiency=0.65, vector_efficiency=0.65,
        cube_memory_efficiency=0.60, vector_memory_efficiency=0.60,
        calibration='Fixed accuracy-report configuration: compute 0.65, operator HBM 0.60; '
                    'communication/DMA unchanged. Generic transpose model is uncalibrated.')
    for device in hardware.nodes:
        if getattr(device, 'kind', '') != 'gpu':
            continue
        for dtype, peak in hardware.metadata['derived_specification']['cube_tflops'].items():
            device.tflops[dtype] = device.tflops[f'{dtype}:cube'] = peak * 0.65
        for dtype, peak in hardware.metadata['derived_specification']['vector_tflops'].items():
            device.tflops[f'{dtype}:vector'] = peak * 0.65
    workflow = create_workflow(hardware, config)
    assert workflow.metadata['resolved_shape_summary']['packed_rows'] == shape['packed_rows']
    run_and_export(workflow, model_name='minimax_h3', hardware_name='Ascend910B3',
                   output_dir=output, config=config, hardware=hardware)
    with (output / 'stage_metrics.csv').open(newline='') as stream:
        stages = list(csv.DictReader(stream))
    result = {
        'scenario': 'Official Ref2VA, Ascend910B3, SP1, 49 updates, all seven stages rerun.',
        'method': 'Current model; fixed report efficiencies compute 0.65 / operator HBM 0.60; '
                  'no measured latency inputs or transpose fitting.',
        'plan': {k: v for k, v in workflow.metadata.items() if not k.startswith('_')},
        'stages': stages,
        'experiment_stage_seconds': {s['name']: float(s['execution_time_ms']) / 1000 for s in stages},
        'experiment_total_seconds': sum(float(s['execution_time_ms']) for s in stages) / 1000,
        'hardware': hardware.metadata,
    }
    (output / 'experiment_result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    (output / 'result.md').write_text(
        '# MiniMax-H3 最新模型完整重跑\n\n'
        '七阶段均由当前代码重新运行。配置为 910B3 单卡/SP1、49 步、计算效率 0.65、'
        '算子 HBM 效率 0.60，沿用已确认请求几何与 layerwise offload。'
        '音频转置卷积采用通用相位分块模型，无实测拟合。\n\n'
        + '\n'.join(f"- {s['name']}: {float(s['execution_time_ms']):.9f} ms" for s in stages)
        + '\n\n详细配置、代码版本与原始结果见 config.snapshot.json、provenance.json 和各阶段 CSV。\n',
        encoding='utf-8')
    print(json.dumps(result['experiment_stage_seconds']))


if __name__ == '__main__':
    main()
