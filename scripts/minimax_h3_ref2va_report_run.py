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
    parser.add_argument('--transformer-fia-compute-efficiency', type=float,
                        help='Absolute Cube and Vector efficiency for DiT denoise FIA only')
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
        'efficiency': json.loads((ROOT / 'examples/minimax_h3_ref2va_910b_sp1.json').read_text())['efficiency'],
    }
    if args.transformer_fia_compute_efficiency is not None:
        config['transformer_fia_compute_efficiency'] = args.transformer_fia_compute_efficiency
    hardware = Ascend910B3(1)
    assumptions = hardware.metadata['simulation_assumptions']
    workflow = create_workflow(hardware, config)
    efficiencies = workflow.hardware.metadata['efficiency']
    assert workflow.metadata['resolved_shape_summary']['packed_rows'] == shape['packed_rows']
    run_and_export(workflow, model_name='minimax_h3', hardware_name='Ascend910B3',
                   output_dir=output, config=config, hardware=hardware)
    with (output / 'stage_metrics.csv').open(newline='') as stream:
        stages = list(csv.DictReader(stream))
    result = {
        'scenario': 'Official Ref2VA, Ascend910B3, SP1, 49 updates, all seven stages rerun.',
        'method': 'Current model efficiency configuration and Ascend910B3 preset; '
                  f"Cube compute {efficiencies['cube_efficiency']:g}, "
                  f"Vector compute {efficiencies['vector_efficiency']:g}; "
                  f"Cube operator HBM {efficiencies['cube_memory_efficiency']:g}, "
                  f"Vector operator HBM {efficiencies['vector_memory_efficiency']:g}; "
                  f"global HBM {assumptions['memory_efficiency']:g}; "
                  'no measured latency inputs or transpose fitting.',
        'plan': {k: v for k, v in workflow.metadata.items() if not k.startswith('_')},
        'stages': stages,
        'experiment_stage_seconds': {s['name']: float(s['execution_time_ms']) / 1000 for s in stages},
        'experiment_total_seconds': sum(float(s['execution_time_ms']) for s in stages) / 1000,
        'hardware': workflow.hardware.metadata,
    }
    override_note = ''
    if args.transformer_fia_compute_efficiency is not None:
        efficiency = args.transformer_fia_compute_efficiency
        result['method'] += (
            f' DiT denoise FIA uses absolute Cube/Vector '
            f'compute efficiency {efficiency:g}; global compute efficiency is replaced, '
            'not multiplied. Text-refiner FIA, Qwen FIA and all memory efficiencies retain their current settings.')
        override_note = (
            f'其中 DiT 去噪的 FIA，Cube/Vector 计算效率单独设为 {efficiency:g}，'
            '覆盖全局计算效率而不叠乘；文本精炼 FIA、Qwen FIA 和各项访存效率沿用当前设置。')
    (output / 'experiment_result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    (output / 'result.md').write_text(
        '# MiniMax-H3 最新模型完整重跑\n\n'
        '七阶段均由当前代码重新运行。配置为 910B3 单卡/SP1、49 步，采用示例模型效率配置：'
        f"Cube 计算效率 {efficiencies['cube_efficiency']:g}、"
        f"Vector 计算效率 {efficiencies['vector_efficiency']:g}、"
        f"Cube 算子 HBM 效率 {efficiencies['cube_memory_efficiency']:g}、"
        f"Vector 算子 HBM 效率 {efficiencies['vector_memory_efficiency']:g}、"
        f"全局 HBM 效率 {assumptions['memory_efficiency']:g}。"
        + override_note
        + '沿用已确认请求几何与 layerwise offload。'
        '音频转置卷积采用通用相位分块模型，无实测拟合。\n\n'
        + '\n'.join(f"- {s['name']}: {float(s['execution_time_ms']):.9f} ms" for s in stages)
        + '\n\n详细配置、代码版本与原始结果见 config.snapshot.json、provenance.json 和各阶段 CSV。\n',
        encoding='utf-8')
    print(json.dumps(result['experiment_stage_seconds']))


if __name__ == '__main__':
    main()
