import pytest

from rooflang.programs.models.minimax_h3.workflow import resolve_config


def test_removed_decode_model_cannot_be_reenabled():
    with pytest.raises(ValueError, match='decode_model'):
        resolve_config({'decode_model': 'legacy'})


@pytest.mark.parametrize('backend', ['auto', 'flash_attention'])
def test_decode_uses_observed_attention_and_mixed_storage_independently_of_dit_backend(backend):
    from rooflang.programs.models.minimax_h3.bridge import build_component
    from rooflang.programs.models.minimax_h3.workflow import _load
    from rooflang.programs.presets.ascend910b import Ascend910B3

    cfg, _, _ = resolve_config({'sp': 1})
    mm = cfg.runtime_multimodal_config()
    mm['attention_backend'] = backend
    graph, _ = build_component('video_vae_decoder', _load('architecture.json'),
                               mm, 1797, Ascend910B3(1))
    kernels = graph.topological_sort()
    attention = [k for k in kernels if k.csv_metadata['operator'] == 'FlashAttentionScore']
    assert len(attention) == 36
    assert all(set(k.flops_by_dtype) == {'fp16:cube', 'fp32:vector'} for k in attention)
    assert not any('FusedInferAttention' in k.csv_metadata['operator'] for k in kernels)
    assert len({k.csv_metadata['layer_idx'] for k in attention}) == 36


def test_decode_does_not_invent_missing_fp32_hardware_throughput():
    from rooflang.programs.models.minimax_h3.workflow import create_workflow
    from rooflang.programs.models.minimax_h3.hardware import A5PRServer

    workflow = create_workflow(A5PRServer(), {})
    with pytest.raises(ValueError, match='fp32:cube'):
        workflow.run('audio_vae_decode')


def test_video_workflow_counts_tiles_and_request_stitch_without_duplicating_weights():
    from rooflang.programs.models.minimax_h3.workflow import create_workflow
    from rooflang.programs.presets.ascend910b import Ascend910B3

    workflow = create_workflow(Ascend910B3(1), {'sp': 1, 'execution_profile': 'sglang_v0_5_21',
                                               'layerwise_offload': False})
    stage = workflow.run('video_vae_decode')['video_vae_decode']
    assert {g.label: g.multiplier for g in stage.graphs} == {
        'video_vae_decode': 196, 'video_vae_decode_stitch': 1}
    kernels = [{entry.kernel for entry in graph.result.trace} for graph in stage.graphs]
    assert sum(len(ops) * g.multiplier for ops, g in zip(kernels, stage.graphs)) == 459714
    assert sum(k.csv_metadata['source_weight_saved'] for k in kernels[1]) == 0
    assert stage.metrics['total_weight_bytes'] == sum(
        k.csv_metadata['source_weight_saved'] for k in kernels[0])
    assert workflow.metadata['stage_precision']['video_vae_decode'] == 'mixed_fp16_fp32'
