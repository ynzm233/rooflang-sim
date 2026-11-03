"""MiniMax-H3 workflow interface, configuration, and capacity checks."""
import pytest
from rooflang.programs.models.minimax_h3 import create_workflow
from rooflang.programs.models.minimax_h3.hardware import A5PRServer
from rooflang.programs.presets.ascend910b import Ascend910B3
from rooflang.programs.models.minimax_h3.workflow import resolve_config, video_vae_tiles


def test_shapes_and_tiling():
    fl,_,_=resolve_config({});ref,_,_=resolve_config({'workflow':'ref2va'})
    assert fl.shape_summary().packed_rows==41763
    assert ref.shape_summary().packed_rows==81532
    assert video_vae_tiles(768,1344)==(256,256,28)
    assert video_vae_tiles(32,64)==(32,64,1)
    with pytest.raises(ValueError):resolve_config({'workflow':'other'})
    with pytest.raises(ValueError):resolve_config({'target':{'frames':125}})
    with pytest.raises(ValueError):resolve_config({'sp':3})
    with pytest.raises(ValueError):resolve_config({'cp':2})


def test_runtime_parameters_change_work_and_capacity():
    w=create_workflow(Ascend910B3(2),{'sp':2,'num_inference_steps':2,'assume_stage_unload':False})
    results=w.run()
    assert w.metadata['transformer_forward_count']==2
    assert results['transformer_denoise'].graphs[0].multiplier==2
    assert not w.metadata['is_valid']
    assert w.metadata['memory_validity_basis']=='all_resident_memory'
    assert w.metadata['_resolved_config']['workload']['num_inference_steps']==2


def test_partial_stage_does_not_claim_full_workflow_capacity():
    hardware=A5PRServer()
    for node in hardware.nodes:
        if node.kind=='hbm': node.capacity_gb=2
    w=create_workflow(hardware,{'assume_stage_unload':False})
    results=w.run('qwen_vision_encode')
    assert set(results)=={'qwen_vision_encode'}
    assert w.metadata['request_count'] is None
    assert w.metadata['transformer_forward_count']==0
    assert w.metadata['memory_validity_basis']=='selected_stage_memory'
    assert not w.metadata['fits_peak_stage_memory']
    assert not w.metadata['is_valid']
    assert w.metadata['all_resident_memory_bytes'] is None


@pytest.mark.parametrize('config', [
    {'compute_dtype':'fp32'}, {'weight_storage_dtype':'fp8'}, {'batch_size':True},
    {'num_inference_steps':1.5}, {'assume_stage_unload':'false'},
    {'text_condition_rows':4.5}, {'target':{'height':768.5}},
    {'prompt':'Generate a  cinematic video.'}, {'prompt':42,'text_condition_rows':6113},
    {'fps':30}, {'fps':24.0},
])
def test_rejects_unsupported_or_coerced_configuration(config):
    with pytest.raises(ValueError): create_workflow(A5PRServer(),config)


def test_prompt_provenance_preserves_whitespace_and_explicit_processor_rows():
    from rooflang.runtime.reproducibility import config_snapshot

    config = {'workflow':'ref2va', 'prompt':'Generate a  cinematic video.',
              'text_condition_rows':6113, 'fps':24}
    hardware = A5PRServer()
    workflow = create_workflow(hardware, config)
    snapshot = config_snapshot(workflow, model_name='minimax_h3',
        hardware_name='A5_pr_server', stage='all', config=config, hardware=hardware)
    workload = snapshot['resolved_config']['workload']
    assert workload['prompt'] == config['prompt']
    assert workload['fps'] == 24
    assert workload['workflows']['ref2va']['text_condition_rows'] == 6113
    assert workflow.metadata['prompt_len'] == 6113


def test_hardware_validation():
    for n in (0,9,True,1.5):
        with pytest.raises(ValueError): A5PRServer(n)
    with pytest.raises(ValueError):create_workflow(A5PRServer(2),{'sp':4})
    hardware=A5PRServer()
    gpu=next(n for n in hardware.nodes if n.kind=='gpu')
    gpu.tflops.pop('fp16:vector')
    with pytest.raises(ValueError):create_workflow(hardware,{})


def test_partial_repeated_run_clears_full_verdict_and_preserves_stage_memory():
    w=create_workflow(Ascend910B3(2),{'sp':2,'workflow':'ref2va','num_inference_steps':1})
    full=w.run()
    expected=full['audio_vae_decode'].metrics['peak_memory_bytes']
    partial=w.run('audio_vae_decode')
    assert partial['audio_vae_decode'].metrics['peak_memory_bytes']==expected
    assert w.metadata['all_resident_memory_bytes'] is None
    assert w.metadata['memory_scope']=='selected_stages_only'
    assert w.metadata['request_count'] is None
