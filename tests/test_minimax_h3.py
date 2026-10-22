"""Whole-graph reproduction against independently rerun model_Sim snapshots."""
import csv
import importlib.util
import json
from pathlib import Path
import pytest
from rooflang.programs.models.minimax_h3 import create_workflow
from rooflang.programs.models.minimax_h3.hardware import A5PRServer
from rooflang.programs.models.minimax_h3.workflow import resolve_config, video_vae_tiles
from rooflang.programs.models.minimax_h3.bridge import component_nodes
from rooflang.runtime.metrics_export import _operator_rows, _semantic_stage, export_metrics

BASE = Path(__file__).parents[1] / 'baselines/minimax_h3'
SPEC = importlib.util.spec_from_file_location(
    'minimax_h3_baseline', Path(__file__).parents[1] / 'scripts/minimax_h3_baseline.py')
BASELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASELINE)

@pytest.mark.parametrize('workflow', ['fl2va','ref2va'])
def test_full_source_reproduction(workflow, tmp_path):
    baseline=json.loads((BASE/workflow/'summary.json').read_text())
    w=create_workflow(BASELINE.source_hardware(),{'workflow':workflow});results=w.run()
    assert w.metadata['shape_summary']==baseline['shape_summary']
    assert w.metadata['operator_graph_multipliers']==baseline['operator_graph_multipliers']
    assert w.metadata['transformer_forward_count']==50
    for key in ('persistent_tensor_bytes','indexed_known_weight_bytes','all_resident_weight_bytes',
                'all_resident_memory_bytes','peak_stage_memory_bytes','operator_graph_weight_proxy_bytes',
                'fits_peak_stage_memory','fits_all_resident_memory'):
        assert w.metadata[key]==baseline[key],key
    total=0
    for name,stage in results.items():
        rows,metrics=_semantic_stage(name,stage,'MiniMax-H3','A5_pr_server')
        expected=next(s for s in baseline['stages'] if s['name']==name)
        for key in ('execution_time_ms','compute_time_ms','memory_time_ms','comm_time_ms',
                    'total_flops','total_mem_bytes','total_weight_bytes','peak_activation_bytes','invocations'):
            assert metrics[key]==pytest.approx(expected[key],rel=1e-11,abs=1e-8),(name,key)
        total+=metrics['execution_time_ms']
        for g in stage.graphs:
            actual=list(_operator_rows(g.result,g.label,'MiniMax-H3','A5_pr_server'))
            with (BASE/workflow/(g.label+'.csv')).open() as f: expected_rows=list(csv.DictReader(f))
            assert len(actual)==len(expected_rows),g.label
            for new,old in zip(actual,expected_rows):
                assert (new['node_name'],new['operator'],new['layer_idx'])==(old['node_name'],old['operator'],int(old['layer_idx']))
                for key in ('execution_time_ms','compute_ms','memory_ms','network_ms','flops','memory_traffic_bytes','weight_saved'):
                    assert new[key]==pytest.approx(float(old[key] or 0),rel=1e-10,abs=1e-8),(g.label,new['node_name'],key)
    assert total==pytest.approx(baseline['execution_time_ms'],rel=1e-11)
    export_metrics(results,tmp_path,model_name='MiniMax-H3',hardware_name='A5_pr_server',metadata=w.metadata)
    assert (tmp_path/'stage_metrics.csv').exists()


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
    w=create_workflow(A5PRServer(),{'num_inference_steps':2,'assume_stage_unload':False})
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
    w=create_workflow(A5PRServer(),{'workflow':'ref2va','num_inference_steps':1})
    full=w.run()
    expected=full['audio_vae_decode'].metrics['peak_memory_bytes']
    partial=w.run('audio_vae_decode')
    assert partial['audio_vae_decode'].metrics['peak_memory_bytes']==expected
    assert w.metadata['all_resident_memory_bytes'] is None
    assert w.metadata['memory_scope']=='selected_stages_only'
    assert w.metadata['request_count'] is None
