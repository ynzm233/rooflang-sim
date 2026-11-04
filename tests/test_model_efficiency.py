"""Public model efficiency configuration and numerical invariants."""
from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest

from rooflang.programs.models.magi2_preview.workflow import create_workflow as magi_workflow
from rooflang.programs.models.magi2_preview.bridge import AnalyticalKernel
from rooflang.programs.models.magi2_preview.operators import RmsNormOperator, QuantConfig
from rooflang.programs.models.minimax_h3.workflow import create_workflow as h3_workflow
from rooflang.programs.models.minimax_h3.hardware import A5PRServer
from rooflang.programs.models.magi2_preview._cost.npu_fused_infer_attention_score import (
    NpuFusedInferAttentionScoreOperator,
)
from rooflang.programs.presets.ascend910b import Ascend910B3
from rooflang.programs.presets.b300 import B300Cluster
from rooflang.runtime.efficiency import (
    communication_efficiency, prepare_hardware, resolve_efficiency,
)
from rooflang.runtime.reproducibility import config_snapshot, hardware_snapshot
from rooflang.runtime.metrics_export import export_metrics


SCALARS = (
    'cube_efficiency', 'vector_efficiency', 'cube_memory_efficiency',
    'vector_memory_efficiency', 'intra_node_efficiency',
)
MODEL_CASES = [
    pytest.param(h3_workflow, Ascend910B3,
                 {'num_inference_steps': 1, 'sp': 1}, 'transformer_text_refiner', id='h3'),
    pytest.param(magi_workflow, lambda: B300Cluster(n_nodes=1),
                 {'workflow': 't2v', 'resolution': '272p', 'preview_steps': 1},
                 'preview_denoise', id='magi'),
]


def test_default_partial_and_mutable_configuration_are_independent():
    expected = {**dict.fromkeys(SCALARS, 1.0), 'collective_efficiencies': {}}
    assert resolve_efficiency(None) == resolve_efficiency({}) == expected
    supplied = {'cube_efficiency': .5, 'collective_efficiencies': {'4': .25, '2': .5}}
    original = deepcopy(supplied)
    resolved = resolve_efficiency(supplied)
    assert resolved['cube_efficiency'] == .5
    assert all(resolved[k] == 1 for k in SCALARS if k != 'cube_efficiency')
    assert list(resolved['collective_efficiencies']) == [4, 2]
    resolved['collective_efficiencies'][4] = 1
    assert supplied == original
    assert resolve_efficiency({}) == expected


@pytest.mark.parametrize('field', SCALARS)
@pytest.mark.parametrize('value', [0, -.1, 1.01, float('nan'), float('inf'), True, None, '0.5'])
def test_invalid_scalar_efficiency_is_rejected(field, value):
    with pytest.raises(ValueError, match='efficiency'):
        resolve_efficiency({field: value})


@pytest.mark.parametrize('value', [False, 0, 'auto', [], {'memory_efficiency': .8},
                                  {'cube_efficieny': .5}])
def test_invalid_efficiency_object_is_rejected(value):
    with pytest.raises(ValueError, match='efficiency'):
        resolve_efficiency(value)


@pytest.mark.parametrize('table', [None, [], {0: .5}, {-1: .5}, {True: .5},
                                  {'many': .5}, {'2.5': .5}, {2: 0}, {2: True},
                                  {2: float('nan')}, {2: .5, '2': .6}])
def test_invalid_collective_table_is_rejected(table):
    with pytest.raises(ValueError, match='collective'):
        resolve_efficiency({'collective_efficiencies': table})


def test_prepared_hardware_keeps_physical_defaults_and_runs_isolated():
    hardware = A5PRServer(4)
    before = hardware_snapshot(hardware)
    efficiency = resolve_efficiency({'cube_efficiency': .5, 'vector_efficiency': .25})
    configured = prepare_hardware(hardware, efficiency)
    device = next(n for n in configured.nodes if n.kind == 'gpu')
    assert device.tflops['fp16:cube'] == 216
    assert device.tflops['fp16:vector'] == 13.5
    memory = configured.find_local_memory(device)
    assert configured.find_fabric(device, memory).src_to_dst_bandwidth_gbs == 1280
    assert hardware_snapshot(hardware) == before
    assert configured is not hardware
    unity = prepare_hardware(hardware, resolve_efficiency({}))
    assert next(n for n in unity.nodes if n.kind == 'gpu').tflops['fp16:cube'] == 432
    assert configured.metadata['efficiency']['cube_efficiency'] == .5


@pytest.mark.parametrize('table, expected', [
    ({'2': .25, '4': .5, '8': 1.0}, .125),
    ({'4': .5, '2': .25, '8': 1.0}, .25),
    ({}, .5),
])
def test_collective_nearest_key_ties_preserve_insertion_order(table, expected):
    hardware = prepare_hardware(A5PRServer(4), resolve_efficiency({
        'intra_node_efficiency': .5, 'collective_efficiencies': table,
    }))
    devices = sorted((n for n in hardware.nodes if n.kind == 'gpu'), key=lambda n: n.name)
    assert communication_efficiency(hardware, devices[:3]) == expected
    assert communication_efficiency(hardware, devices[:3], point_to_point=True) == .5


@pytest.mark.parametrize('create, make_hardware, config, stage', MODEL_CASES)
def test_model_efficiency_changes_time_not_work_or_shared_hardware(
    create, make_hardware, config, stage, tmp_path,
):
    hardware = make_hardware()
    before = hardware_snapshot(hardware)
    base = create(hardware, config).run(stage)[stage]
    supplied = {**config, 'efficiency': {
        'cube_efficiency': .5, 'vector_efficiency': .25,
        'cube_memory_efficiency': .4, 'vector_memory_efficiency': .2,
    }}
    original = deepcopy(supplied)
    workflow = create(hardware, supplied)
    changed = workflow.run(stage)[stage]
    seen = set()
    for old_graph, new_graph in zip(base.graphs, changed.graphs, strict=True):
        assert old_graph.multiplier == new_graph.multiplier
        assert {m.name:v for m,v in old_graph.result.peak_memory.items()} == {
            m.name:v for m,v in new_graph.result.peak_memory.items()}
        for old, new in zip(old_graph.result.trace, new_graph.result.trace, strict=True):
            assert old.kernel.flops == new.kernel.flops
            assert old.kernel.transferred_bytes == new.kernel.transferred_bytes
            operator = getattr(new.kernel, 'operator', None)
            if operator is None:
                continue
            component = operator.get_compute_component()
            if component is None:
                continue
            assert new.memory_time_us == pytest.approx(
                old.memory_time_us / new.kernel.local_memory_efficiency)
            if hasattr(operator, 'get_memory_bytes_by_unit'):
                assert new.kernel.csv_metadata['memory_efficiency_basis'] == 'operator_memory_bytes_by_unit'
            unit = component.unit_type
            if unit in ('cube', 'vector') and old.compute_time_us:
                efficiency = .5 if unit == 'cube' else .25
                assert new.compute_time_us == pytest.approx(old.compute_time_us / efficiency)
                seen.add(unit)
            if unit == 'mixed' and not hasattr(operator, 'get_memory_bytes_by_unit'):
                assert new.kernel.local_memory_efficiency == .4
                seen.add('mixed_memory')
    assert {'cube', 'vector'} <= seen
    if create is magi_workflow:
        assert 'mixed_memory' in seen
    assert supplied == original
    assert hardware_snapshot(hardware) == before
    snapshot = config_snapshot(workflow, model_name='test', hardware_name='test',
                               stage=stage, config=supplied, hardware=hardware)
    assert snapshot['hardware']['metadata']['efficiency']['cube_efficiency'] == .5
    resolved = snapshot['resolved_config']
    assert resolved.get('workload', resolved)['efficiency']['vector_efficiency'] == .25
    model = 'minimax_h3' if create is h3_workflow else 'magi2_preview'
    export_metrics({stage: changed}, tmp_path, model_name=model,
                   hardware_name='test', metadata=workflow.metadata)
    with (tmp_path / f'{stage}.csv').open(newline='') as file:
        rows = list(csv.DictReader(file))
    assert {'local_memory_efficiency', 'memory_bytes_by_unit',
            'memory_efficiency_by_unit', 'memory_efficiency_basis'} <= rows[0].keys()
    assert any(row['memory_efficiency_by_unit'] for row in rows)
    if create is magi_workflow:
        assert any(row['compute_effective_tflops_by_unit'] for row in rows)
        with (tmp_path / f'{model}_plan_metrics.csv').open(newline='') as file:
            plan, = csv.DictReader(file)
        assert plan['compute_peak_basis'] == 'hardware_dtype_peak_with_unit_efficiency'


def test_fia_override_remains_absolute_and_refiner_is_outside_its_scope():
    hardware = Ascend910B3(2)
    config = {'workflow': 'ref2va', 'sp': 2, 'num_inference_steps': 1,
              'efficiency': {'cube_efficiency': .5, 'vector_efficiency': .5}}
    base_workflow = h3_workflow(hardware, config)
    overridden = h3_workflow(hardware, {**config, 'transformer_fia_compute_efficiency': .3})
    stage = 'transformer_denoise'
    base = base_workflow.run(stage)[stage].graphs[0].result
    changed = overridden.run(stage)[stage].graphs[0].result
    count = 0
    for old, new in zip(base.trace, changed.trace, strict=True):
        if type(getattr(new.kernel, 'operator', None)) is NpuFusedInferAttentionScoreOperator:
            assert new.compute_time_us == pytest.approx(old.compute_time_us * .5 / .3)
            count += 1
        else:
            assert new.compute_time_us == old.compute_time_us
        assert new.memory_time_us == old.memory_time_us
        assert new.kernel.flops == old.kernel.flops
    assert count > 0
    stage = 'transformer_text_refiner'
    before = base_workflow.run(stage)[stage].graphs[0].result
    after = overridden.run(stage)[stage].graphs[0].result
    assert after.total_time_us == before.total_time_us


def test_magi_retains_dtype_peak_basis_on_npu_when_unit_efficiencies_are_separated():
    efficiency = resolve_efficiency({'cube_efficiency': .8, 'vector_efficiency': .4})
    hardware = prepare_hardware(Ascend910B3(), efficiency)
    device = hardware.devices[0]
    operator = RmsNormOperator(batch_seq=32, dim=128,
                               quant_config=QuantConfig(compute_dtype='fp16'))
    kernel = AnalyticalKernel(operator, {}, 'norm', efficiency=efficiency)
    # MAGI historically uses the dtype (Cube) peak even for vector operators.
    assert kernel.scheduled_compute_time_us(device) == pytest.approx(
        operator.get_compute_component().total_flops() / (294.912 * .4 * 1e6))
    assert kernel.csv_metadata['compute_effective_tflops_by_unit']['fp16:vector'] == pytest.approx(
        294.912 * .4)


@pytest.mark.parametrize('case, create, make_hardware, config, selected', [
    ('h3_b3', h3_workflow, lambda: Ascend910B3(2),
     {'workflow': 'ref2va', 'sp': 2, 'num_inference_steps': 1,
      'efficiency': {'cube_efficiency': .8, 'vector_efficiency': .6,
                     'cube_memory_efficiency': .8, 'vector_memory_efficiency': .6}}, 'all'),
    ('h3_a5', h3_workflow, lambda: A5PRServer(4),
     {'workflow': 'fl2va', 'sp': 4, 'num_inference_steps': 1,
      'efficiency': {'cube_efficiency': .7, 'vector_efficiency': .6,
                     'intra_node_efficiency': .7,
                     'collective_efficiencies': {'2': .142857, '4': .428571, '8': 1.0}}},
     'transformer_denoise'),
    ('magi_b300', magi_workflow, lambda: B300Cluster(n_nodes=1),
     {'workflow': 't2v', 'resolution': '272p', 'preview_steps': 1}, 'all'),
])
def test_pre_migration_simulation_baseline_is_preserved(case, create, make_hardware, config, selected):
    baseline = json.loads((Path(__file__).parent / 'fixtures' /
                           'model_efficiency_baseline.json').read_text())['cases'][case]
    results = create(make_hardware(), config).run(selected)
    assert results.keys() == baseline.keys()
    for stage, result in results.items():
        for graph, expected in zip(result.graphs, baseline[stage], strict=True):
            actual = graph.result
            assert graph.label == expected['label']
            assert graph.multiplier == expected['multiplier']
            assert actual.total_time_us == pytest.approx(expected['time_us'], rel=1e-12)
            assert len(actual.trace) == expected['trace_count']
            assert sum(e.kernel.flops for e in actual.trace) == pytest.approx(
                expected['total_flops'], rel=1e-12)
            assert sum(e.kernel.transferred_bytes for e in actual.trace) == pytest.approx(
                expected['transferred_bytes'], rel=1e-12)
            assert {m.name:v for m,v in actual.peak_memory.items()} == expected['peak_memory']
