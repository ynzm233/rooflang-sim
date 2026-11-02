import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def validation():
    path = Path(__file__).parents[1] / 'scripts/minimax_h3_decode_validation.py'
    spec = importlib.util.spec_from_file_location('decode_validation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def measured(tmp_path):
    summaries = []
    for stage in ('audio_vae_decode', 'video_vae_decode'):
        row = dict(stage='h3/' + stage, dit_step='', kernel_name='kernel',
                   op_type='Add', input_shapes='"2;2"', input_dtypes='FLOAT;FLOAT',
                   output_shapes='"2"', output_dtypes='FLOAT', calls='2',
                   valid_calls='2', invalid_duration_count='0', total_ms='4',
                   avg_ms='2', min_ms='1', max_ms='3')
        write_csv(tmp_path / (stage + '.csv'), [row])
        summaries.append(dict(stage='h3/' + stage, dit_step='', marker_calls='1',
                              host_total_ms='1', kernel_count='2',
                              invalid_duration_count='0', kernel_total_ms='4',
                              device_busy_ms='4', device_span_ms='7'))
    write_csv(tmp_path / 'stage_summary.csv', summaries)
    return tmp_path


def test_measurement_totals_are_leaf_kernel_time_not_host_or_span(validation, measured):
    result = validation.load_measurements(measured)
    video = result['stages']['video_vae_decode']
    assert video['kernel_total_ms'] == 4
    assert video['device_span_ms'] == 7
    assert video['host_total_ms'] == 1
    assert video['kernel_count'] == 2
    assert video['by_operator']['Add'] == {'calls': 2, 'kernel_total_ms': 4}
    assert len(result['files']['video_vae_decode.csv']['sha256']) == 64


@pytest.mark.parametrize('field,value', [
    ('valid_calls', '1'), ('invalid_duration_count', '1'), ('total_ms', 'nan'),
    ('calls', '-2'), ('calls', '2.0'), ('avg_ms', '3'), ('op_type', ''),
])
def test_bad_operator_data_is_rejected(validation, measured, field, value):
    path = measured / 'audio_vae_decode.csv'
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    rows[0][field] = value
    write_csv(path, rows)
    with pytest.raises(ValueError, match='audio_vae_decode.csv'):
        validation.load_measurements(measured)


def test_stage_summary_must_reconcile(validation, measured):
    path = measured / 'stage_summary.csv'
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    rows[0]['kernel_count'] = '3'
    write_csv(path, rows)
    with pytest.raises(ValueError, match='reconcile'):
        validation.load_measurements(measured)


def test_duplicate_kernel_signature_is_rejected(validation, measured):
    path = measured / 'audio_vae_decode.csv'
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    write_csv(path, rows + rows)
    with pytest.raises(ValueError, match='duplicate'):
        validation.load_measurements(measured)


def test_graph_repetitions_count_work_once_per_kernel_and_per_invocation(validation):
    class Kernel:
        csv_metadata = {'operator': 'Add'}

    kernel = Kernel()
    entry = SimpleNamespace(kernel=kernel, start_us=0, end_us=2000)
    stage = SimpleNamespace(graphs=[SimpleNamespace(
        result=SimpleNamespace(trace=[entry, entry], total_time_us=2000), multiplier=3)])
    result = validation.summarize_simulation(stage)
    assert result['kernel_count'] == 3
    assert result['kernel_total_ms'] == 6
    assert result['by_operator']['Add']['calls'] == 3


def test_error_is_signed_without_pass_threshold(validation):
    assert validation.signed_error_pct(3, 4) == -25
    assert validation.signed_error_pct(6, 4) == 50
    assert validation.signed_error_pct(1, 0) is None


def test_zero_measured_time_is_rendered_as_undefined_error(validation):
    measured = dict(kernel_total_ms=0, kernel_count=1, host_total_ms=0,
                    device_busy_ms=0, device_span_ms=0)
    report = {'stages': {'audio_vae_decode': {
        'measured': measured, 'simulated': {'kernel_total_ms': 1, 'kernel_count': 1},
        'kernel_time_error_pct': None, 'operators': [],
    }}, 'configuration_source': {'path': 'config.json'}}
    assert '未定义' in validation._markdown(report)
