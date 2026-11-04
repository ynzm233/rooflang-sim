"""Historical comparison inputs are checked before simulation or output writes."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def comparison(monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'scripts/minimax_h3_audio_transpose_comparison.py'
    spec = importlib.util.spec_from_file_location('audio_transpose_comparison', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'build_component',
                        lambda *args, **kwargs: pytest.fail('simulation must not start'))
    return module


@pytest.mark.parametrize('problem', [
    'baseline', 'manifest', 'measured', 'stage_summary', 'hash_mismatch',
    'invalid_json', 'missing_hash',
])
def test_invalid_input_does_not_create_output(comparison, tmp_path, capsys, problem):
    baseline_dir = tmp_path / 'baseline'
    measured_dir = tmp_path / 'measured'
    baseline_dir.mkdir()
    measured_dir.mkdir()
    baseline = baseline_dir / 'audio_vae_decode.csv'
    baseline.write_text('original baseline bytes\n')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'source_sha256': {
        baseline.name: hashlib.sha256(baseline.read_bytes()).hexdigest()}}))
    measured = measured_dir / 'audio_vae_decode.csv'
    measured.write_text('measurement bytes\n')
    stage_summary = measured_dir / 'stage_summary.csv'
    stage_summary.write_text('stage measurement bytes\n')
    missing = {'baseline': baseline, 'manifest': manifest, 'measured': measured,
               'stage_summary': stage_summary}
    if problem in missing:
        missing[problem].unlink()
    elif problem == 'hash_mismatch':
        baseline.write_text('different baseline bytes\n')
    elif problem == 'invalid_json':
        manifest.write_text('{')
    else:
        manifest.write_text('{}')
    output = tmp_path / 'new_output'
    with pytest.raises(SystemExit) as error:
        comparison.main(['--baseline-dir', str(baseline_dir), '--manifest', str(manifest),
                         '--measured-dir', str(measured_dir), '--output-dir', str(output)])
    assert error.value.code == 2
    assert not output.exists()
    message = capsys.readouterr().err
    assert ('missing input file' if problem in missing else
            'baseline SHA256' if problem == 'hash_mismatch' else
            'cannot validate baseline') in message


def test_matching_hash_reaches_prediction_before_parsing_latency(comparison, tmp_path, monkeypatch):
    baseline = tmp_path / 'audio_vae_decode.csv'
    baseline.write_text('identity bytes only; not valid measurement CSV\n')
    (tmp_path / 'stage_summary.csv').write_text('also not valid measurement CSV\n')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'source_sha256': {
        baseline.name: hashlib.sha256(baseline.read_bytes()).hexdigest()}}))

    def stop_at_prediction(*args, **kwargs):
        raise RuntimeError('prediction started')

    monkeypatch.setattr(comparison, 'build_component', stop_at_prediction)
    monkeypatch.setattr(comparison, 'rows',
                        lambda *args: pytest.fail('latency must not be parsed before prediction'))
    output = tmp_path / 'new_output'
    with pytest.raises(RuntimeError, match='prediction started'):
        comparison.main(['--baseline-dir', str(tmp_path), '--manifest', str(manifest),
                         '--measured-dir', str(tmp_path), '--output-dir', str(output)])
    assert not output.exists()
