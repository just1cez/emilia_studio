"""Consumer-visible CLI command and persistence checks."""
import json

from emilia_studio.cli import main


def test_configuration_persists_across_working_directories(tmp_path, monkeypatch, capsys):
    config_dir = tmp_path / 'settings'
    monkeypatch.setenv('EMILIA_CONFIG_DIR', str(config_dir))
    monkeypatch.delenv('EMILIA_CONFIG_TOML', raising=False)
    root = tmp_path / 'workspace'
    assert main(['config', 'init', '--root-dir', str(root)]) == 0
    assert main(['config', 'set', 'quality_port', '8205']) == 0
    monkeypatch.chdir(tmp_path)
    assert main(['config', 'show']) == 0
    assert '8205' in capsys.readouterr().out
    assert main(['config', 'init']) == 1
    assert main(['config', 'set', 'quality_port', '0']) == 1


def test_gallery_invalid_index_and_empty_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('EMILIA_OUTPUT_DIR', str(tmp_path))
    assert main(['gallery', '--open', '1']) == 1
    assert 'outside' in capsys.readouterr().err
    assert main(['gallery', '--limit', '0']) == 1


def test_machine_workflows_output_is_json(capsys):
    assert main(['workflows']) == 0
    report = json.loads(capsys.readouterr().out)
    assert 'emilia_ipadapter_highres_production.api.json' in report['canonical_workflows']


def test_unknown_download_key_fails_without_network(capsys):
    assert main(['pull', 'unknown-model']) == 1
    assert 'Unknown model' in capsys.readouterr().err


def test_noninteractive_entry_does_not_consume_input(monkeypatch):
    import io
    stream = io.StringIO('presets\n')
    monkeypatch.setattr('sys.stdin', stream)
    assert main([]) == 0
    assert stream.tell() == 0
