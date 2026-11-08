"""Portable config discovery and writable/resource boundary regressions."""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emilia_studio import config as configuration


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key in tuple(os.environ):
        if key.startswith("EMILIA_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("EMILIA_CONFIG_DIR", str(tmp_path / "user-config"))
    monkeypatch.setattr(configuration, "get_user_data_dir", lambda: tmp_path / "user-data")
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)


def test_config_selection_precedence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr(configuration, "get_source_checkout_dir", lambda: checkout)
    (checkout / "config.toml").write_text('[services]\ncomfy_port = 8101\n', encoding="utf-8")
    assert configuration.load_config().comfy_port == 8101

    user_config = configuration.get_user_config_path()
    user_config.parent.mkdir()
    user_config.write_text('[services]\ncomfy_port = 8102\n', encoding="utf-8")
    assert configuration.load_config().comfy_port == 8102

    env_config = tmp_path / "environment.toml"
    env_config.write_text('[services]\ncomfy_port = 8103\n', encoding="utf-8")
    monkeypatch.setenv("EMILIA_CONFIG_TOML", str(env_config))
    assert configuration.load_config().comfy_port == 8103

    explicit_config = tmp_path / "explicit.toml"
    explicit_config.write_text('[services]\ncomfy_port = 8104\n', encoding="utf-8")
    loaded = configuration.load_config(explicit_config)
    assert loaded.comfy_port == 8104
    assert loaded.config_source == str(explicit_config.resolve())

    monkeypatch.setenv("EMILIA_COMFY_PORT", "8105")
    assert configuration.load_config(explicit_config).comfy_port == 8105


def test_cwd_never_selects_configuration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(configuration, "get_source_checkout_dir", lambda: None)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "config.toml").write_text('[services]\ncomfy_port = 1234\n', encoding="utf-8")
    (unrelated / "models_manifest.json").write_text("{}", encoding="utf-8")
    child = unrelated / "child"
    child.mkdir()
    for cwd in (unrelated, child):
        monkeypatch.chdir(cwd)
        loaded = configuration.load_config()
        assert loaded.comfy_port == 8188
        assert loaded.root_dir == tmp_path / "user-data"
        assert loaded.config_source == "auto-discovered"


def test_relative_toml_paths_use_config_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    directory = tmp_path / "settings"
    directory.mkdir()
    config_file = directory / "config.toml"
    values = {
        "root_dir": "../workspace", "app_dir": "runtime/app", "output_dir": "outputs",
        "temp_dir": "scratch", "workflows_dir": "templates/api",
        "visual_workflows_dir": "templates/visual", "resources_dir": "resources",
        "conda_python": "runtime/python.exe", "ollama_exe": "tools/ollama.exe",
    }
    config_file.write_text(
        "[paths]\n" + "\n".join(f'{key} = "{value}"' for key, value in values.items()),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    loaded = configuration.load_config(config_file)
    for key, value in values.items():
        assert getattr(loaded, key) == (directory / value).resolve()
    monkeypatch.setenv("EMILIA_ROOT_DIR", str(tmp_path / "environment-workspace"))
    assert configuration.load_config(config_file).root_dir == tmp_path / "environment-workspace"


def test_explicit_missing_config_does_not_fall_back(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    missing = tmp_path / "missing.toml"
    with pytest.raises(FileNotFoundError):
        configuration.load_config(missing)
    monkeypatch.setenv("EMILIA_CONFIG_TOML", str(missing))
    with pytest.raises(FileNotFoundError):
        configuration.load_config()


def test_quoted_toml_roundtrip(tmp_path: Path) -> None:
    original = configuration.StudioConfig(
        root_dir=tmp_path / "Studio's Space",
        comfy_host='host"quoted',
        ollama_base='http://localhost:11434/"quoted"\\path',
        ollama_model='model "quoted"\nsecond line\t雪',
    )
    destination = original.save_user_config(tmp_path / "settings" / "config.toml")
    loaded = configuration.load_config(destination)
    assert loaded.root_dir == original.root_dir
    assert loaded.comfy_host == original.comfy_host
    assert loaded.ollama_base == original.ollama_base
    assert loaded.ollama_model == original.ollama_model
    # Quoted path text is escaped even on platforms that cannot create such names.
    original.output_dir = tmp_path / 'output "quoted"'
    parsed = tomllib.loads(original.to_toml_string())
    assert parsed["paths"]["output_dir"] == original.output_dir.as_posix()


def test_frozen_resources_are_not_a_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    resources = tmp_path / "extraction" / "emilia_studio" / "resources"
    resources.mkdir(parents=True)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(configuration, "get_bundled_resources_dir", lambda: resources)
    monkeypatch.setattr(configuration.shutil, "which", lambda command: None)
    loaded = configuration.load_config()
    workspace = tmp_path / "user-data"
    assert loaded.root_dir == workspace
    assert loaded.resources_dir == resources
    assert loaded.workflows_dir == resources / "workflows" / "api"
    assert loaded.visual_workflows_dir == resources / "workflows" / "visual"
    assert loaded.assets_dir == resources / "assets"
    assert loaded.manifest_path == resources / "models_manifest.json"
    assert loaded.patches_dir == resources / "patches"
    assert loaded.app_dir == workspace / "app"
    assert loaded.output_dir == workspace / "app" / "output"
    assert loaded.temp_dir == workspace / "temp"
    assert loaded.ledger_db_path == workspace / "workflows" / "studio_ledger.sqlite3"
    assert not workspace.exists(), "Reading config must not create writable state"
    assert loaded.conda_python is None
    serialized_paths = tomllib.loads(loaded.to_toml_string())["paths"]
    assert "resources_dir" not in serialized_paths
    assert "workflows_dir" not in serialized_paths
    assert "visual_workflows_dir" not in serialized_paths
    assert "conda_python" not in serialized_paths


def test_frozen_interpreter_never_resolves_to_self(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    executable = tmp_path / "emilia.exe"
    executable.touch()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))
    monkeypatch.setattr(configuration.shutil, "which", lambda command: str(executable))
    assert configuration.StudioConfig(root_dir=tmp_path / "workspace").conda_python is None
    with pytest.raises(ValueError, match="external Python"):
        configuration.StudioConfig(root_dir=tmp_path, conda_python=executable)
    monkeypatch.setenv("EMILIA_CONDA_PYTHON", str(executable))
    with pytest.raises(ValueError, match="external Python"):
        configuration.load_config()


def test_source_checkout_keeps_legacy_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    monkeypatch.setattr(configuration, "get_source_checkout_dir", lambda: checkout)
    monkeypatch.setattr(configuration, "get_bundled_resources_dir", lambda: None)
    loaded = configuration.load_config()
    assert loaded.root_dir == checkout
    assert loaded.resources_dir == checkout
    assert loaded.workflows_dir == checkout / "workflows" / "api"
    assert loaded.ledger_db_path == checkout / "workflows" / "studio_ledger.sqlite3"
