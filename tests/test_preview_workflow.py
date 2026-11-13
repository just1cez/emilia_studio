"""CPU-only behavioral contracts for official text-only distilled previews."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from emilia_studio import cli
from emilia_studio.config import StudioConfig
from emilia_studio.workflows import EmiliaWorkflowBuilder

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(tmp_path):
    return StudioConfig(root_dir=tmp_path, resources_dir=ROOT)


@pytest.mark.parametrize("method", ["build_ultra_preview", "build_highres_preview", "build_lightning_preview"])
@pytest.mark.parametrize("options", [{"steps": 18}, {"steps": 4}, {"steps": True}, {"pag": True}])
def test_preview_rejects_incompatible_sampling(config, method, options):
    with pytest.raises(ValueError):
        getattr(EmiliaWorkflowBuilder(config), method)(**options)


def test_preview_does_not_mutate_production_or_cached_templates(config):
    builder = EmiliaWorkflowBuilder(config)
    quality = builder.build_highres_native()
    ultra = builder.build_ultra_native()
    builder.build_highres_preview(positive_prompt="custom scout")
    builder.build_ultra_preview(positive_prompt="custom scout", camera_angle="side")
    assert builder.build_highres_native() == quality
    assert builder.build_ultra_native() == ultra
    assert quality == json.loads((config.workflows_dir / "emilia_ipadapter_highres_production.api.json").read_text(encoding="utf-8"))
    assert "91" not in ultra


@pytest.mark.parametrize("argv", [["render"], ["batch"], ["idea", "garden"]])
@pytest.mark.parametrize("flags", [
    ["--steps", "18"], ["--pag"], ["--fast"], ["--face-steps", "16"],
    ["--hand-steps", "16"], ["--fp16-upscaler"], ["--hires-bridge", "anime"],
    ["--garment-region", "full_garment"], ["--upscaler", "anything.pth"],
    ["--secondary-upscaler", "anything.pth"], ["--upscale-blend", "0"],
    ["--reference-mode", "single"],
])
def test_cli_rejects_preview_conflicts_before_starting_services(monkeypatch, config, argv, flags):
    monkeypatch.setattr(cli, "load_config", lambda *_: config)

    def forbidden(*args, **kwargs):
        pytest.fail("An incompatible preview must fail before constructing services")

    monkeypatch.setattr(cli, "OllamaPromptCompiler", forbidden)
    monkeypatch.setattr(cli, "EmiliaRenderEngine", forbidden)
    assert cli.main([*argv, "--preview", *flags]) == 1


