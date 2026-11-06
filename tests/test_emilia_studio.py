"""Unit tests for the Emilia Studio architecture overhaul."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

def _find_repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "emilia_studio").is_dir():
            return p
    return Path(__file__).resolve().parents[1]


ROOT = _find_repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if (ROOT / "app").is_dir() and str(ROOT / "app") not in sys.path:
    sys.path.insert(0, str(ROOT / "app"))
from emilia_studio.compiler import (
    OllamaPromptCompiler,
    clean_generated_prompt,
    deduplicate_tags,
    is_emilia_request,
)
from emilia_studio.config import StudioConfig, load_config
from emilia_studio.engine import EmiliaRenderEngine
from emilia_studio.inspector import QualityInspector
from emilia_studio.ledger import ExperimentLedger
from emilia_studio.workflows import EmiliaWorkflowBuilder, _RAW_WORKFLOW_CACHE


def test_config_loading_and_paths() -> None:
    cfg = load_config()
    assert cfg.root_dir == ROOT
    assert cfg.workflows_dir.is_dir()
    assert cfg.visual_workflows_dir.is_dir()
    assert cfg.comfy_port == 8188
    assert cfg.quality_port == 8199

def test_workflow_dsl_exact_parity_with_production_jsons() -> None:
    cfg = load_config()
    builder = EmiliaWorkflowBuilder(cfg)

    # 1. HighRes Native baseline parity
    expected_native = json.loads(
        (cfg.workflows_dir / "emilia_ipadapter_highres_production.api.json").read_text(
            encoding="utf-8"
        )
    )
    built_native = builder.build_highres_native(seed=20261027, pag=False)
    assert built_native == expected_native

    # 2. HighRes Native PAG injection parity
    built_pag = builder.build_highres_native(seed=20261027, pag=True)
    assert "60" in built_pag
    assert built_pag["60"]["class_type"] == "PerturbedAttentionGuidance"
    assert built_pag["60"]["inputs"] == {"model": ["42", 0], "scale": 1.0}
    assert built_pag["3"]["inputs"]["model"] == ["60", 0]
    assert built_pag["9"]["inputs"]["filename_prefix"] == "emilia/community/quality_pag_native"

    # 3. HighRes Finish parity
    expected_finish = json.loads(
        (cfg.workflows_dir / "emilia_ipadapter_highres_upscale.api.json").read_text(
            encoding="utf-8"
        )
    )
    built_finish = builder.build_highres_finish(
        image_path="emilia_quality_selected_native.png",
        seed=20261027,
        pag=False,
    )
    assert built_finish == expected_finish

    # 4. Eye repair parity
    expected_eye = json.loads(
        (cfg.workflows_dir / "emilia_eye_repair.api.json").read_text(encoding="utf-8")
    )
    built_eye = builder.build_eye_repair(
        image_path="emilia_eye_repair_source.png",
        seed=20260936,
    )
    assert built_eye == expected_eye


def test_visual_workflow_sync_zero_drift() -> None:
    cfg = load_config()
    builder = EmiliaWorkflowBuilder(cfg)
    api_graph = builder.build_highres_finish(
        image_path="custom_native.png",
        seed=123456,
        face_denoise=0.25,
        hand_denoise=0.28,
    )
    visual_path = cfg.visual_workflows_dir / "Emilia_09_Quality_Finish_2x.json"
    synced = builder.sync_visual_workflow(api_graph, visual_path)
    nodes_by_id = {str(n["id"]): n for n in synced["nodes"]}
    assert nodes_by_id["21"]["widgets_values_named"]["seed"] == 123456 + 3
    assert nodes_by_id["21"]["widgets_values_named"]["denoise"] == 0.25
    assert nodes_by_id["21"]["widgets_values_named"]["noise_mask_feather"] == 0
    assert nodes_by_id["28"]["widgets_values_named"]["seed"] == 123456 + 4
    assert nodes_by_id["28"]["widgets_values_named"]["denoise"] == 0.28
    assert nodes_by_id["28"]["widgets_values_named"]["noise_mask_feather"] == 0


def test_runtime_patches_plugin_idempotence() -> None:
    if not (ROOT / "app" / "comfy").is_dir():
        pytest.skip("ComfyUI runtime app/ not present in standalone CI")
    from custom_nodes import emilia_runtime_patches

    first = emilia_runtime_patches.apply_all_patches()
    second = emilia_runtime_patches.apply_all_patches()
    verified = emilia_runtime_patches.verify_patches()
    assert first["vae_memory_management"] is True
    assert first["utils_mask_caching"] is True
    assert first["nodes_io_optimizations"] is True
    assert first["upscale_model_in_place_clamp"] is True
    assert first["ipadapter_attention_copy_removal"] is True
    assert first == second
    assert all(verified.values())
def test_quality_inspector_iris_and_heterochromia(tmp_path: Path) -> None:
    inspector = QualityInspector(use_yolo=False)

    def _make_face_image(left_rgb: tuple[int, int, int], right_rgb: tuple[int, int, int]) -> Path:
        im = Image.new("RGB", (1152, 1728), (235, 225, 220))
        draw = ImageDraw.Draw(im)
        # Heuristic face bbox is (0.30*w, 0.08*h, 0.70*w, 0.34*h) -> (345.6, 138.24, 806.4, 587.52)
        # Left eye box is approx [419, 300, 553, 399], Right eye box is approx [599, 300, 733, 399]
        draw.rectangle([430, 310, 540, 390], fill=left_rgb)
        draw.rectangle([610, 310, 720, 390], fill=right_rgb)
        # Add some background texture so std_rgb > 1.0 and dynamic_range >= 10
        draw.rectangle([100, 1000, 1000, 1600], fill=(80, 120, 90))
        out = tmp_path / f"face_{left_rgb}_{right_rgb}.png"
        im.save(out)
        return out

    # 1. Matching canonical violet irises (RGB ~(145, 70, 220) -> Hue ~270 deg)
    violet_img = _make_face_image((145, 70, 220), (145, 70, 220))
    rep_ok = inspector.inspect_image(violet_img, stage="native")
    assert rep_ok.eye_drift_detected is False
    assert rep_ok.heterochromia_score < 5.0
    assert rep_ok.recommended_action == "accept"
    assert rep_ok.passed is True

    # 2. One-blue-one-pink heterochromia defect (Left cyan-blue ~(40, 160, 240), Right pink ~(240, 70, 150))
    drift_img = _make_face_image((40, 160, 240), (240, 70, 150))
    rep_drift = inspector.inspect_image(drift_img, stage="native")
    assert rep_drift.eye_drift_detected is True
    assert rep_drift.heterochromia_score > 32.0
    assert rep_drift.recommended_action == "repair_eyes"
    assert rep_drift.passed is False

    # 3. Strict seed lock mode records drift but keeps effective_action == "accept"
    rep_locked = inspector.inspect_image(drift_img, stage="native", strict_seed_lock=True)
    assert rep_locked.eye_drift_detected is True
    assert rep_locked.recommended_action == "repair_eyes"
    assert rep_locked.effective_action == "accept"


def test_compiler_prompt_cleaning_and_routing() -> None:
    raw = "<think>Thinking about tags...</think>\n```text\nPositive prompt: standing in snow, gentle smile, warm lantern\nNegative prompt: blurry\n```"
    cleaned = clean_generated_prompt(raw)
    assert cleaned == "standing in snow, gentle smile, warm lantern"

    assert is_emilia_request("爱蜜莉雅在雪地里微笑") is True
    assert is_emilia_request("Emilia holding a flower") is True
    assert is_emilia_request("风景画，高山流水") is False

    dedup = deduplicate_tags("1girl, solo, Silver Hair, silver hair, purple eyes")
    assert dedup == "1girl, solo, Silver Hair, purple eyes"

    compiler = OllamaPromptCompiler()
    framing_p, weight_p = compiler.infer_composition_hints("爱蜜莉雅面部特写头像", "close-up, portrait")
    assert framing_p == "portrait"
    assert weight_p == 0.50

    framing_f, weight_f = compiler.infer_composition_hints("爱蜜莉雅全身站在花园小径", "full body, garden")
    assert framing_f == "full_body"
    assert weight_f == 0.40


def test_sqlite_ledger_ingestion_and_round_comparison(tmp_path: Path) -> None:
    db_file = tmp_path / "test_ledger.sqlite3"
    ledger = ExperimentLedger(db_path=db_file)
    ingested = ledger.ingest_existing_artifacts()
    assert ingested >= 50  # At least the 50 benchmark records from rounds 1..5
    stats = ledger.get_summary_stats()
    assert stats["total_runs"] >= 50
    assert stats["unique_seeds"] >= 10

    cmp_1_5 = ledger.compare_rounds(1, 5)
    assert cmp_1_5["common_seeds"] == 10
    assert cmp_1_5["exact_pixel_matches"] == 10


def test_real_seed_20261030_multi_subject_rejection() -> None:
    cfg = load_config()
    inspector = QualityInspector(app_dir=cfg.app_dir, use_yolo=True)
    defect_img = cfg.app_dir / "output" / "emilia" / "quality" / "01_base_00011_.png"
    good_img = cfg.app_dir / "output" / "emilia" / "quality" / "01_base_00001_.png"
    if defect_img.is_file():
        rep_bad = inspector.inspect_image(defect_img, stage="native")
        assert rep_bad.face_count == 2
        assert rep_bad.multiple_subjects is True
        assert rep_bad.recommended_action == "reject_seed"
        assert rep_bad.passed is False
    if good_img.is_file():
        rep_good = inspector.inspect_image(good_img, stage="native")
        assert rep_good.face_count == 1
        assert rep_good.multiple_subjects is False
        assert rep_good.recommended_action == "accept"
        assert rep_good.passed is True

def test_setup_manifest_verification() -> None:
    cfg = load_config()
    if not (cfg.app_dir / "models" / "checkpoints" / "Illustrious-XL-v0.1.safetensors").is_file():
        pytest.skip("Local model weights not present in standalone CI")
    from emilia_studio.cli import main as cli_main

    rc = cli_main(["setup", "--sync"])
    assert rc == 0


def test_engine_local_output_zero_copy_ref_resolution(tmp_path: Path) -> None:
    cfg = StudioConfig(
        root_dir=ROOT,
        app_dir=tmp_path / "app",
        output_dir=tmp_path / "app" / "output",
        temp_dir=tmp_path / "app" / "temp",
    )
    out_file = cfg.output_dir / "emilia" / "quality" / "01_base_00001_.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_bytes(b"fake_png")
    engine = EmiliaRenderEngine(config=cfg, ledger=ExperimentLedger(db_path=tmp_path / "l.db", config=cfg))
    ref = engine._resolve_local_load_image_ref(out_file)
    assert ref == "emilia/quality/01_base_00001_.png [output]"

    in_file = (cfg.app_dir / "input" / "official_face_center_clean.png")
    in_file.parent.mkdir(parents=True, exist_ok=True)
    in_file.write_bytes(b"fake_in")
    assert engine._resolve_local_load_image_ref(in_file) == "official_face_center_clean.png"


def test_workflow_cache_and_upscale_factor() -> None:
    builder = EmiliaWorkflowBuilder(load_config())
    g_default = builder.build_highres_finish("test.png", seed=20261027, upscale_factor=2.0)
    assert g_default["15"]["inputs"]["scale_by"] == 0.5
    g_custom = builder.build_highres_finish("test.png", seed=20261027, upscale_factor=3.0)
    assert g_custom["15"]["inputs"]["scale_by"] == 0.75
    assert len(_RAW_WORKFLOW_CACHE) >= 1


def test_lanczos_and_sigma_cache_bit_exact() -> None:
    if not (ROOT / "app" / "comfy").is_dir():
        pytest.skip("ComfyUI runtime app/ not present in standalone CI")
    import torch
    import comfy.utils
    from custom_nodes import emilia_runtime_patches

    torch.manual_seed(20261027)
    sample = torch.rand((1, 3, 128, 96), dtype=torch.float32)
    out1 = comfy.utils.lanczos(sample, 64, 48)
    out2 = emilia_runtime_patches._patched_lanczos(sample, 64, 48)
    assert torch.equal(out1, out2)
    assert out1.movedim(1, -1).is_contiguous()

    cap_mod = emilia_runtime_patches._resolve_cross_attention_patch_module()
    assert cap_mod is not None
    sigmas = torch.tensor([1.2345], dtype=torch.float32)
    q = torch.zeros((2, 16, 64), dtype=torch.float32)
    attn_patch = cap_mod.Attn2Replace(
        lambda out, q, k, v, extra_options, **kw: torch.zeros_like(out),
        sigma_start=10.0,
        sigma_end=0.1,
        cond=None,
    )
    _ = attn_patch(q, q, q, {"n_heads": 4, "sigmas": sigmas})
    assert hasattr(sigmas, "_emilia_sigma_val")
    assert abs(sigmas._emilia_sigma_val - 1.2345) < 1e-5
