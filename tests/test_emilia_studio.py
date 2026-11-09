"""Unit tests for the Emilia Studio architecture overhaul."""

from __future__ import annotations
import importlib.util
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
    assert cfg.root_dir.exists() or cfg.config_source in ("auto-discovered", str(ROOT / "config.toml"))
    assert cfg.workflows_dir.is_dir()
    assert cfg.visual_workflows_dir.is_dir()
    assert cfg.comfy_port == 8188
    assert cfg.quality_port == 8199

def test_workflow_dsl_exact_parity_with_production_jsons(tmp_path: Path) -> None:
    cfg = load_config()
    # Ensure canonical model files exist in a synthetic app_dir when running in standalone CI
    if not (cfg.app_dir / "models" / "loras" / "Emilia_Rezero_Q-000005.safetensors").is_file():
        ci_app = tmp_path / "ci_parity_app"
        for rel in (
            "models/loras/Emilia_Rezero_Q-000005.safetensors",
            "models/ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors",
            "input/emilia_official_skirt_and_bodice.png",
        ):
            target = ci_app / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"")
        cfg = StudioConfig(
            root_dir=ROOT,
            app_dir=ci_app,
            workflows_dir=cfg.workflows_dir,
            visual_workflows_dir=cfg.visual_workflows_dir,
        )
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

    # 5. Ultra Multi-Scale Native & Finish parity
    expected_ultra_native = json.loads(
        (cfg.workflows_dir / "emilia_ultra_multiscale_native.api.json").read_text(encoding="utf-8")
    )
    built_ultra_native = builder.build_ultra_native(seed=20261027, pag=False)
    assert built_ultra_native == expected_ultra_native

    expected_ultra_finish = json.loads(
        (cfg.workflows_dir / "emilia_ultra_multiscale_finish.api.json").read_text(encoding="utf-8")
    )
    built_ultra_finish = builder.build_ultra_finish(
        image_path="emilia_ultra_selected_base.png",
        seed=20261027,
        pag=False,
    )
    assert built_ultra_finish == expected_ultra_finish


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
    if not (ROOT / "app" / "comfy").is_dir() or importlib.util.find_spec("torch") is None or importlib.util.find_spec("cv2") is None:
        pytest.skip("ComfyUI runtime app/, torch, or cv2 not present in standalone CI")
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
    cfg = load_config()
    if not (cfg.app_dir / "temp" / "architecture_iterations").is_dir():
        pytest.skip("Local architecture_iterations benchmark artifacts not present in standalone CI")
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
    defect_img = cfg.app_dir / "output" / "emilia" / "quality" / "01_base_00011_.png"
    good_img = cfg.app_dir / "output" / "emilia" / "quality" / "01_base_00001_.png"
    seated_img = cfg.app_dir / "output" / "emilia" / "experiments_20261109" / "garden_single_lanczos_final_00001_.png"
    standing_img = cfg.app_dir / "output" / "emilia" / "ultra" / "03_finished_2x_00008_.png"
    if not any(p.is_file() for p in (defect_img, good_img, seated_img, standing_img)):
        pytest.skip("Local rendered output fixtures not present in standalone CI")
    if importlib.util.find_spec("ultralytics") is None and cfg.conda_python is None:
        pytest.skip("ultralytics and external conda_python not present in standalone CI")
    inspector = QualityInspector(app_dir=cfg.app_dir, python_executable=cfg.conda_python, use_yolo=True)
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
    if seated_img.is_file() and standing_img.is_file():
        rep_seated = inspector.inspect_image(seated_img, stage="final", framing="full_body", require_full_body_pose=True)
        assert rep_seated.pose_analysis["full_body_complete"] is False
        assert rep_seated.recommended_action == "reject_seed"
        rep_standing = inspector.inspect_image(standing_img, stage="final", framing="full_body", require_full_body_pose=True)
        assert rep_standing.pose_analysis["full_body_complete"] is True
        assert rep_standing.recommended_action == "accept"
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
    if not (ROOT / "app" / "comfy").is_dir() or importlib.util.find_spec("torch") is None or importlib.util.find_spec("cv2") is None:
        pytest.skip("ComfyUI runtime app/, torch, or cv2 not present in standalone CI")
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


def test_fast_port_bind_check_and_stage_whitelist(tmp_path: Path) -> None:
    import socket
    import time
    from emilia_studio.engine import _assert_port_available

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]

    t0 = time.perf_counter()
    _assert_port_available("127.0.0.1", free_port)
    assert (time.perf_counter() - t0) < 0.25

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        busy_port = occupied.getsockname()[1]
        with pytest.raises(RuntimeError, match="occupied"):
            _assert_port_available("127.0.0.1", busy_port)

    app_dir = tmp_path / "app"
    for node_dir in ("comfyui-ipadapter-plus", "comfyui-impact-pack", "comfyui-impact-subpack", "emilia_runtime_patches"):
        (app_dir / "custom_nodes" / node_dir).mkdir(parents=True)
    cfg = StudioConfig(root_dir=ROOT, app_dir=app_dir, output_dir=app_dir / "output", temp_dir=app_dir / "temp")
    eng = EmiliaRenderEngine(config=cfg, inspector=QualityInspector(use_yolo=False), ledger=ExperimentLedger(tmp_path / "l.db", cfg))
    assert eng._stage_custom_node_whitelist("native") == ["comfyui-ipadapter-plus", "emilia_runtime_patches"]
    assert eng._stage_custom_node_whitelist("batch_native") == ["comfyui-ipadapter-plus", "emilia_runtime_patches"]
    assert eng._stage_custom_node_whitelist("finish") == ["comfyui-impact-pack", "comfyui-impact-subpack", "emilia_runtime_patches"]
    assert eng._stage_custom_node_whitelist("eye_repair") == ["comfyui-impact-pack", "comfyui-impact-subpack", "emilia_runtime_patches"]


def test_inspector_histogram_stats_and_roi_corner_equivalence(tmp_path: Path) -> None:
    rng = np.random.default_rng(20261027)
    arr = rng.integers(0, 256, size=(256, 384, 3), dtype=np.uint8)
    img_path = tmp_path / "synth.png"
    Image.fromarray(arr, mode="RGB").save(img_path)

    inspector = QualityInspector(use_yolo=False)
    assert inspector.warmup() is False
    rep = inspector.inspect_image(img_path, expected_size=(384, 256))
    assert abs(rep.mean_rgb - float(arr.mean(dtype=np.float64))) < 1e-4
    assert abs(rep.std_rgb - float(arr.std(dtype=np.float64))) < 1e-4
    assert rep.min_rgb == int(arr.min())
    assert rep.max_rgb == int(arr.max())

    h, w, _ = arr.shape
    ch, cw = max(16, int(round(0.06 * h))), max(32, int(round(0.22 * w)))
    full_luma = 0.299 * arr[..., 0].astype(np.float32) + 0.587 * arr[..., 1].astype(np.float32) + 0.114 * arr[..., 2].astype(np.float32)
    br_patch = full_luma[h - ch :, w - cw :]
    expected_br = round((float((np.abs(np.diff(br_patch, axis=1)) > 42.0).mean()) + float((np.abs(np.diff(br_patch, axis=0)) > 42.0).mean())) / 2.0, 5)
    assert rep.corner_artifact_analysis["corners"]["bottom_right"] == expected_br


def test_l3_blocked_quantization_and_ipadapter_kv_mask_caches() -> None:
    if not (ROOT / "app" / "comfy").is_dir() or importlib.util.find_spec("torch") is None or importlib.util.find_spec("cv2") is None:
        pytest.skip("ComfyUI runtime app/, torch, or cv2 not present in standalone CI")
    import torch
    from custom_nodes import emilia_runtime_patches

    torch.manual_seed(20261110)
    sample = torch.rand((600, 128, 3), dtype=torch.float32) * 1.3 - 0.15
    arr_ref = 255.0 * sample.numpy()
    u8_ref = np.clip(arr_ref, 0, 255, out=arr_ref).astype(np.uint8)
    u8_fast = emilia_runtime_patches._fast_tensor_to_u8_numpy(sample)
    assert np.array_equal(u8_ref, u8_fast)

    readonly_view = u8_fast.view()
    readonly_view.flags.writeable = False
    f_fast = emilia_runtime_patches._fast_u8_numpy_to_tensor(readonly_view)
    f_ref = torch.from_numpy(u8_fast.astype(np.float32) / 255.0)
    assert torch.equal(f_fast, f_ref)

    cap_mod = emilia_runtime_patches._resolve_cross_attention_patch_module()
    assert cap_mod is not None

    class _FakeIPAdapter:
        def __init__(self) -> None:
            self.calls = 0
            proj = torch.nn.Linear(32, 64, bias=False)
            torch.nn.init.orthogonal_(proj.weight)

            def _counted(x: torch.Tensor) -> torch.Tensor:
                self.calls += 1
                return proj(x)

            self.ip_layers = type("L", (), {"to_kvs": {"1_to_k_ip": _counted, "1_to_v_ip": _counted}})()

        def get_multigpu_clone(self, _device: Any) -> "_FakeIPAdapter":
            return self

    fake_ipa = _FakeIPAdapter()
    q = torch.randn((2, 24, 64), dtype=torch.float32)
    out_base = torch.zeros_like(q)
    cond = torch.randn((1, 4, 32), dtype=torch.float32)
    uncond = torch.randn((1, 4, 32), dtype=torch.float32)
    mask = torch.rand((1, 48, 32), dtype=torch.float32)
    extra = {
        "n_heads": 4,
        "cond_or_uncond": [0, 1],
        "block": ("input", 4),
        "transformer_index": 0,
        "original_shape": (2, 4, 6, 4),
    }
    out_step1 = cap_mod.ipadapter_attention(
        out_base, q, q, q, extra, module_key="1", ipadapter=fake_ipa,
        weight=0.45, cond=cond, uncond=uncond, mask=mask, embeds_scaling="V only",
    )
    assert fake_ipa.calls == 4
    assert hasattr(cond, "_emilia_kv_cache")
    assert hasattr(mask, "_emilia_attn_mask_cache")

    out_step2 = cap_mod.ipadapter_attention(
        out_base, q, q, q, extra, module_key="1", ipadapter=fake_ipa,
        weight=0.45, cond=cond, uncond=uncond, mask=mask, embeds_scaling="V only",
    )
    assert fake_ipa.calls == 4  # Zero additional linear projections on step 2+!
    assert torch.equal(out_step1, out_step2)
