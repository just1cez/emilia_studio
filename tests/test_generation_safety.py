"""Behavioral safety tests; rendering alone is replaced by CPU image fixtures."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emilia_studio.config import load_config
from emilia_studio.engine import EmiliaRenderEngine, QualityGateRejected
from emilia_studio.inspector import QualityInspector
from emilia_studio.ledger import ExperimentLedger


def _unpatched_function(*args, **kwargs):
    return None


@pytest.fixture
def garment_node(monkeypatch):
    """Load the actual plugin without booting ComfyUI or allocating CUDA memory."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("cv2")
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    for name in ("model_management", "sd", "utils"):
        module = types.ModuleType(f"comfy.{name}")
        setattr(comfy, name, module)
        monkeypatch.setitem(sys.modules, f"comfy.{name}", module)
    comfy.sd.VAE = type("VAE", (), {})
    comfy.utils.tiled_scale_multidim = _unpatched_function
    comfy.utils.lanczos = _unpatched_function
    nodes = types.ModuleType("nodes")
    for name in ("LoadImage", "LoadLatent", "SaveImage"):
        setattr(nodes, name, type(name, (), {"IS_CHANGED": staticmethod(_unpatched_function)}))
    monkeypatch.setitem(sys.modules, "nodes", nodes)
    monkeypatch.setitem(sys.modules, "folder_paths", types.ModuleType("folder_paths"))
    monkeypatch.setitem(sys.modules, "comfy_extras", types.ModuleType("comfy_extras"))
    spec = importlib.util.spec_from_file_location(
        "garment_safety_runtime", ROOT / "patches" / "emilia_runtime_patches" / "__init__.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EmiliaRelativeGarmentSEGS(), torch


@pytest.mark.parametrize("face", [(0, 20, 55, 85), (202, 20, 257, 85), (90, 135, 160, 205)])
@pytest.mark.parametrize("region", ["brooch_and_collar", "bodice_and_peplum", "full_garment"])
@pytest.mark.parametrize("crop_factor", [1.0, 1.6, 4.0])
def test_garment_crop_contains_mask_at_canvas_edges(garment_node, face, region, crop_factor):
    node, torch = garment_node
    image = torch.zeros((1, 259, 257, 3))
    mask = torch.zeros((1, 259, 257))
    x1, y1, x2, y2 = face
    mask[0, y1:y2, x1:x2] = 1
    (shape, segments), output = node.compute_segs(image, mask, region, crop_factor, 16)
    assert shape == (259, 257)
    assert len(segments) == 1
    seg = segments[0]
    cx1, cy1, cx2, cy2 = seg.crop_region
    gx1, gy1, gx2, gy2 = seg.bbox
    assert 0 <= cx1 <= gx1 < gx2 <= cx2 <= 257
    assert 0 <= cy1 <= gy1 < gy2 <= cy2 <= 259
    assert seg.cropped_mask.shape == (cy2 - cy1, cx2 - cx1)
    assert seg.confidence is None
    assert np.isfinite(seg.cropped_mask).all()
    assert 0 <= float(output.min()) <= float(output.max()) <= 1
    np.testing.assert_array_equal(output[0, cy1:cy2, cx1:cx2].numpy(), seg.cropped_mask)
    assert float(output.sum()) == pytest.approx(float(seg.cropped_mask.sum()), rel=1e-6)


def test_garment_rejects_ambiguous_components_and_batches(garment_node):
    node, torch = garment_node
    image = torch.zeros((1, 256, 256, 3))
    mask = torch.zeros((1, 256, 256))
    for candidate in (mask.clone(), torch.ones((2, 256, 256))):
        (_, segments), output = node.compute_segs(image, candidate)
        assert not segments and not bool(output.any())
    mask[0, 20:80, 30:80] = 1
    mask[0, 20:80, 150:200] = 1
    (_, segments), output = node.compute_segs(image, mask)
    assert not segments and not bool(output.any())


@pytest.mark.parametrize("feather", [0, 1, 8, 64])
def test_garment_feather_is_inner_cosine(garment_node, feather):
    node, torch = garment_node
    image = torch.zeros((1, 259, 257, 3))
    mask = torch.zeros((1, 259, 257))
    mask[0, 20:85, 80:140] = 1
    (_, segments), output = node.compute_segs(image, mask, feather_px=feather)
    x1, y1, x2, y2 = segments[0].bbox
    row = output[0, (y1 + y2) // 2, x1:x2].numpy()
    f = min(feather, (x2 - x1 - 1) // 2, (y2 - y1 - 1) // 2)
    if f:
        expected = 0.5 - 0.5 * np.cos(np.arange(f + 1) / f * np.pi)
        np.testing.assert_allclose(row[:f + 1], expected, atol=1e-6)
        assert row[0] == row[-1] == 0
    else:
        np.testing.assert_array_equal(row, 1)


def _fixture_image(path: Path, size: tuple[int, int], kind: str) -> None:
    image = Image.new("RGB", size, (128, 128, 128) if kind == "reject" else (230, 230, 230))
    if kind != "reject":
        draw = ImageDraw.Draw(image)
        w, h = size
        draw.rectangle((0, h * 0.8, w, h), fill=(40, 40, 40))
        if kind == "repair":
            draw.rectangle((0, h * 0.17, w, h * 0.24), fill=(30, 180, 60))
    image.save(path)


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    config = replace(load_config(), app_dir=tmp_path / "app", output_dir=tmp_path / "output", temp_dir=tmp_path / "temp")
    config.output_dir.mkdir()
    ledger = ExperimentLedger(tmp_path / "ledger.sqlite3", config)
    engine = EmiliaRenderEngine(config=config, inspector=QualityInspector(use_yolo=False), ledger=ledger)
    calls = []
    sequence = []

    async def stage(graph, name, run_dir, port, extra_pnginfo=None):
        kind = sequence.pop(0)
        calls.append((name, graph))
        filename = f"{name}.png"
        size = (2304, 3456) if name == "finish" else (1152, 1728)
        _fixture_image(config.output_dir / filename, size, kind)
        node_id = "16" if name == "finish" else "22" if name == "eye_repair" else "9"
        return {"seconds": 1.0, "submitted_graph": graph, "outputs": {node_id: {"images": [{"filename": filename}]}}}

    monkeypatch.setattr(engine, "render_stage_isolated", stage)
    return engine, sequence, calls


def test_rejected_seed_exhaustion_never_finishes_and_saves_receipt(pipeline):
    engine, sequence, calls = pipeline
    sequence.extend(["reject", "reject"])
    with pytest.raises(QualityGateRejected) as caught:
        asyncio.run(engine.render_quality_pipeline(seed=123, strict_seed_lock=False, max_seed_retries=1))
    assert [name for name, _ in calls] == ["native", "native_retry_2"]
    receipt = json.loads(caught.value.receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "quality_rejected"
    assert receipt["final"] is None
    assert [attempt["seed"] for attempt in receipt["attempts"]] == [123, 1123]
    assert all(not attempt["inspection"]["passed"] for attempt in receipt["attempts"])
    assert engine.ledger.get_summary_stats()["total_runs"] == 1


def test_retry_acceptance_reaches_finish(pipeline):
    engine, sequence, calls = pipeline
    sequence.extend(["reject", "accept", "accept"])
    receipt = asyncio.run(engine.render_quality_pipeline(seed=123, strict_seed_lock=False, max_seed_retries=1))
    assert [name for name, _ in calls] == ["native", "native_retry_2", "finish"]
    assert receipt["seed"] == 1123
    assert receipt["quality_inspection"]["final"]["passed"]


@pytest.mark.parametrize("kind", ["reject", "repair"])
def test_strict_seed_lock_preserves_stage_sequence_and_seed(pipeline, kind):
    engine, sequence, calls = pipeline
    sequence.extend([kind, kind])
    receipt = asyncio.run(engine.render_quality_pipeline(seed=123, strict_seed_lock=True, max_seed_retries=3, auto_eye_repair=True))
    assert [name for name, _ in calls] == ["native", "finish"]
    assert receipt["seed"] == 123
    assert len(receipt["attempts"]) == 1
    assert not receipt["quality_inspection"]["native"]["passed"]
    assert receipt["quality_inspection"]["native"]["effective_action"] == "accept"
    assert receipt["quality_inspection"]["eye_repair"] is None
    assert calls[0][1] == engine.builder.build_highres_native(seed=123, ipadapter_weight=0.40)
    assert calls[1][1] == engine.builder.build_highres_finish(image_path=receipt["native"], seed=123)


@pytest.mark.parametrize("profile", ["quality", "ultra"])
@pytest.mark.parametrize("repaired_kind", ["reject", "repair", "accept"])
def test_eye_repair_is_reinspected_before_finish(pipeline, profile, repaired_kind):
    engine, sequence, calls = pipeline
    sequence.extend(["repair", repaired_kind, "accept"])
    operation = engine.render_quality_pipeline(profile=profile, strict_seed_lock=False, auto_eye_repair=True)
    if repaired_kind == "accept":
        receipt = asyncio.run(operation)
        assert [name for name, _ in calls] == ["native", "eye_repair", "finish"]
        assert receipt["quality_inspection"]["repaired"]["passed"]
        assert calls[-1][1]["1"]["inputs"]["image"].endswith("eye_repair.png")
    else:
        with pytest.raises(QualityGateRejected) as caught:
            asyncio.run(operation)
        assert [name for name, _ in calls] == ["native", "eye_repair"]
        assert not caught.value.receipt["quality_inspection"]["repaired"]["passed"]


def test_final_rejection_has_receipt_without_success(pipeline):
    engine, sequence, calls = pipeline
    sequence.extend(["accept", "reject"])
    with pytest.raises(QualityGateRejected) as caught:
        asyncio.run(engine.render_quality_pipeline(strict_seed_lock=False))
    assert [name for name, _ in calls] == ["native", "finish"]
    assert Path(caught.value.receipt["final"]).is_file()
    assert caught.value.receipt_path.is_file()


@pytest.mark.parametrize("options", [{"native_options": {"hires_steps": 12}}, {"finish_options": {"garment_region": "full_garment"}}])
def test_quality_profile_rejects_tuning_before_stages(pipeline, options):
    engine, _, calls = pipeline
    with pytest.raises(ValueError):
        asyncio.run(engine.render_quality_pipeline(**options))
    assert not calls


@pytest.mark.parametrize("options", [{"native_options": {"width": 64}}, {"finish_options": {"seed": 9}}, {"finish_options": {"upscale_factor": 1}}])
def test_ultra_rejects_nonwhitelisted_options_before_stages(pipeline, options):
    engine, _, calls = pipeline
    with pytest.raises(ValueError):
        asyncio.run(engine.render_quality_pipeline(profile="ultra", **options))
    assert not calls


def test_ledger_error_propagates_after_receipt_saved(pipeline):
    engine, sequence, _ = pipeline
    sequence.extend(["reject"])
    # A real filesystem/SQLite failure, not a swallowed mock exception.
    engine.ledger.db_path = engine.config.output_dir
    with pytest.raises(Exception) as caught:
        asyncio.run(engine.render_quality_pipeline(strict_seed_lock=False))
    assert not isinstance(caught.value, QualityGateRejected)
    receipts = list(engine.config.temp_dir.glob("quality_*/rejected.run.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text(encoding="utf-8"))["status"] == "quality_rejected"


@pytest.mark.parametrize("boxes", [[], [(20, 20, 80, 80), (100, 20, 160, 80)], [(0, 20, 60, 80)], [(70, 160, 130, 220)]])
def test_garment_diagnostics_unavailable_without_unclipped_single_face(boxes):
    report = QualityInspector(use_yolo=False).analyze_garment_color(Image.new("RGB", (200, 240), "white"), boxes)
    assert not report["available"]
    assert not report["brooch"]["available"]
    assert not report["garment"]["available"]


def test_garment_diagnostics_report_only_observed_color_evidence():
    inspector = QualityInspector(use_yolo=False)
    image = Image.new("RGB", (200, 240), (30, 30, 30))
    box = [(70, 20, 130, 80)]
    assert not inspector.analyze_garment_color(image, box)["available"]
    ImageDraw.Draw(image).rectangle((65, 80, 135, 125), fill=(230, 230, 230))
    ImageDraw.Draw(image).rectangle((95, 88, 105, 98), fill=(25, 170, 75))
    report = inspector.analyze_garment_color(image, box)
    assert report["advisory_only"]
    assert report["brooch"]["available"]
    assert 0 < report["brooch"]["green_pixel_fraction"] < 0.25
    assert report["garment"]["available"]


def test_missing_garment_color_evidence_does_not_reject_image(tmp_path):
    path = tmp_path / "candidate.png"
    _fixture_image(path, (1152, 1728), "accept")
    report = QualityInspector(use_yolo=False).inspect_image(path, stage="native")
    assert not report.garment_analysis["available"]
    assert report.passed


def test_required_eye_repair_disabled_stops_before_finish(pipeline):
    engine, sequence, calls = pipeline
    sequence.append("repair")
    with pytest.raises(QualityGateRejected) as caught:
        asyncio.run(engine.render_quality_pipeline(strict_seed_lock=False, auto_eye_repair=False))
    assert [name for name, _ in calls] == ["native"]
    assert caught.value.receipt["quality_inspection"]["native"]["recommended_action"] == "repair_eyes"


def test_ultra_options_reach_real_builders_and_receipt(pipeline):
    engine, sequence, calls = pipeline
    sequence.extend(["accept", "accept"])
    native_options = {"hires_steps": 12, "reference_mode": "single", "camera_angle": "auto"}
    finish_options = {"face_denoise": 0.21, "hand_steps": 14, "upscale_blend_strength": 0.0}
    receipt = asyncio.run(engine.render_quality_pipeline(
        seed=123, profile="ultra", native_options=native_options, finish_options=finish_options,
    ))
    assert calls[0][1] == engine.builder.build_ultra_native(seed=123, ipadapter_weight=0.18, **native_options)
    assert calls[1][1] == engine.builder.build_ultra_finish(image_path=receipt["native"], seed=123, **finish_options)
    assert receipt["selected_settings"]["native_options"] == native_options
    assert receipt["selected_settings"]["finish_options"] == finish_options


def test_pose_completeness_rejects_seated_full_body_and_accepts_standing(tmp_path):
    inspector = QualityInspector(use_yolo=False)
    path = tmp_path / "pose_test.png"
    _fixture_image(path, (1152, 1728), "accept")

    # Simulate a seated/cropped pose detection: low ankle confidence (0.12) and low pelvis (hip_y=1200/1728=0.694)
    seated_kpts = [[576.0, 200.0, 0.99]] * 11 + [
        [600.0, 1200.0, 0.95], [500.0, 1200.0, 0.95],
        [600.0, 1500.0, 0.60], [500.0, 1500.0, 0.60],
        [600.0, 1600.0, 0.12], [500.0, 1600.0, 0.10],
    ]
    pose_info_bad = inspector.analyze_pose_completeness(
        Image.new("RGB", (1152, 1728)),
        [(380.0, 120.0, 760.0, 440.0)],  # fh = 320 -> fh_ratio = 0.1852 > 0.165
        framing="full_body",
        detector_source="yolo_cpu",
    )
    assert pose_info_bad["full_body_complete"] is False
    assert any("face_height_ratio" in r for r in pose_info_bad["reasons"])

    # Simulate a clean standing full-body pose
    standing_kpts = [[576.0, 200.0, 0.99]] * 11 + [
        [620.0, 800.0, 0.98], [520.0, 800.0, 0.98],
        [620.0, 1150.0, 0.95], [520.0, 1150.0, 0.95],
        [620.0, 1500.0, 0.85], [520.0, 1500.0, 0.85],
    ]
    inspector._last_pose_detections = ([[200.0, 80.0, 900.0, 1620.0]], [0.90], [standing_kpts])
    pose_info_ok = inspector.analyze_pose_completeness(
        Image.new("RGB", (1152, 1728)),
        [(450.0, 120.0, 680.0, 320.0)],  # fh = 200 -> fh_ratio = 0.1157 <= 0.165
        framing="full_body",
        detector_source="yolo_cpu",
    )
    assert pose_info_ok["full_body_complete"] is True
    assert pose_info_ok["standing_pose"] is True
    assert pose_info_ok["reasons"] == []


def test_batch_pipeline_reuses_same_stage_sessions_and_isolates_stages(tmp_path, monkeypatch):
    config = replace(load_config(), app_dir=tmp_path / "app", output_dir=tmp_path / "output", temp_dir=tmp_path / "temp")
    config.output_dir.mkdir()
    ledger = ExperimentLedger(tmp_path / "ledger.sqlite3", config)
    engine = EmiliaRenderEngine(config=config, inspector=QualityInspector(use_yolo=False), ledger=ledger)
    sessions_opened: list[str] = []
    executed_stages: list[str] = []

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def fake_stage_session(stage: str, run_dir: Path, port: int):
        sessions_opened.append(stage)
        yield "fake_session", f"http://127.0.0.1:{port}", run_dir / stage

    async def fake_exec(session, base_url, graph, stage, log_dir=None, extra_pnginfo=None):
        executed_stages.append(stage)
        filename = f"{stage}_{len(executed_stages)}.png"
        size = (2304, 3456) if stage == "finish" else (1152, 1728)
        _fixture_image(config.output_dir / filename, size, "accept")
        node_id = "16" if stage == "finish" else "9"
        from emilia_studio.engine import StageResult
        return StageResult(stage=stage, prompt_id="pid", seconds=1.0, outputs={node_id: {"images": [{"filename": filename}]}}, submitted_graph=graph)

    monkeypatch.setattr(engine, "_stage_server_session", fake_stage_session)
    monkeypatch.setattr(engine, "_execute_graph_on_session", fake_exec)

    receipts = asyncio.run(engine.render_batch_pipeline(seeds=[1001, 1002, 1003], profile="ultra"))
    assert len(receipts) == 3
    assert sessions_opened == ["batch_native", "batch_finish"]
    assert executed_stages == ["native_1", "native_2", "native_3", "finish", "finish", "finish"]
    assert engine.ledger.get_summary_stats()["total_runs"] == 3
