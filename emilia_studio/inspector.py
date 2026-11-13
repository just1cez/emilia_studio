"""Deterministic local quality inspector for Emilia Studio."""

from __future__ import annotations

import atexit
import base64
import contextlib
import hashlib
import io
import json
import queue
import shutil
import subprocess
import sys
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import inspection_worker

# Fewer saturated pixels than this is noise, not an iris: its color is reported as unmeasured.
_MIN_IRIS_PIXELS = 8

# Serializes in-process YOLO inference; the detectors are shared, non-reentrant CPU models.
_YOLO_LOCK = threading.Lock()


class InspectionRuntimeError(RuntimeError):
    """Required CPU detection could not run in the configured environment."""


class _ExternalWorker:
    """Persistent JSON-lines CPU YOLO worker process with bounded request time.

    Both pipes are drained on daemon threads: an unread stderr pipe blocks the
    worker once it fills, and a blocking stdout read could never time out.
    """

    def __init__(self, command: list[str]) -> None:
        try:
            self._proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            raise InspectionRuntimeError(f"Cannot launch CPU YOLO interpreter {command[0]}: {exc}") from exc
        self._responses: queue.Queue[str | None] = queue.Queue()
        self._stderr_tail: deque[str] = deque(maxlen=40)
        self._lock = threading.Lock()
        self._stderr_pump = threading.Thread(target=self._pump_stderr, daemon=True)
        self._stderr_pump.start()
        threading.Thread(target=self._pump_stdout, daemon=True).start()

    def _pump_stdout(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            self._responses.put(line)
        self._responses.put(None)

    def _pump_stderr(self) -> None:
        assert self._proc.stderr is not None
        for line in self._proc.stderr:
            self._stderr_tail.append(line.rstrip())

    @property
    def alive(self) -> bool:
        return self._proc.poll() is None

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            self._proc.wait(timeout=5)

    def _fail(self, message: str) -> InspectionRuntimeError:
        """Stop the worker (no stale answer may reach a later request) and attach its last stderr lines."""
        self.close()
        self._stderr_pump.join(timeout=2)
        tail = "\n".join(self._stderr_tail)
        return InspectionRuntimeError(f"{message}\n{tail}" if tail else message)

    def request(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        """Send one request and return its JSON object answer, or raise InspectionRuntimeError."""
        assert self._proc.stdin is not None
        with self._lock:
            try:
                self._proc.stdin.write(json.dumps(payload) + "\n")
                self._proc.stdin.flush()
                line = self._responses.get(timeout=timeout)
            except queue.Empty:
                raise self._fail(f"CPU YOLO worker did not answer within {timeout:.0f}s.") from None
            except (OSError, ValueError) as exc:
                raise self._fail(f"CPU YOLO worker communication failed: {exc}") from exc
            if line is None:
                raise self._fail("CPU YOLO worker exited before answering.")
        try:
            response = json.loads(line)
        except ValueError as exc:
            raise InspectionRuntimeError("CPU YOLO worker returned invalid JSON.") from exc
        if not isinstance(response, dict):
            raise InspectionRuntimeError("CPU YOLO worker returned an invalid response.")
        return response


_EXTERNAL_WORKERS: dict[str, _ExternalWorker] = {}
_EXTERNAL_WORKER_LOCK = threading.Lock()


def _cleanup_external_workers() -> None:
    with _EXTERNAL_WORKER_LOCK:
        for worker in _EXTERNAL_WORKERS.values():
            worker.close()
        _EXTERNAL_WORKERS.clear()


atexit.register(_cleanup_external_workers)



def _rgb_to_hsv_deg(rgb_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert uint8 RGB array (H, W, 3) to (H_deg [0..360), S [0..1], V [0..1])."""
    im_hsv = Image.fromarray(rgb_arr, mode="RGB").convert("HSV")
    hsv = np.asarray(im_hsv, dtype=np.float32)
    h_deg = hsv[..., 0] * (360.0 / 255.0)
    s_norm = hsv[..., 1] / 255.0
    v_norm = hsv[..., 2] / 255.0
    return h_deg, s_norm, v_norm


def _circular_mean_deg(angles_deg: np.ndarray) -> float:
    """Compute circular mean of a non-empty array of angles in degrees [0, 360)."""
    rads = np.deg2rad(angles_deg)
    mean_rad = np.arctan2(float(np.mean(np.sin(rads))), float(np.mean(np.cos(rads))))
    return float(np.rad2deg(mean_rad) % 360.0)


def _circular_diff_deg(angle_a: float, angle_b: float) -> float:
    """Compute shortest circular distance between two angles in degrees [0, 180]."""
    diff = abs(angle_a - angle_b) % 360.0
    return float(min(diff, 360.0 - diff))

def _is_violet_hue(hue_deg: float, min_deg: float = 250.0, max_deg: float = 302.0) -> bool:
    """Check whether hue (in degrees) is within Emilia's canonical violet/purple range."""
    return min_deg <= (hue_deg % 360.0) <= max_deg


@dataclass
class Detections:
    """Anatomy detections of one image, in that image's pixel coordinates."""

    face_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    face_confidences: list[float] = field(default_factory=list)
    hand_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    hand_confidences: list[float] = field(default_factory=list)
    source: str = "yolo_cpu"
    pose_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    pose_keypoints: list[list[list[float]]] = field(default_factory=list)


def _detections_from_worker(data: Any, sx: float, sy: float) -> Detections:
    """Scale raw detector output (detection-image pixels) back to the original image."""

    def boxes(raw: list[list[float]]) -> list[tuple[float, float, float, float]]:
        return [(float(b[0]) * sx, float(b[1]) * sy, float(b[2]) * sx, float(b[3]) * sy) for b in raw]

    try:
        return Detections(
            face_boxes=boxes(data["face_boxes"]),
            face_confidences=[float(value) for value in data["face_confidences"]],
            hand_boxes=boxes(data["hand_boxes"]),
            hand_confidences=[float(value) for value in data["hand_confidences"]],
            pose_boxes=boxes(data.get("pose_boxes", [])),
            pose_keypoints=[
                [[float(k[0]) * sx, float(k[1]) * sy, *(float(v) for v in k[2:])] for k in person]
                for person in data.get("pose_keypoints", [])
            ],
        )
    except (ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
        raise InspectionRuntimeError("CPU YOLO detector returned an invalid detection response.") from exc


@dataclass
class InspectionReport:
    """Inspection report for a generated native or finished Emilia image."""

    image_path: str
    exists: bool
    valid_png: bool
    width: int = 0
    height: int = 0
    expected_size: tuple[int, int] | None = None
    dimensions_match: bool = True
    pixel_sha256: str = ""
    non_blank: bool = False
    non_nan: bool = True
    mean_rgb: float = 0.0
    std_rgb: float = 0.0
    min_rgb: int = 0
    max_rgb: int = 0
    dynamic_range: float = 0.0
    face_count: int = 0
    face_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    face_confidences: list[float] = field(default_factory=list)
    multiple_subjects: bool = False
    hand_count: int = 0
    hand_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    hand_confidences: list[float] = field(default_factory=list)
    # None means "not measured" (no usable iris pixels), never a fabricated default.
    left_eye_hue_deg: float | None = None
    right_eye_hue_deg: float | None = None
    heterochromia_score: float | None = None
    left_eye_in_violet_range: bool | None = None
    right_eye_in_violet_range: bool | None = None
    eye_color_verified: bool = False
    eye_drift_detected: bool = False
    iris_analysis: dict[str, Any] = field(default_factory=dict)
    garment_analysis: dict[str, Any] = field(default_factory=lambda: {
        "available": False, "advisory_only": True, "method": "face_relative_geometry_color_prior",
        "reason": "Image unavailable", "brooch": {"available": False}, "garment": {"available": False},
    })
    pose_analysis: dict[str, Any] = field(default_factory=lambda: {
        "available": False, "framing": "full_body", "full_body_complete": True,
        "standing_pose": True, "face_height_ratio": 0.0, "max_ankle_confidence": 0.0,
        "hip_y_ratio": 0.0, "ankle_y_ratio": 0.0, "leg_span_ratio": 0.0, "reasons": [],
    })
    corner_artifact_analysis: dict[str, Any] = field(default_factory=lambda: {
        "advisory_only": True, "max_corner_edge_density": 0.0,
        "suspected_corner_watermark": False, "corners": {},
    })
    recommended_action: str = "accept"
    strict_seed_lock: bool = False
    effective_action: str = "accept"
    reasons: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether the image passed all quality checks without needing repair or rejection."""
        return (
            self.exists
            and self.valid_png
            and self.dimensions_match
            and self.non_blank
            and self.non_nan
            and self.recommended_action == "accept"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to a JSON-compatible dictionary."""
        data = asdict(self)
        if self.expected_size is not None:
            data["expected_size"] = list(self.expected_size)
        data["face_boxes"] = [list(b) for b in self.face_boxes]
        data["hand_boxes"] = [list(b) for b in self.hand_boxes]
        data["passed"] = self.passed
        return data


class QualityInspector:
    """Fast, deterministic CPU-only image inspector for Emilia Studio."""

    NATIVE_SIZE: tuple[int, int] = (1152, 1728)
    FINAL_SIZE: tuple[int, int] = (2304, 3456)
    SAFE_NATIVE_SIZE: tuple[int, int] = (832, 1216)
    SAFE_FINAL_SIZE: tuple[int, int] = (1664, 2432)

    def __init__(
        self,
        app_dir: str | Path | None = None,
        use_yolo: bool = True,
        face_conf_threshold: float = 0.45,
        hand_conf_threshold: float = 0.30,
        heterochromia_threshold_deg: float = 32.0,
        violet_hue_range_deg: tuple[float, float] = (250.0, 302.0),
        python_executable: str | Path | None = None,
        worker_timeout_s: float = 180.0,
    ) -> None:
        if app_dir is None:
            from .config import load_config

            app_dir = load_config().app_dir
        self.app_dir = Path(app_dir)
        self.use_yolo = use_yolo
        self.face_conf_threshold = face_conf_threshold
        self.hand_conf_threshold = hand_conf_threshold
        self.heterochromia_threshold_deg = heterochromia_threshold_deg
        self.violet_hue_range_deg = violet_hue_range_deg
        self.python_executable = str(python_executable) if python_executable is not None else None
        # Covers a cold torch/ultralytics import from a spinning disk as well as one detection.
        self.worker_timeout_s = worker_timeout_s

        self.face_model_path = self.app_dir / "models" / "ultralytics" / "bbox" / "face_yolov8n.pt"
        self.hand_model_path = self.app_dir / "models" / "ultralytics" / "bbox" / "hand_yolov8n.pt"
        self.pose_model_path = self.app_dir / "models" / "ultralytics" / "pose" / "yolov8n-pose.pt"

        # None until first use: True runs YOLO in this interpreter, False in the external worker.
        self._local_yolo: bool | None = None

    def _external_worker(self) -> _ExternalWorker:
        """Return the live worker for the configured interpreter, starting it when needed."""
        executable = self.python_executable
        if executable is None:
            raise InspectionRuntimeError("Configure conda_python for CPU YOLO inspection.")
        resolved_executable = shutil.which(executable) or executable
        if getattr(sys, "frozen", False) and Path(resolved_executable).resolve() == Path(sys.executable).resolve():
            raise InspectionRuntimeError("conda_python must be an external Python interpreter, not Emilia.")
        script = Path(__file__).with_name("inspection_worker.py")
        if not script.is_file():
            raise InspectionRuntimeError(f"Bundled inspection worker is missing: {script}")
        key = f"{Path(resolved_executable).resolve()}::{script.resolve()}"
        with _EXTERNAL_WORKER_LOCK:
            worker = _EXTERNAL_WORKERS.get(key)
            if worker is None or not worker.alive:
                worker = _ExternalWorker([executable, "-u", str(script)])
                _EXTERNAL_WORKERS[key] = worker
            return worker

    def _runs_locally(self) -> bool:
        """Whether YOLO runs in this interpreter (True) or in the configured external one (False)."""
        for path in (self.face_model_path, self.hand_model_path):
            if not path.is_file():
                raise InspectionRuntimeError(f"Required YOLO model is missing: {path}")
        if self._local_yolo is None:
            try:
                if getattr(sys, "frozen", False):
                    raise ImportError("Frozen Emilia uses an external CPU inference environment")
                import ultralytics  # noqa: F401

                self._local_yolo = True
            except Exception as exc:
                if self.python_executable is None:
                    raise InspectionRuntimeError(
                        "CPU YOLO is unavailable; configure conda_python with torch and ultralytics installed."
                    ) from exc
                self._local_yolo = False
        return self._local_yolo

    def _model_paths(self, run_pose: bool) -> dict[str, str]:
        return {
            "face_path": str(self.face_model_path.resolve()),
            "hand_path": str(self.hand_model_path.resolve()),
            "pose_path": str(self.pose_model_path.resolve()) if (run_pose and self.pose_model_path.is_file()) else "",
        }

    def warmup(self, *, run_pose: bool = True) -> bool:
        """Pre-load and fuse CPU YOLO detectors off the critical path."""
        if not self.use_yolo or not self.face_model_path.is_file() or not self.hand_model_path.is_file():
            return False
        paths = self._model_paths(run_pose)
        if self._runs_locally():
            with _YOLO_LOCK:
                inspection_worker.warmup_models(**paths)
            return True
        response = self._external_worker().request({"cmd": "warmup", **paths}, self.worker_timeout_s)
        return bool(response.get("ok"))

    def detect_anatomy(
        self,
        image: Image.Image,
        *,
        run_pose: bool = True,
        image_path: Path | None = None,
    ) -> Detections:
        """Run real CPU YOLO; heuristics are available only when explicitly disabled.

        The same detector code (``inspection_worker.run_detection``) serves both the
        in-process and the external-interpreter path, on the same downscaled pixels.
        """
        rgb = image.convert("RGB") if image.mode != "RGB" else image
        if not self.use_yolo:
            # Explicitly disabled detection retains the heuristic inspection mode.
            w, h = rgb.size
            return Detections(
                face_boxes=[(0.30 * w, 0.08 * h, 0.70 * w, 0.34 * h)], face_confidences=[1.0], source="heuristic_cpu"
            )
        local = self._runs_locally()
        request: dict[str, Any] = {
            **self._model_paths(run_pose),
            "face_threshold": float(self.face_conf_threshold),
            "hand_threshold": float(self.hand_conf_threshold),
        }
        w_orig, h_orig = rgb.size
        det_img, sx, sy = rgb, 1.0, 1.0
        if max(w_orig, h_orig) > 960:
            scale_down = 640.0 / float(max(w_orig, h_orig))
            small_w = max(32, int(round(w_orig * scale_down)))
            small_h = max(32, int(round(h_orig * scale_down)))
            det_img = rgb.resize((small_w, small_h), Image.Resampling.BILINEAR)
            sx = float(w_orig) / float(small_w)
            sy = float(h_orig) / float(small_h)

        if local:
            with _YOLO_LOCK:
                data: Any = inspection_worker.run_detection(det_img, **request)
        else:
            if det_img is rgb and image_path is not None and image_path.is_file():
                request["image_path"] = str(image_path.resolve())
            else:
                with io.BytesIO() as encoded:
                    det_img.save(encoded, format="PNG", compress_level=1)
                    request["image_b64"] = base64.b64encode(encoded.getvalue()).decode("ascii")
            response = self._external_worker().request(request, self.worker_timeout_s)
            if not response.get("ok"):
                raise InspectionRuntimeError(
                    f"CPU YOLO inspection failed using {self.python_executable}: {response.get('error')}"
                )
            data = response.get("data")
        return _detections_from_worker(data, sx, sy)

    def analyze_iris_color(
        self,
        image: Image.Image | str | Path,
        face_bbox: tuple[float, float, float, float] | None = None,
        *,
        rgb_array: np.ndarray | None = None,
        _detect_if_missing: bool = True,
    ) -> dict[str, Any]:
        """Locate left and right eye regions, isolate saturated iris pixels in HSV, and check for drift."""
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                opened.load()
                rgb = opened.convert("RGB")
        else:
            rgb = image.convert("RGB") if image.mode != "RGB" else image

        w, h = rgb.size
        detector_source = "provided_bbox"
        if face_bbox is None:
            if _detect_if_missing:
                detections = self.detect_anatomy(rgb, run_pose=False)
                detector_source = detections.source
                if detections.face_boxes:
                    face_bbox = detections.face_boxes[0]
            if face_bbox is None:
                face_bbox = (0.30 * w, 0.08 * h, 0.70 * w, 0.34 * h)
                detector_source = "fallback_upper_body"
        x1, y1, x2, y2 = face_bbox
        fw = max(1.0, x2 - x1)
        fh = max(1.0, y2 - y1)

        # Eye vertical band and left/right horizontal sub-boxes within the anime face bounding box
        ey1 = int(max(0, round(y1 + 0.36 * fh)))
        ey2 = int(min(h, round(y1 + 0.58 * fh)))
        lx1 = int(max(0, round(x1 + 0.16 * fw)))
        lx2 = int(min(w, round(x1 + 0.45 * fw)))
        rx1 = int(max(0, round(x1 + 0.55 * fw)))
        rx2 = int(min(w, round(x1 + 0.84 * fw)))

        arr = rgb_array if rgb_array is not None else np.asarray(rgb)
        left_crop = arr[ey1:ey2, lx1:lx2]
        right_crop = arr[ey1:ey2, rx1:rx2]

        def _extract_iris_hues(crop: np.ndarray) -> np.ndarray:
            if crop.size == 0:
                return np.empty((0,), dtype=np.float32)
            ch, cs, cv = _rgb_to_hsv_deg(crop)
            # Isolate saturated iris pixels; exclude warm skin/eyelash tones (hue 0..65 and 345..360 deg), sclera (low S), and dark lashes (low V)
            mask = (cs > 0.32) & (cv > 0.20) & (cv < 0.92) & (ch >= 65.0) & (ch <= 345.0)
            if int(np.count_nonzero(mask)) < 8:
                mask = (cs > 0.22) & (cv > 0.15) & (cv < 0.95) & (ch >= 65.0) & (ch <= 345.0)
            return ch[mask]
        l_hues = _extract_iris_hues(left_crop)
        r_hues = _extract_iris_hues(right_crop)

        # An eye without enough iris pixels (closed, occluded, profile view) is unmeasured:
        # it neither proves nor disproves the violet color and never triggers a repair.
        left_measured = l_hues.size >= _MIN_IRIS_PIXELS
        right_measured = r_hues.size >= _MIN_IRIS_PIXELS
        l_mean = _circular_mean_deg(l_hues) if left_measured else None
        r_mean = _circular_mean_deg(r_hues) if right_measured else None
        heterochromia_score = (
            _circular_diff_deg(l_mean, r_mean) if l_mean is not None and r_mean is not None else None
        )

        v_min, v_max = self.violet_hue_range_deg
        l_in_violet = _is_violet_hue(l_mean, v_min, v_max) if l_mean is not None else None
        r_in_violet = _is_violet_hue(r_mean, v_min, v_max) if r_mean is not None else None

        heterochromia_drift = heterochromia_score is not None and heterochromia_score > self.heterochromia_threshold_deg
        eye_drift_detected = bool(l_in_violet is False or r_in_violet is False or heterochromia_drift)

        drift_reasons: list[str] = []
        if heterochromia_drift:
            drift_reasons.append(
                f"heterochromia_score={heterochromia_score:.2f} deg exceeds {self.heterochromia_threshold_deg:.1f} deg"
            )
        if l_in_violet is False:
            drift_reasons.append(
                f"left_eye_hue={l_mean:.1f} deg outside violet range [{v_min:.0f}, {v_max:.0f}]"
            )
        if r_in_violet is False:
            drift_reasons.append(
                f"right_eye_hue={r_mean:.1f} deg outside violet range [{v_min:.0f}, {v_max:.0f}]"
            )
        unmeasured_eyes = [
            f"{side} eye: {count} iris pixels (< {_MIN_IRIS_PIXELS}); color not measured"
            for side, count, measured in (
                ("left", int(l_hues.size), left_measured),
                ("right", int(r_hues.size), right_measured),
            )
            if not measured
        ]

        return {
            "detector_source": detector_source,
            "face_bbox": [round(float(c), 2) for c in face_bbox],
            "left_eye_box": [lx1, ey1, lx2, ey2],
            "right_eye_box": [rx1, ey1, rx2, ey2],
            "left_iris_pixel_count": int(l_hues.size),
            "right_iris_pixel_count": int(r_hues.size),
            "left_eye_hue_deg": round(l_mean, 2) if l_mean is not None else None,
            "right_eye_hue_deg": round(r_mean, 2) if r_mean is not None else None,
            "heterochromia_score": round(heterochromia_score, 2) if heterochromia_score is not None else None,
            "left_eye_in_violet_range": l_in_violet,
            "right_eye_in_violet_range": r_in_violet,
            "eye_color_verified": left_measured and right_measured,
            "unmeasured_eyes": unmeasured_eyes,
            "eye_drift_detected": eye_drift_detected,
            "drift_reasons": drift_reasons,
        }

    def analyze_garment_color(
        self,
        image: Image.Image,
        face_boxes: list[tuple[float, float, float, float]],
        rgb_array: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Advisory CPU color samples from a geometry prior, not segmentation.

        Fractions describe pixels in a face-relative rectangle, not identified
        garments or jewels. They cannot establish outfit identity, construction,
        pleat counts or missing accessories and never participate in the gate.
        """
        result: dict[str, Any] = {
            "available": False, "advisory_only": True,
            "method": "face_relative_geometry_color_prior",
            "reason": "Exactly one detected face is required",
            "brooch": {"available": False}, "garment": {"available": False},
        }
        if len(face_boxes) != 1:
            return result
        box = np.asarray(face_boxes[0], dtype=np.float64)
        if box.shape != (4,) or not np.isfinite(box).all():
            result["reason"] = "Invalid face geometry"
            return result
        x1, y1, x2, y2 = (float(v) for v in box)
        fw, fh = x2 - x1, y2 - y1
        w, h = image.size
        if fw < 16 or fh < 16 or x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            result["reason"] = "Face geometry is too small or clipped"
            return result
        cx = (x1 + x2) / 2
        bounds = (cx - 0.65 * fw, y2 - 0.06 * fh, cx + 0.65 * fw, y2 + 0.82 * fh)
        if bounds[0] < 0 or bounds[1] < 0 or bounds[2] > w or bounds[3] > h:
            result["reason"] = "Garment geometry ROI is clipped"
            return result
        rx1, ry1 = int(np.floor(bounds[0])), int(np.floor(bounds[1]))
        rx2, ry2 = int(np.ceil(bounds[2])), int(np.ceil(bounds[3]))
        result["roi"] = [rx1, ry1, rx2, ry2]
        arr = rgb_array if rgb_array is not None else np.asarray(image.convert("RGB"))
        crop = arr[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            result["reason"] = "Empty garment geometry ROI"
            return result
        hue, saturation, value = _rgb_to_hsv_deg(crop)
        green = (hue >= 90) & (hue <= 170) & (saturation >= 0.35) & (value >= 0.20)
        violet = (hue >= 250) & (hue <= 302) & (saturation >= 0.20) & (value >= 0.20)
        light = (saturation <= 0.20) & (value >= 0.70)
        pixels = hue.size
        minimum = max(16, int(np.ceil(pixels * 0.005)))
        green_count = int(green.sum())
        garment_count = int(violet.sum() + light.sum())
        # Broad green areas are not reliable localized brooch-color evidence.
        if minimum <= green_count <= pixels * 0.25:
            result["brooch"] = {
                "available": True, "green_pixel_fraction": round(green_count / pixels, 5),
                "interpretation": "Green pixels in collar ROI; brooch identity is not established",
            }
        else:
            result["brooch"]["reason"] = "No reliable localized green-color evidence"
        if garment_count >= max(32, pixels * 0.05):
            result["garment"] = {
                "available": True,
                "violet_pixel_fraction": round(float(violet.mean()), 5),
                "light_neutral_pixel_fraction": round(float(light.mean()), 5),
                "interpretation": "ROI color evidence only; garment identity is not established",
            }
        else:
            result["garment"]["reason"] = "No reliable light-neutral or violet-color evidence"
        result["available"] = result["brooch"]["available"] or result["garment"]["available"]
        result["reason"] = "Advisory ROI color evidence" if result["available"] else "No reliable color evidence"
        return result

    def analyze_pose_completeness(
        self,
        image: Image.Image,
        detections: Detections,
        *,
        framing: str = "full_body",
    ) -> dict[str, Any]:
        """Evaluate full-body standing and ankle completeness using face-height ratio and CPU YOLO pose keypoints."""
        face_boxes, detector_source = detections.face_boxes, detections.source
        w, h = image.size
        result: dict[str, Any] = {
            "available": False,
            "framing": framing,
            "full_body_complete": True,
            "standing_pose": True,
            "face_height_ratio": 0.0,
            "max_ankle_confidence": 0.0,
            "hip_y_ratio": 0.0,
            "ankle_y_ratio": 0.0,
            "leg_span_ratio": 0.0,
            "reasons": [],
        }
        if not face_boxes or h <= 0 or w <= 0:
            return result
        fx1, fy1, fx2, fy2 = face_boxes[0]
        fh = max(1.0, fy2 - fy1)
        fcx, fcy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
        fh_ratio = fh / float(h)
        result["face_height_ratio"] = round(fh_ratio, 4)

        matched_kpts: list[list[float]] | None = None
        for box, kpts in zip(detections.pose_boxes, detections.pose_keypoints):
            if len(box) >= 4 and len(kpts) >= 17:
                if box[0] <= fcx <= box[2] and box[1] <= fcy <= box[3] and (box[2] - box[0]) >= 0.20 * w:
                    matched_kpts = kpts
                    break

        if matched_kpts is not None:
            result["available"] = True
            l_hip, r_hip = matched_kpts[11], matched_kpts[12]
            l_ankle, r_ankle = matched_kpts[15], matched_kpts[16]
            hip_y = min(float(l_hip[1]), float(r_hip[1])) / float(h)
            ankle_y = max(float(l_ankle[1]), float(r_ankle[1])) / float(h)
            max_ankle_conf = max(float(l_ankle[2]), float(r_ankle[2]))
            leg_span = max(0.0, ankle_y - hip_y)
            result["max_ankle_confidence"] = round(max_ankle_conf, 4)
            result["hip_y_ratio"] = round(hip_y, 4)
            result["ankle_y_ratio"] = round(ankle_y, 4)
            result["leg_span_ratio"] = round(leg_span, 4)

            if framing == "full_body":
                reasons: list[str] = []
                if fh_ratio > 0.165:
                    reasons.append(
                        f"face_height_ratio={fh_ratio:.3f} exceeds 0.165 (seated/close-up framing instead of standing full_body)"
                    )
                if max_ankle_conf < 0.30:
                    reasons.append(
                        f"max_ankle_confidence={max_ankle_conf:.3f} below 0.30 (cropped feet or kneeling/seated pose)"
                    )
                if hip_y > 0.60:
                    reasons.append(
                        f"hip_y_ratio={hip_y:.3f} exceeds 0.60 (pelvis too low in frame for standing full_body)"
                    )
                if ankle_y < 0.68:
                    reasons.append(
                        f"ankle_y_ratio={ankle_y:.3f} below 0.68 (seated or floating legs)"
                    )
                if reasons:
                    result["full_body_complete"] = False
                    result["standing_pose"] = False
                    result["reasons"] = reasons
        elif detector_source == "yolo_cpu" and framing == "full_body":
            result["available"] = True
            if fh_ratio > 0.165:
                result["full_body_complete"] = False
                result["standing_pose"] = False
                result["reasons"] = [
                    f"face_height_ratio={fh_ratio:.3f} exceeds 0.165 (seated/close-up framing instead of standing full_body)"
                ]
        return result

    def analyze_corner_artifacts(self, rgb_array: np.ndarray) -> dict[str, Any]:
        """Measure high-frequency bimodal stroke density in the 4 outer corners (advisory watermark check)."""
        if rgb_array.ndim != 3 or rgb_array.shape[0] < 64 or rgb_array.shape[1] < 64:
            return {
                "advisory_only": True,
                "max_corner_edge_density": 0.0,
                "suspected_corner_watermark": False,
                "corners": {},
            }
        h, w, _ = rgb_array.shape
        ch = max(16, int(round(0.06 * h)))
        cw = max(32, int(round(0.22 * w)))
        rois = {
            "top_left": rgb_array[:ch, :cw],
            "top_right": rgb_array[:ch, w - cw :],
            "bottom_left": rgb_array[h - ch :, :cw],
            "bottom_right": rgb_array[h - ch :, w - cw :],
        }
        corners: dict[str, float] = {}
        for name, patch_rgb in rois.items():
            patch = (
                0.299 * patch_rgb[..., 0].astype(np.float32)
                + 0.587 * patch_rgb[..., 1].astype(np.float32)
                + 0.114 * patch_rgb[..., 2].astype(np.float32)
            )
            gx = np.abs(np.diff(patch, axis=1))
            gy = np.abs(np.diff(patch, axis=0))
            sharp_frac = float((gx > 42.0).mean() + (gy > 42.0).mean()) / 2.0
            corners[name] = round(sharp_frac, 5)
        max_density = max(corners.values()) if corners else 0.0
        return {
            "advisory_only": True,
            "max_corner_edge_density": round(max_density, 5),
            "suspected_corner_watermark": bool(max_density > 0.12),
            "corners": corners,
        }

    def inspect_image(
        self,
        image_path: str | Path,
        expected_size: tuple[int, int] | None = None,
        stage: str | None = None,
        solo_prompt: bool = True,
        strict_seed_lock: bool = False,
        framing: str = "full_body",
        require_full_body_pose: bool = False,
    ) -> InspectionReport:
        """Run full deterministic inspection on an image and return an InspectionReport."""
        path = Path(image_path)
        if expected_size is None and stage is not None:
            if stage == "native":
                expected_size = self.NATIVE_SIZE
            elif stage in ("final", "finish"):
                expected_size = self.FINAL_SIZE

        if not path.is_file():
            return InspectionReport(
                image_path=str(path),
                exists=False,
                valid_png=False,
                expected_size=expected_size,
                dimensions_match=False,
                recommended_action="reject_seed",
                strict_seed_lock=strict_seed_lock,
                effective_action="accept" if strict_seed_lock else "reject_seed",
                reasons=["Image file does not exist"],
            )

        try:
            with Image.open(path) as im:
                im.load()
                width, height = im.size
                if im.mode == "RGB":
                    rgb_im = im
                    arr = np.asarray(rgb_im)
                    pixel_sha256 = hashlib.sha256(memoryview(arr)).hexdigest()
                else:
                    pixel_sha256 = hashlib.sha256(im.tobytes()).hexdigest()
                    rgb_im = im.convert("RGB")
                    arr = np.asarray(rgb_im)

                dimensions_match = True
                reasons: list[str] = []
                if expected_size is not None and (width, height) != tuple(expected_size):
                    dimensions_match = False
                    reasons.append(
                        f"Dimensions {(width, height)} do not match expected {expected_size}"
                    )

                if arr.dtype == np.uint8 and arr.size > 0:
                    non_nan = True
                    counts = np.asarray(rgb_im.histogram(), dtype=np.float64).reshape(-1, 256).sum(axis=0)
                    vals = np.arange(256, dtype=np.float64)
                    n_vals = float(arr.size)
                    mean_rgb = float(np.dot(counts, vals) / n_vals)
                    std_rgb = float(np.sqrt(np.dot(counts, (vals - mean_rgb) ** 2) / n_vals))
                    nz = np.flatnonzero(counts)
                    min_rgb = int(nz[0])
                    max_rgb = int(nz[-1])
                else:
                    if np.issubdtype(arr.dtype, np.integer):
                        non_nan = True
                    else:
                        non_nan = bool(np.isfinite(arr).all())
                    mean_rgb = float(arr.mean(dtype=np.float64)) if (non_nan and arr.size > 0) else 0.0
                    std_rgb = float(arr.std(dtype=np.float64)) if (non_nan and arr.size > 0) else 0.0
                    min_rgb = int(arr.min()) if arr.size > 0 else 0
                    max_rgb = int(arr.max()) if arr.size > 0 else 0
                dynamic_range = float(max_rgb - min_rgb)
                non_blank = bool(non_nan and std_rgb > 1.0 and dynamic_range >= 10.0)

                if not non_nan:
                    reasons.append("Image contains NaN or infinite pixel values")
                if not non_blank:
                    reasons.append(
                        f"Image appears blank or flat (std={std_rgb:.2f}, range={dynamic_range:.1f})"
                    )

                detections = self.detect_anatomy(
                    rgb_im, run_pose=(framing == "full_body"), image_path=path
                )
                face_boxes, face_confs = detections.face_boxes, detections.face_confidences
                hand_boxes, hand_confs = detections.hand_boxes, detections.hand_confidences
                detector_source = detections.source
                face_count = len(face_boxes)
                hand_count = len(hand_boxes)
                multiple_subjects = bool(solo_prompt and face_count > 1)

                primary_face = face_boxes[0] if face_boxes else None
                iris = self.analyze_iris_color(
                    rgb_im, face_bbox=primary_face, rgb_array=arr, _detect_if_missing=False
                )
                iris["anatomy_detector_source"] = detector_source
                try:
                    garment = self.analyze_garment_color(
                        rgb_im, face_boxes if detector_source == "yolo_cpu" else [], rgb_array=arr,
                    )
                except Exception as exc:
                    # Optional color diagnostics must never invalidate decoded output.
                    garment = {
                        "available": False, "advisory_only": True,
                        "method": "face_relative_geometry_color_prior",
                        "reason": f"Color diagnostics unavailable: {type(exc).__name__}",
                        "brooch": {"available": False}, "garment": {"available": False},
                    }
                pose_info = self.analyze_pose_completeness(rgb_im, detections, framing=framing)
                corner_info = self.analyze_corner_artifacts(arr)
        except InspectionRuntimeError:
            raise
        except Exception as exc:
            return InspectionReport(
                image_path=str(path),
                exists=True,
                valid_png=False,
                expected_size=expected_size,
                dimensions_match=False,
                recommended_action="reject_seed",
                strict_seed_lock=strict_seed_lock,
                effective_action="accept" if strict_seed_lock else "reject_seed",
                reasons=[f"Failed to decode PNG: {exc}"],
            )
        if face_count == 0:
            recommended_action = "reject_seed"
            reasons.append("No face detected in candidate image")
        elif multiple_subjects:
            recommended_action = "reject_seed"
            reasons.append(f"Multiple faces detected ({face_count}) for solo subject prompt")
        elif not dimensions_match or not non_blank or not non_nan:
            recommended_action = "reject_seed"
        elif require_full_body_pose and framing == "full_body" and not pose_info["full_body_complete"]:
            recommended_action = "reject_seed"
            reasons.extend(pose_info["reasons"])
        elif iris["eye_drift_detected"]:
            recommended_action = "repair_eyes"
            reasons.extend(iris["drift_reasons"])
        else:
            recommended_action = "accept"

        # In strict_seed_lock / benchmark mode, preserve pixel-exact pipeline execution while recording diagnostics
        effective_action = "accept" if strict_seed_lock else recommended_action

        return InspectionReport(
            image_path=str(path),
            exists=True,
            valid_png=True,
            width=width,
            height=height,
            expected_size=expected_size,
            dimensions_match=dimensions_match,
            pixel_sha256=pixel_sha256,
            non_blank=non_blank,
            non_nan=non_nan,
            mean_rgb=round(mean_rgb, 4),
            std_rgb=round(std_rgb, 4),
            min_rgb=min_rgb,
            max_rgb=max_rgb,
            dynamic_range=round(dynamic_range, 2),
            face_count=face_count,
            face_boxes=face_boxes,
            face_confidences=[round(c, 4) for c in face_confs],
            multiple_subjects=multiple_subjects,
            hand_count=hand_count,
            hand_boxes=hand_boxes,
            hand_confidences=[round(c, 4) for c in hand_confs],
            left_eye_hue_deg=iris["left_eye_hue_deg"],
            right_eye_hue_deg=iris["right_eye_hue_deg"],
            heterochromia_score=iris["heterochromia_score"],
            left_eye_in_violet_range=iris["left_eye_in_violet_range"],
            right_eye_in_violet_range=iris["right_eye_in_violet_range"],
            eye_color_verified=iris["eye_color_verified"],
            eye_drift_detected=iris["eye_drift_detected"],
            iris_analysis=iris,
            garment_analysis=garment,
            pose_analysis=pose_info,
            corner_artifact_analysis=corner_info,
            recommended_action=recommended_action,
            strict_seed_lock=strict_seed_lock,
            effective_action=effective_action,
            reasons=reasons,
        )
