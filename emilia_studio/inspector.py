"""Deterministic local quality inspector for Emilia Studio."""

from __future__ import annotations

import hashlib
import atexit
import base64
import io
import json
import shutil
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _resolve_default_app_dir() -> Path:
    root = Path("D:/comfyui")
    if (root / "app").exists():
        return root / "app"
    return Path(__file__).resolve().parents[1] / "app"


_YOLO_MODEL_CACHE: dict[str, Any] = {}
_WARMED_MODEL_KEYS: set[str] = set()
_YOLO_LOCK = threading.Lock()
_EXTERNAL_WORKERS: dict[str, subprocess.Popen[str]] = {}
_EXTERNAL_WORKER_LOCK = threading.Lock()


def _cleanup_external_workers() -> None:
    with _EXTERNAL_WORKER_LOCK:
        for proc in _EXTERNAL_WORKERS.values():
            try:
                if proc.poll() is None:
                    proc.terminate()
            except Exception:
                pass
        _EXTERNAL_WORKERS.clear()


atexit.register(_cleanup_external_workers)

class InspectionRuntimeError(RuntimeError):
    """Required CPU detection could not run in the configured environment."""

def _rgb_to_hsv_deg(rgb_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert uint8 RGB array (H, W, 3) to (H_deg [0..360), S [0..1], V [0..1])."""
    im_hsv = Image.fromarray(rgb_arr, mode="RGB").convert("HSV")
    hsv = np.asarray(im_hsv, dtype=np.float32)
    h_deg = hsv[..., 0] * (360.0 / 255.0)
    s_norm = hsv[..., 1] / 255.0
    v_norm = hsv[..., 2] / 255.0
    return h_deg, s_norm, v_norm


def _circular_mean_deg(angles_deg: np.ndarray) -> float:
    """Compute circular mean of angles in degrees [0, 360)."""
    if angles_deg.size == 0:
        return 270.0
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
    left_eye_hue_deg: float = 270.0
    right_eye_hue_deg: float = 270.0
    heterochromia_score: float = 0.0
    left_eye_in_violet_range: bool = True
    right_eye_in_violet_range: bool = True
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
    ) -> None:
        self.app_dir = Path(app_dir) if app_dir is not None else _resolve_default_app_dir()
        self.use_yolo = use_yolo
        self.face_conf_threshold = face_conf_threshold
        self.hand_conf_threshold = hand_conf_threshold
        self.heterochromia_threshold_deg = heterochromia_threshold_deg
        self.violet_hue_range_deg = violet_hue_range_deg
        self.python_executable = str(python_executable) if python_executable is not None else None

        self.face_model_path = self.app_dir / "models" / "ultralytics" / "bbox" / "face_yolov8n.pt"
        self.hand_model_path = self.app_dir / "models" / "ultralytics" / "bbox" / "hand_yolov8n.pt"
        self.pose_model_path = self.app_dir / "models" / "ultralytics" / "pose" / "yolov8n-pose.pt"

        self._face_model: Any = None
        self._hand_model: Any = None
        self._pose_model: Any = None
        self._last_pose_detections: tuple[list[list[float]], list[float], list[list[list[float]]]] = ([], [], [])
        self._yolo_load_attempted: bool = False
    def _ensure_yolo_models(self) -> bool:
        """Load local CPU detectors, or select the configured external interpreter."""
        if not self.use_yolo:
            return False
        if self._yolo_load_attempted:
            return self._face_model is not None and self._hand_model is not None
        for path in (self.face_model_path, self.hand_model_path):
            if not path.is_file():
                raise InspectionRuntimeError(f"Required YOLO model is missing: {path}")
        with _YOLO_LOCK:
            if self._yolo_load_attempted:
                return self._face_model is not None and self._hand_model is not None
            try:
                if getattr(sys, "frozen", False):
                    raise ImportError("Frozen Emilia uses an external CPU inference environment")
                from ultralytics import YOLO

                for path in (self.face_model_path, self.hand_model_path):
                    key = str(path.resolve())
                    if key not in _YOLO_MODEL_CACHE:
                        _YOLO_MODEL_CACHE[key] = YOLO(key)
                self._face_model = _YOLO_MODEL_CACHE[str(self.face_model_path.resolve())]
                self._hand_model = _YOLO_MODEL_CACHE[str(self.hand_model_path.resolve())]
                if self.pose_model_path.is_file():
                    p_key = str(self.pose_model_path.resolve())
                    if p_key not in _YOLO_MODEL_CACHE:
                        _YOLO_MODEL_CACHE[p_key] = YOLO(p_key)
                    self._pose_model = _YOLO_MODEL_CACHE[p_key]
            except Exception as exc:
                self._face_model = None
                self._hand_model = None
                self._pose_model = None
                if self.python_executable is None:
                    raise InspectionRuntimeError(
                        "CPU YOLO is unavailable; configure conda_python with torch and ultralytics installed."
                    ) from exc
            self._yolo_load_attempted = True
            return self._face_model is not None and self._hand_model is not None

    def warmup(self, *, run_pose: bool = True) -> bool:
        """Pre-load and fuse CPU YOLO detectors off the critical path."""
        if not self.use_yolo:
            return False
        if not self.face_model_path.is_file() or not self.hand_model_path.is_file():
            return False
        if self._ensure_yolo_models():
            dummy = Image.new("RGB", (64, 64), (0, 0, 0))
            with _YOLO_LOCK:
                pairs: list[tuple[Path, Any]] = [
                    (self.face_model_path, self._face_model),
                    (self.hand_model_path, self._hand_model),
                ]
                if run_pose and self._pose_model is not None:
                    pairs.append((self.pose_model_path, self._pose_model))
                for path, model in pairs:
                    if model is None:
                        continue
                    key = str(path.resolve())
                    if key not in _WARMED_MODEL_KEYS:
                        model.predict(dummy, imgsz=64, device="cpu", verbose=False)
                        _WARMED_MODEL_KEYS.add(key)
            return True
        if self.python_executable is not None:
            executable = self.python_executable
            resolved_executable = shutil.which(executable) or executable
            worker = Path(__file__).with_name("inspection_worker.py")
            if not worker.is_file():
                return False
            req = {
                "cmd": "warmup",
                "face_path": str(self.face_model_path.resolve()),
                "hand_path": str(self.hand_model_path.resolve()),
                "pose_path": str(self.pose_model_path.resolve()) if (run_pose and self.pose_model_path.is_file()) else "",
            }
            worker_key = f"{Path(resolved_executable).resolve()}::{worker.resolve()}"
            with _EXTERNAL_WORKER_LOCK:
                proc = _EXTERNAL_WORKERS.get(worker_key)
                if proc is None or proc.poll() is not None:
                    proc = subprocess.Popen(
                        [executable, "-u", str(worker), "--server"],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding="utf-8",
                        bufsize=1,
                    )
                    _EXTERNAL_WORKERS[worker_key] = proc
                if proc.stdin is not None and proc.stdout is not None:
                    proc.stdin.write(json.dumps(req) + "\n")
                    proc.stdin.flush()
                    resp_line = proc.stdout.readline()
                    if resp_line:
                        return bool(json.loads(resp_line).get("ok"))
        return False
    def _detect_anatomy_external(
        self,
        image: Image.Image,
        *,
        run_pose: bool = True,
        image_path: Path | None = None,
    ) -> tuple[
        list[tuple[float, float, float, float]], list[float],
        list[tuple[float, float, float, float]], list[float], str,
    ]:
        """Run the bundled standalone worker via a persistent JSON-lines server process."""
        executable = self.python_executable
        if executable is None:
            raise InspectionRuntimeError("Configure conda_python for CPU YOLO inspection.")
        resolved_executable = shutil.which(executable) or executable
        if getattr(sys, "frozen", False) and Path(resolved_executable).resolve() == Path(sys.executable).resolve():
            raise InspectionRuntimeError("conda_python must be an external Python interpreter, not Emilia.")
        worker = Path(__file__).with_name("inspection_worker.py")
        if not worker.is_file():
            raise InspectionRuntimeError(f"Bundled inspection worker is missing: {worker}")
        pose_arg = (
            str(self.pose_model_path.resolve())
            if (run_pose and self.pose_model_path.is_file())
            else ""
        )
        req: dict[str, Any] = {
            "face_path": str(self.face_model_path.resolve()),
            "hand_path": str(self.hand_model_path.resolve()),
            "face_threshold": float(self.face_conf_threshold),
            "hand_threshold": float(self.hand_conf_threshold),
            "pose_path": pose_arg,
        }
        w_orig, h_orig = image.size
        max_dim = max(w_orig, h_orig)
        if max_dim > 960:
            scale_down = 640.0 / float(max_dim)
            small_w = max(32, int(round(w_orig * scale_down)))
            small_h = max(32, int(round(h_orig * scale_down)))
            small_img = image.resize((small_w, small_h), Image.Resampling.BILINEAR)
            sx = float(w_orig) / float(small_w)
            sy = float(h_orig) / float(small_h)
            with io.BytesIO() as encoded:
                small_img.save(encoded, format="PNG", compress_level=1)
                req["image_b64"] = base64.b64encode(encoded.getvalue()).decode("ascii")
        elif image_path is not None and image_path.is_file():
            req["image_path"] = str(image_path.resolve())
            sx, sy = 1.0, 1.0
        else:
            with io.BytesIO() as encoded:
                image.save(encoded, format="PNG", compress_level=1)
                req["image_b64"] = base64.b64encode(encoded.getvalue()).decode("ascii")
            sx, sy = 1.0, 1.0
        worker_key = f"{Path(resolved_executable).resolve()}::{worker.resolve()}"
        with _EXTERNAL_WORKER_LOCK:
            proc = _EXTERNAL_WORKERS.get(worker_key)
            if proc is None or proc.poll() is not None:
                try:
                    proc = subprocess.Popen(
                        [executable, "-u", str(worker), "--server"],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding="utf-8",
                        bufsize=1,
                    )
                    _EXTERNAL_WORKERS[worker_key] = proc
                except OSError as exc:
                    raise InspectionRuntimeError(f"Cannot launch CPU YOLO interpreter {executable}: {exc}") from exc
            assert proc.stdin is not None and proc.stdout is not None
            try:
                proc.stdin.write(json.dumps(req) + "\n")
                proc.stdin.flush()
                resp_line = proc.stdout.readline()
            except OSError as exc:
                _EXTERNAL_WORKERS.pop(worker_key, None)
                raise InspectionRuntimeError(f"CPU YOLO worker communication failed: {exc}") from exc
            if not resp_line:
                err_out = proc.stderr.read() if proc.stderr is not None else ""
                _EXTERNAL_WORKERS.pop(worker_key, None)
                raise InspectionRuntimeError(f"CPU YOLO inspection failed using {executable}: {err_out.strip()}")
        try:
            resp = json.loads(resp_line)
        except ValueError as exc:
            raise InspectionRuntimeError("CPU YOLO worker returned invalid JSON.") from exc
        if not resp.get("ok"):
            raise InspectionRuntimeError(f"CPU YOLO inspection failed using {executable}: {resp.get('error')}")
        try:
            data = resp["data"]
            self._last_pose_detections = (
                [[float(b[0]) * sx, float(b[1]) * sy, float(b[2]) * sx, float(b[3]) * sy] for b in data.get("pose_boxes", [])],
                [float(v) for v in data.get("pose_confidences", [])],
                [
                    [
                        [float(k[0]) * sx, float(k[1]) * sy, *(float(v) for v in k[2:])]
                        for k in person
                    ]
                    for person in data.get("pose_keypoints", [])
                ],
            )
            return (
                [(float(b[0]) * sx, float(b[1]) * sy, float(b[2]) * sx, float(b[3]) * sy) for b in data["face_boxes"]],
                [float(value) for value in data["face_confidences"]],
                [(float(b[0]) * sx, float(b[1]) * sy, float(b[2]) * sx, float(b[3]) * sy) for b in data["hand_boxes"]],
                [float(value) for value in data["hand_confidences"]],
                "yolo_cpu",
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise InspectionRuntimeError("CPU YOLO worker returned an invalid detection response.") from exc

    def detect_anatomy(
        self,
        image: Image.Image,
        *,
        run_pose: bool = True,
        image_path: Path | None = None,
    ) -> tuple[
        list[tuple[float, float, float, float]],
        list[float],
        list[tuple[float, float, float, float]],
        list[float],
        str,
    ]:
        """Run real CPU YOLO; heuristics are available only when explicitly disabled."""
        rgb = image.convert("RGB") if image.mode != "RGB" else image
        if self._ensure_yolo_models():
            w_orig, h_orig = rgb.size
            max_dim = max(w_orig, h_orig)
            if max_dim > 960:
                scale_down = 640.0 / float(max_dim)
                small_w = max(32, int(round(w_orig * scale_down)))
                small_h = max(32, int(round(h_orig * scale_down)))
                det_img = rgb.resize((small_w, small_h), Image.Resampling.BILINEAR)
                sx = float(w_orig) / float(small_w)
                sy = float(h_orig) / float(small_h)
            else:
                det_img = rgb
                sx, sy = 1.0, 1.0

            face_boxes: list[tuple[float, float, float, float]] = []
            face_confs: list[float] = []
            hand_boxes: list[tuple[float, float, float, float]] = []
            hand_confs: list[float] = []

            with _YOLO_LOCK:
                if self._face_model is not None:
                    f_res = self._face_model.predict(
                        det_img, device="cpu", verbose=False, conf=self.face_conf_threshold
                    )[0]
                    for box in f_res.boxes:
                        coords = box.xyxy[0].tolist()
                        face_boxes.append(
                            (float(coords[0]) * sx, float(coords[1]) * sy, float(coords[2]) * sx, float(coords[3]) * sy)
                        )
                        face_confs.append(float(box.conf[0]))

                if self._hand_model is not None:
                    h_res = self._hand_model.predict(
                        det_img, device="cpu", verbose=False, conf=self.hand_conf_threshold
                    )[0]
                    for box in h_res.boxes:
                        coords = box.xyxy[0].tolist()
                        hand_boxes.append(
                            (float(coords[0]) * sx, float(coords[1]) * sy, float(coords[2]) * sx, float(coords[3]) * sy)
                        )
                        hand_confs.append(float(box.conf[0]))

                if run_pose and self._pose_model is not None:
                    p_res = self._pose_model.predict(
                        det_img, device="cpu", verbose=False, conf=0.005
                    )[0]
                    p_boxes = [
                        [float(c[0]) * sx, float(c[1]) * sy, float(c[2]) * sx, float(c[3]) * sy]
                        for box in p_res.boxes
                        for c in (box.xyxy[0].tolist(),)
                    ]
                    p_confs = [float(box.conf[0]) for box in p_res.boxes]
                    raw_kpts = (
                        p_res.keypoints.data.tolist()
                        if p_res.keypoints is not None and len(p_res.boxes) > 0
                        else []
                    )
                    p_kpts = [
                        [[float(k[0]) * sx, float(k[1]) * sy, *(float(v) for v in k[2:])] for k in person]
                        for person in raw_kpts
                    ]
                    self._last_pose_detections = (p_boxes, p_confs, p_kpts)
                else:
                    self._last_pose_detections = ([], [], [])

            return face_boxes, face_confs, hand_boxes, hand_confs, "yolo_cpu"

        if self.use_yolo:
            return self._detect_anatomy_external(rgb, run_pose=run_pose, image_path=image_path)

        # Explicitly disabled detection retains the existing heuristic inspection mode.
        w, h = rgb.size
        heuristic_box = (0.30 * w, 0.08 * h, 0.70 * w, 0.34 * h)
        self._last_pose_detections = ([], [], [])
        return [heuristic_box], [1.0], [], [], "heuristic_cpu"

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
                f_boxes, _, _, _, detector_source = self.detect_anatomy(rgb, run_pose=False)
                if f_boxes:
                    face_bbox = f_boxes[0]
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

        l_mean = _circular_mean_deg(l_hues)
        r_mean = _circular_mean_deg(r_hues)
        heterochromia_score = (
            _circular_diff_deg(l_mean, r_mean)
            if (l_hues.size > 0 and r_hues.size > 0)
            else 0.0
        )

        v_min, v_max = self.violet_hue_range_deg
        l_in_violet = _is_violet_hue(l_mean, v_min, v_max) if l_hues.size > 0 else True
        r_in_violet = _is_violet_hue(r_mean, v_min, v_max) if r_hues.size > 0 else True

        hue_out_of_range = not (l_in_violet and r_in_violet)
        heterochromia_drift = heterochromia_score > self.heterochromia_threshold_deg
        eye_drift_detected = bool(hue_out_of_range or heterochromia_drift)

        drift_reasons: list[str] = []
        if heterochromia_drift:
            drift_reasons.append(
                f"heterochromia_score={heterochromia_score:.2f} deg exceeds {self.heterochromia_threshold_deg:.1f} deg"
            )
        if not l_in_violet:
            drift_reasons.append(
                f"left_eye_hue={l_mean:.1f} deg outside violet range [{v_min:.0f}, {v_max:.0f}]"
            )
        if not r_in_violet:
            drift_reasons.append(
                f"right_eye_hue={r_mean:.1f} deg outside violet range [{v_min:.0f}, {v_max:.0f}]"
            )

        return {
            "detector_source": detector_source,
            "face_bbox": [round(float(c), 2) for c in face_bbox],
            "left_eye_box": [lx1, ey1, lx2, ey2],
            "right_eye_box": [rx1, ey1, rx2, ey2],
            "left_iris_pixel_count": int(l_hues.size),
            "right_iris_pixel_count": int(r_hues.size),
            "left_eye_hue_deg": round(l_mean, 2),
            "right_eye_hue_deg": round(r_mean, 2),
            "heterochromia_score": round(heterochromia_score, 2),
            "left_eye_in_violet_range": l_in_violet,
            "right_eye_in_violet_range": r_in_violet,
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
        face_boxes: list[tuple[float, float, float, float]],
        *,
        framing: str = "full_body",
        detector_source: str = "yolo_cpu",
    ) -> dict[str, Any]:
        """Evaluate full-body standing and ankle completeness using face-height ratio and CPU YOLO pose keypoints."""
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

        p_boxes, _, p_kpts = self._last_pose_detections
        matched_kpts: list[list[float]] | None = None
        for box, kpts in zip(p_boxes, p_kpts):
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

                face_boxes, face_confs, hand_boxes, hand_confs, detector_source = (
                    self.detect_anatomy(
                        rgb_im, run_pose=(framing == "full_body"), image_path=path
                    )
                )
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
                pose_info = self.analyze_pose_completeness(
                    rgb_im, face_boxes, framing=framing, detector_source=detector_source
                )
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
