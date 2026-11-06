"""Deterministic local quality inspector for Emilia Studio."""

from __future__ import annotations

import hashlib
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
    ) -> None:
        self.app_dir = Path(app_dir) if app_dir is not None else _resolve_default_app_dir()
        self.use_yolo = use_yolo
        self.face_conf_threshold = face_conf_threshold
        self.hand_conf_threshold = hand_conf_threshold
        self.heterochromia_threshold_deg = heterochromia_threshold_deg
        self.violet_hue_range_deg = violet_hue_range_deg

        self.face_model_path = self.app_dir / "models" / "ultralytics" / "bbox" / "face_yolov8n.pt"
        self.hand_model_path = self.app_dir / "models" / "ultralytics" / "bbox" / "hand_yolov8n.pt"

        self._face_model: Any = None
        self._hand_model: Any = None
        self._yolo_load_attempted: bool = False

    def _ensure_yolo_models(self) -> bool:
        """Lazily load YOLO face and hand detectors onto CPU (zero GPU VRAM)."""
        if not self.use_yolo:
            return False
        if self._yolo_load_attempted:
            return self._face_model is not None
        self._yolo_load_attempted = True
        try:
            from ultralytics import YOLO

            if self.face_model_path.is_file():
                self._face_model = YOLO(str(self.face_model_path))
            if self.hand_model_path.is_file():
                self._hand_model = YOLO(str(self.hand_model_path))
            return self._face_model is not None
        except Exception:
            self._face_model = None
            self._hand_model = None
            return False

    def detect_anatomy(
        self, image: Image.Image
    ) -> tuple[
        list[tuple[float, float, float, float]],
        list[float],
        list[tuple[float, float, float, float]],
        list[float],
        str,
    ]:
        """Run CPU-only face and hand detection or return heuristic fallback."""
        rgb = image.convert("RGB") if image.mode != "RGB" else image
        if self._ensure_yolo_models():
            face_boxes: list[tuple[float, float, float, float]] = []
            face_confs: list[float] = []
            hand_boxes: list[tuple[float, float, float, float]] = []
            hand_confs: list[float] = []

            if self._face_model is not None:
                f_res = self._face_model.predict(
                    rgb, device="cpu", verbose=False, conf=self.face_conf_threshold
                )[0]
                for box in f_res.boxes:
                    coords = box.xyxy[0].tolist()
                    face_boxes.append(
                        (float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]))
                    )
                    face_confs.append(float(box.conf[0]))

            if self._hand_model is not None:
                h_res = self._hand_model.predict(
                    rgb, device="cpu", verbose=False, conf=self.hand_conf_threshold
                )[0]
                for box in h_res.boxes:
                    coords = box.xyxy[0].tolist()
                    hand_boxes.append(
                        (float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]))
                    )
                    hand_confs.append(float(box.conf[0]))

            return face_boxes, face_confs, hand_boxes, hand_confs, "yolo_cpu"

        # Heuristic upper-body face region when YOLO is disabled or unavailable
        w, h = rgb.size
        heuristic_box = (0.30 * w, 0.08 * h, 0.70 * w, 0.34 * h)
        return [heuristic_box], [1.0], [], [], "heuristic_cpu"

    def analyze_iris_color(
        self,
        image: Image.Image | str | Path,
        face_bbox: tuple[float, float, float, float] | None = None,
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
            f_boxes, _, _, _, detector_source = self.detect_anatomy(rgb)
            if f_boxes:
                face_bbox = f_boxes[0]
            else:
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

        arr = np.asarray(rgb)
        left_crop = arr[ey1:ey2, lx1:lx2]
        right_crop = arr[ey1:ey2, rx1:rx2]

        def _extract_iris_hues(crop: np.ndarray) -> np.ndarray:
            if crop.size == 0:
                return np.empty((0,), dtype=np.float32)
            ch, cs, cv = _rgb_to_hsv_deg(crop)
            # Isolate saturated iris pixels; exclude skin tones (hue 8..55 deg), sclera (low S), and dark lashes (low V)
            mask = (cs > 0.32) & (cv > 0.20) & (cv < 0.92) & ~((ch >= 8.0) & (ch <= 55.0))
            if int(np.count_nonzero(mask)) < 8:
                mask = (cs > 0.22) & (cv > 0.15) & (cv < 0.95) & ~((ch >= 10.0) & (ch <= 50.0))
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

    def inspect_image(
        self,
        image_path: str | Path,
        expected_size: tuple[int, int] | None = None,
        stage: str | None = None,
        solo_prompt: bool = True,
        strict_seed_lock: bool = False,
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
                pixel_sha256 = hashlib.sha256(im.tobytes()).hexdigest()
                rgb_im = im.convert("RGB") if im.mode != "RGB" else im.copy()
                arr = np.asarray(rgb_im)
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

        dimensions_match = True
        reasons: list[str] = []
        if expected_size is not None and (width, height) != tuple(expected_size):
            dimensions_match = False
            reasons.append(f"Dimensions {(width, height)} do not match expected {expected_size}")

        arr_f = arr.astype(np.float32)
        non_nan = bool(np.isfinite(arr_f).all())
        mean_rgb = float(np.mean(arr_f)) if non_nan else 0.0
        std_rgb = float(np.std(arr_f)) if non_nan else 0.0
        min_rgb = int(np.min(arr)) if arr.size > 0 else 0
        max_rgb = int(np.max(arr)) if arr.size > 0 else 0
        dynamic_range = float(max_rgb - min_rgb)
        non_blank = bool(non_nan and std_rgb > 1.0 and dynamic_range >= 10.0)

        if not non_nan:
            reasons.append("Image contains NaN or infinite pixel values")
        if not non_blank:
            reasons.append(f"Image appears blank or flat (std={std_rgb:.2f}, range={dynamic_range:.1f})")

        face_boxes, face_confs, hand_boxes, hand_confs, detector_source = self.detect_anatomy(rgb_im)
        face_count = len(face_boxes)
        hand_count = len(hand_boxes)
        multiple_subjects = bool(solo_prompt and face_count > 1)

        primary_face = face_boxes[0] if face_boxes else None
        iris = self.analyze_iris_color(rgb_im, face_bbox=primary_face)
        iris["anatomy_detector_source"] = detector_source

        if face_count == 0:
            recommended_action = "reject_seed"
            reasons.append("No face detected in candidate image")
        elif multiple_subjects:
            recommended_action = "reject_seed"
            reasons.append(f"Multiple faces detected ({face_count}) for solo subject prompt")
        elif not dimensions_match or not non_blank or not non_nan:
            recommended_action = "reject_seed"
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
            recommended_action=recommended_action,
            strict_seed_lock=strict_seed_lock,
            effective_action=effective_action,
            reasons=reasons,
        )
