"""CPU YOLO detection shared by the in-process inspector and the external worker process.

Imported by ``emilia_studio.inspector`` when ultralytics is available in the same
interpreter. Run as a script by an external Python (frozen ``emilia.exe``, or a control
environment without torch), it serves JSON-lines requests on stdin/stdout until EOF:

    {"cmd": "warmup", "face_path": ..., "hand_path": ..., "pose_path": ...}
    {"face_path": ..., "hand_path": ..., "face_threshold": ..., "hand_threshold": ...,
     "pose_path": ..., "image_path": ... | "image_b64": ...}

Each request gets exactly one ``{"ok": true, ...}`` or ``{"ok": false, "error": ...}``
line. The module itself needs only the standard library; Pillow, torch and ultralytics
are imported on use. Detector output on stdout is redirected to stderr.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import sys
from pathlib import Path


_MODEL_CACHE: dict[str, object] = {}
_WARMED_MODELS: set[str] = set()


def _get_model(model_path: str):
    from ultralytics import YOLO

    key = str(Path(model_path).resolve())
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = YOLO(key)
    return _MODEL_CACHE[key]


def warmup_models(face_path: str, hand_path: str, pose_path: str = "") -> None:
    """Load each detector and run one tiny prediction so the first real image pays no setup cost."""
    from PIL import Image

    dummy = Image.new("RGB", (64, 64), (0, 0, 0))
    for model_path in (face_path, hand_path, pose_path):
        if not model_path or not Path(model_path).is_file():
            continue
        key = str(Path(model_path).resolve())
        model = _get_model(key)
        if key not in _WARMED_MODELS:
            model.predict(dummy, imgsz=64, device="cpu", verbose=False)
            _WARMED_MODELS.add(key)


def run_detection(
    image,
    face_path: str,
    hand_path: str,
    face_threshold: float,
    hand_threshold: float,
    pose_path: str = "",
) -> dict[str, object]:
    """Detect faces, hands and (with ``pose_path``) body keypoints, always on the CPU."""
    for model_path in (face_path, hand_path):
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Required YOLO model is missing: {model_path}")
    detections: dict[str, object] = {}
    for name, model_path, threshold in (
        ("face", face_path, float(face_threshold)),
        ("hand", hand_path, float(hand_threshold)),
    ):
        model = _get_model(model_path)
        result = model.predict(image, device="cpu", verbose=False, conf=threshold)[0]
        boxes = []
        confidences = []
        for box in result.boxes:
            boxes.append([float(value) for value in box.xyxy[0].tolist()])
            confidences.append(float(box.conf[0]))
        detections[f"{name}_boxes"] = boxes
        detections[f"{name}_confidences"] = confidences
    if pose_path and Path(pose_path).is_file():
        pose_model = _get_model(pose_path)
        p_res = pose_model.predict(image, device="cpu", verbose=False, conf=0.005)[0]
        detections["pose_boxes"] = [
            [float(value) for value in box.xyxy[0].tolist()]
            for box in p_res.boxes
        ]
        detections["pose_confidences"] = [float(box.conf[0]) for box in p_res.boxes]
        detections["pose_keypoints"] = (
            p_res.keypoints.data.tolist()
            if p_res.keypoints is not None and len(p_res.boxes) > 0
            else []
        )
    return detections


def _answer(request: dict[str, object]) -> dict[str, object]:
    from PIL import Image

    with contextlib.redirect_stdout(sys.stderr):
        if request.get("cmd") == "warmup":
            warmup_models(
                str(request.get("face_path") or ""),
                str(request.get("hand_path") or ""),
                str(request.get("pose_path") or ""),
            )
            return {"ok": True, "warmed": True}
        if request.get("image_path"):
            with Image.open(request["image_path"]) as source:
                source.load()
                image = source.convert("RGB")
        else:
            with Image.open(io.BytesIO(base64.b64decode(request["image_b64"]))) as source:
                image = source.convert("RGB")
        detections = run_detection(
            image,
            request["face_path"],
            request["hand_path"],
            float(request["face_threshold"]),
            float(request["hand_threshold"]),
            str(request.get("pose_path") or ""),
        )
    return {"ok": True, "data": detections}


def serve() -> None:
    """Answer one JSON line per request line until stdin closes."""
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            response = _answer(json.loads(line))
        except Exception as exc:
            # The parent turns this into InspectionRuntimeError; the worker stays usable.
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    serve()
