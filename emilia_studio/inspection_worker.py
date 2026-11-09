"""Standalone external CPU YOLO worker bundled with Emilia Studio.

Arguments: face weights, hand weights, face threshold, hand threshold.
Reads one PNG from binary stdin and writes one JSON detection object to stdout.
Only the external environment needs Pillow, torch, and ultralytics; this script
never imports emilia_studio. Diagnostics are reserved for stderr.
"""

from __future__ import annotations

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


def _warmup_models(face_path: str, hand_path: str, pose_path: str = "") -> None:
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


def _run_detection(
    image,
    face_path: str,
    hand_path: str,
    face_threshold: float,
    hand_threshold: float,
    pose_path: str = "",
) -> dict[str, object]:
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
        p_boxes = [
            [float(value) for value in box.xyxy[0].tolist()]
            for box in p_res.boxes
        ]
        p_confs = [float(box.conf[0]) for box in p_res.boxes]
        p_kpts = (
            p_res.keypoints.data.tolist()
            if p_res.keypoints is not None and len(p_res.boxes) > 0
            else []
        )
        detections["pose_boxes"] = p_boxes
        detections["pose_confidences"] = p_confs
        detections["pose_keypoints"] = p_kpts
    return detections


def _serve_forever() -> None:
    import base64
    from PIL import Image

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("cmd") == "quit":
                break
            if req.get("cmd") == "warmup":
                with contextlib.redirect_stdout(sys.stderr):
                    _warmup_models(
                        str(req.get("face_path") or ""),
                        str(req.get("hand_path") or ""),
                        str(req.get("pose_path") or ""),
                    )
                sys.stdout.write(json.dumps({"ok": True, "warmed": True}) + "\n")
                sys.stdout.flush()
                continue
            with contextlib.redirect_stdout(sys.stderr):
                if req.get("image_path"):
                    with Image.open(req["image_path"]) as src:
                        src.load()
                        image = src.convert("RGB")
                else:
                    raw_bytes = base64.b64decode(req["image_b64"])
                    with Image.open(io.BytesIO(raw_bytes)) as src:
                        image = src.convert("RGB")
                detections = _run_detection(
                    image,
                    req["face_path"],
                    req["hand_path"],
                    float(req["face_threshold"]),
                    float(req["hand_threshold"]),
                    str(req.get("pose_path") or ""),
                )
            sys.stdout.write(json.dumps({"ok": True, "data": detections}) + "\n")
            sys.stdout.flush()
        except Exception as exc:
            sys.stdout.write(json.dumps({"ok": False, "error": str(exc)}) + "\n")
            sys.stdout.flush()


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        _serve_forever()
        return
    face_path, hand_path, face_threshold, hand_threshold = sys.argv[1:5]
    pose_path = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else ""
    with contextlib.redirect_stdout(sys.stderr):
        from PIL import Image

        with Image.open(io.BytesIO(sys.stdin.buffer.read())) as source:
            image = source.convert("RGB")
        detections = _run_detection(
            image,
            face_path,
            hand_path,
            float(face_threshold),
            float(hand_threshold),
            pose_path,
        )
    json.dump(detections, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
