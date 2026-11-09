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


def main() -> None:
    face_path, hand_path, face_threshold, hand_threshold = sys.argv[1:5]
    pose_path = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else ""
    for model_path in (face_path, hand_path):
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Required YOLO model is missing: {model_path}")

    with contextlib.redirect_stdout(sys.stderr):
        from PIL import Image
        from ultralytics import YOLO

        with Image.open(io.BytesIO(sys.stdin.buffer.read())) as source:
            image = source.convert("RGB")
        detections: dict[str, object] = {}
        for name, model_path, threshold in (
            ("face", face_path, float(face_threshold)),
            ("hand", hand_path, float(hand_threshold)),
        ):
            model = YOLO(model_path)
            result = model.predict(image, device="cpu", verbose=False, conf=threshold)[0]
            boxes = []
            confidences = []
            for box in result.boxes:
                boxes.append([float(value) for value in box.xyxy[0].tolist()])
                confidences.append(float(box.conf[0]))
            detections[f"{name}_boxes"] = boxes
            detections[f"{name}_confidences"] = confidences
        if pose_path and Path(pose_path).is_file():
            pose_model = YOLO(pose_path)
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
    json.dump(detections, sys.stdout)
    sys.stdout.write("\n")

if __name__ == "__main__":
    main()
