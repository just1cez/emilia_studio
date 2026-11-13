"""CPU-only synthetic edge cases for inspection, not landmark accuracy benchmarks."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from emilia_studio import inspector as inspector_module
from emilia_studio.inspector import Detections, QualityInspector, _find_iris_components


SKIN = (242, 224, 214)
VIOLET = (150, 70, 220)
GREEN = (40, 195, 70)
FACE = (150.0, 100.0, 450.0, 400.0)


@pytest.fixture
def inspector(tmp_path):
    return QualityInspector(app_dir=tmp_path, use_yolo=False)


def face_image(eyes=()):
    image = Image.new("RGB", (600, 600), SKIN)
    draw = ImageDraw.Draw(image)
    for x, y, color in eyes:
        draw.ellipse((x - 13, y - 13, x + 13, y + 13), fill=color)
    return image


def analyze(inspector, image, **kwargs):
    return inspector.analyze_iris_color(image, face_bbox=FACE, _detect_if_missing=False, **kwargs)


def assert_boxes_inside(result, size):
    width, height = size
    for key in ("left_eye_box", "right_eye_box"):
        x1, y1, x2, y2 = result[key]
        assert 0 <= x1 <= x2 <= width
        assert 0 <= y1 <= y2 <= height


def test_geometry_pair_ignores_large_lateral_hair_and_low_brooch(inspector):
    image = face_image([(242, 241, VIOLET), (358, 241, VIOLET)])
    draw = ImageDraw.Draw(image)
    draw.rectangle((174, 185, 205, 297), fill=GREEN)
    draw.ellipse((320, 315, 390, 375), fill=GREEN)
    result = analyze(inspector, image)
    assert result["localization_prior"] == "face_relative_geometry"
    assert result["eye_color_verified"] is True
    assert result["left_eye_in_violet_range"] is True
    assert result["right_eye_in_violet_range"] is True
    assert result["eye_drift_detected"] is False


def test_ornaments_alone_do_not_become_a_pair(inspector):
    image = face_image()
    draw = ImageDraw.Draw(image)
    draw.ellipse((200, 140, 245, 175), fill=VIOLET)
    draw.ellipse((345, 320, 390, 360), fill=GREEN)
    result = analyze(inspector, image)
    assert result["left_eye_hue_deg"] is None
    assert result["right_eye_hue_deg"] is None
    assert result["eye_drift_detected"] is False
    assert result["eye_color_verified"] is False
    assert result["localization_limitations"]


def test_visible_profile_green_eye_still_reports_drift(inspector):
    image = face_image([(232, 226, GREEN)])
    ImageDraw.Draw(image).ellipse((345, 320, 390, 360), fill=VIOLET)
    result = analyze(inspector, image)
    assert result["left_eye_in_violet_range"] is False
    assert result["right_eye_hue_deg"] is None
    assert result["heterochromia_score"] is None
    assert result["eye_drift_detected"] is True
    assert result["eye_color_verified"] is False


def test_pose_association_skips_other_person_and_handles_tilt(inspector):
    image = face_image([(220, 184, VIOLET), (360, 262, GREEN)])
    people = [
        [[520, 250, 0.99], [230, 210, 0.99], [370, 210, 0.99]],
        [[300, 250, 0.90], [360, 262, 0.88], [220, 184, 0.92]],
    ]
    result = analyze(inspector, image, pose_keypoints=people,
                     pose_boxes=[(460, 100, 590, 590), (100, 50, 500, 590)])
    assert result["localization_prior"] == "cpu_yolo_coco_pose"
    assert result["left_eye_in_violet_range"] is True
    assert result["right_eye_in_violet_range"] is False
    assert result["eye_drift_detected"] is True
    assert_boxes_inside(result, image.size)


def test_reliable_pose_supports_strong_rotation_not_horizontal_pair_gate(inspector):
    image = face_image([(270, 170, VIOLET), (325, 315, VIOLET)])
    people = [[[310, 245, 0.9], [325, 315, 0.95], [270, 170, 0.95]]]
    result = analyze(inspector, image, pose_keypoints=people)
    assert result["localization_prior"] == "cpu_yolo_coco_pose"
    assert result["eye_color_verified"] is True
    assert result["eye_drift_detected"] is False
    assert result["left_eye_box"][1] < 180
    assert result["right_eye_box"][1] > 290


@pytest.mark.parametrize("hidden_confidence", [0.1, 0.95])
def test_occluded_pose_eye_not_replaced_by_distant_colored_hair(inspector, hidden_confidence):
    image = face_image([(240, 220, VIOLET), (360, 280, GREEN)])
    people = [[[300, 260, 0.95], [240, 220, 0.95], [360, 220, hidden_confidence]]]
    result = analyze(inspector, image, pose_keypoints=people)
    assert result["left_eye_in_violet_range"] is True
    assert result["right_eye_hue_deg"] is None
    assert result["eye_color_verified"] is False
    assert result["eye_drift_detected"] is False
    assert len(result["unmeasured_eyes"]) == 1


def test_unreliable_pose_uses_geometry_without_color_preference(inspector):
    image = face_image([(242, 241, GREEN), (358, 241, VIOLET)])
    people = [[[300, 260, 0.95], [220, 184, 0.10], [360, 262, 0.20]]]
    result = analyze(inspector, image, pose_keypoints=people)
    assert result["localization_prior"] == "face_relative_geometry"
    assert result["left_eye_in_violet_range"] is False
    assert result["right_eye_in_violet_range"] is True
    assert result["eye_drift_detected"] is True


def test_ambiguous_people_fall_back_instead_of_selecting_first(inspector):
    image = face_image([(242, 241, GREEN), (358, 241, VIOLET)])
    people = [
        [[300, 260, 0.95], [220, 200, 0.95], [360, 260, 0.95]],
        [[300, 260, 0.95], [240, 240, 0.95], [380, 220, 0.95]],
    ]
    result = analyze(inspector, image, pose_keypoints=people)
    assert result["localization_prior"] == "face_relative_geometry"
    assert result["left_eye_in_violet_range"] is False
    assert result["right_eye_in_violet_range"] is True


def test_two_equally_plausible_blobs_do_not_guess_occluded_side(inspector):
    image = face_image([(225, 225, GREEN), (250, 260, VIOLET), (358, 241, VIOLET)])
    result = analyze(inspector, image)
    assert result["left_eye_hue_deg"] is None
    assert result["right_eye_in_violet_range"] is True
    assert result["eye_drift_detected"] is False


@pytest.mark.parametrize("bbox", [
    (-100.0, -50.0, 450.0, 400.0),
    (500.0, 500.0, 900.0, 900.0),
    (-500.0, -500.0, -100.0, -100.0),
    (450.0, 400.0, 150.0, 100.0),
    (150.0, 100.0, 150.0, 400.0),
])
def test_clipped_and_degenerate_boxes_never_wrap_pixels(inspector, bbox):
    image = face_image()
    ImageDraw.Draw(image).rectangle((540, 540, 599, 599), fill=GREEN)
    result = inspector.analyze_iris_color(image, face_bbox=bbox, _detect_if_missing=False)
    assert_boxes_inside(result, image.size)
    assert result["eye_color_verified"] is False
    assert result["eye_drift_detected"] is False


@pytest.mark.parametrize("bbox", [(float("nan"), 0, 300, 300), (0, 0, float("inf"), 300)])
def test_nonfinite_face_coordinates_are_rejected(inspector, bbox):
    with pytest.raises(ValueError, match="finite"):
        inspector.analyze_iris_color(face_image(), face_bbox=bbox, _detect_if_missing=False)


def test_component_working_image_is_bounded_without_changing_source(monkeypatch):
    array = np.full((2048, 2048, 3), SKIN, dtype=np.uint8)
    array[800:900, 600:700] = VIOLET
    original = array.copy()
    shapes = []
    convert = inspector_module._rgb_to_hsv_deg

    def record_shape(band):
        shapes.append(band.shape)
        return convert(band)

    monkeypatch.setattr(inspector_module, "_rgb_to_hsv_deg", record_shape)
    components = _find_iris_components(array, 0, 0, 2048, 2048, 2048, 2048)
    assert shapes and all(max(shape[:2]) <= 384 for shape in shapes)
    assert len(components) == 1
    assert np.array_equal(array, original)


def test_dense_hair_mask_is_not_a_giant_eye():
    array = np.full((800, 800, 3), GREEN, dtype=np.uint8)
    assert _find_iris_components(array, 0, 0, 800, 800, 800, 800) == []


def test_inspection_preserves_original_rgb_array_and_image(inspector):
    image = face_image([(242, 241, VIOLET), (358, 241, GREEN)])
    array = np.array(image)
    array.setflags(write=False)
    original_bytes = image.tobytes()
    result = analyze(inspector, image, rgb_array=array)
    assert result["eye_drift_detected"] is True
    assert image.tobytes() == original_bytes
    assert array.tobytes() == original_bytes


def test_standalone_detection_reuses_pose_from_same_call(inspector, monkeypatch):
    calls = []
    image = face_image([(220, 184, VIOLET), (360, 262, GREEN)])
    detections = Detections(
        face_boxes=[FACE], pose_boxes=[(100, 50, 500, 590)],
        pose_keypoints=[[[300, 250, 0.9], [360, 262, 0.9], [220, 184, 0.9]]],
    )

    def detect(image, **kwargs):
        calls.append(kwargs)
        return detections

    monkeypatch.setattr(inspector, "detect_anatomy", detect)
    result = inspector.analyze_iris_color(image)
    assert calls == [{"run_pose": True}]
    assert result["localization_prior"] == "cpu_yolo_coco_pose"
    assert result["eye_color_verified"] is True
    assert result["eye_drift_detected"] is True


def test_inspect_image_reuses_single_detection_for_iris(inspector, monkeypatch, tmp_path):
    image = face_image([(242, 241, VIOLET), (358, 241, GREEN)])
    path = tmp_path / "face.png"
    image.save(path)
    calls = []
    detections = Detections(
        face_boxes=[FACE], face_confidences=[0.95], source="yolo_cpu",
        pose_boxes=[(100, 50, 500, 590)],
        pose_keypoints=[[[300, 260, 0.9], [358, 241, 0.9], [242, 241, 0.9]]],
    )

    def detect(image, **kwargs):
        calls.append(kwargs)
        return detections

    monkeypatch.setattr(inspector, "detect_anatomy", detect)
    report = inspector.inspect_image(path, stage="native", expected_size=(600, 600))
    assert len(calls) == 1 and calls[0]["run_pose"] is True
    assert report.iris_analysis["localization_prior"] == "cpu_yolo_coco_pose"
    assert report.eye_drift_detected is True


def test_partially_clipped_face_keeps_original_landmark_coordinate_frame(inspector):
    image = face_image([(109, 241, GREEN)])
    result = inspector.analyze_iris_color(
        image, face_bbox=(-100.0, 100.0, 200.0, 400.0), _detect_if_missing=False,
    )
    assert result["left_eye_hue_deg"] is None
    assert result["right_eye_in_violet_range"] is False
    assert result["eye_drift_detected"] is True
    assert_boxes_inside(result, image.size)


def test_warm_iris_prior_is_explicitly_unmeasured_not_verified_violet(inspector):
    image = face_image([(242, 241, (210, 90, 40)), (358, 241, (210, 90, 40))])
    result = analyze(inspector, image)
    assert result["eye_color_verified"] is False
    assert result["left_eye_hue_deg"] is None and result["right_eye_hue_deg"] is None
    assert result["eye_drift_detected"] is False
