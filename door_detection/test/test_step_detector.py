from pathlib import Path

import cv2

from door_detection.detector import DoorHandleDetector


def _package_root() -> Path:
    return Path(__file__).parents[1]


def test_step_detector_detects_floor_edge():
    package_root = _package_root()
    image_path = package_root / 'test_data' / \
        'frame_1643551444_290952129.jpg'
    model_path = package_root / 'pytorch' / 'models' / 'floor_door_edge.onnx'

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert image is not None
    image = cv2.resize(image, (384, 288), interpolation=cv2.INTER_LINEAR)

    detector = DoorHandleDetector(model_path)
    assert detector.detect(image) is not None