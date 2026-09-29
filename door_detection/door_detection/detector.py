from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class PixelPoint:
    x: int
    y: int


@dataclass(frozen=True)
class DoorHandleLocation:
    handle_start: PixelPoint
    handle_end: PixelPoint


def installed_model_path(model_name: str) -> Path:
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory('door_detection')) / 'models' / model_name


def _non_maxima_suppression(image: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    dilated = cv2.dilate(image, kernel)
    return np.where(image >= dilated, 255, 0).astype(np.uint8)


class DoorHandleDetector:
    def __init__(self, model_path: str | Path):
        self._detection_net = cv2.dnn.readNetFromONNX(str(model_path))

    def detect(self, image: np.ndarray) -> DoorHandleLocation | None:
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('The detector expects a BGR color image')

        float_image = image.astype(np.float32) / 255.0
        dnn_input = np.transpose(float_image, (2, 0, 1))[np.newaxis, ...]

        self._detection_net.setInput(dnn_input)
        labels = self._detection_net.forward()

        if labels.ndim == 4:
            output = labels[0]
        elif labels.ndim == 3:
            output = labels
        else:
            raise RuntimeError(
                f'Unexpected detector output shape: {labels.shape}')

        heatmaps = np.asarray(output)
        if heatmaps.shape[0] == 1:
            heatmap = heatmaps[0]
            mask = _non_maxima_suppression(heatmap)
            points = [
                PixelPoint(int(x), int(y))
                for y, x in np.argwhere((mask > 127) & (heatmap > 0.4))
            ]
            if len(points) == 2:
                return DoorHandleLocation(points[0], points[1])
            return None

        if heatmaps.shape[0] == 2:
            start_max, _, _, start_location = cv2.minMaxLoc(heatmaps[0])
            end_max, _, _, end_location = cv2.minMaxLoc(heatmaps[1])
            if start_max > 0.5 and end_max > 0.5:
                return DoorHandleLocation(
                    PixelPoint(int(start_location[0]), int(start_location[1])),
                    PixelPoint(int(end_location[0]), int(end_location[1])),
                )
            return None

        raise RuntimeError(
            f'Unexpected number of detector heatmaps: {heatmaps.shape[0]}')

    @staticmethod
    def annotate_detected(
        image: np.ndarray,
        door_handle: DoorHandleLocation | None,
    ) -> None:
        if door_handle is not None:
            cv2.circle(
                image,
                (door_handle.handle_start.x, door_handle.handle_start.y),
                3,
                (0, 0, 255),
                -1,
            )
            cv2.circle(
                image,
                (door_handle.handle_end.x, door_handle.handle_end.y),
                3,
                (0, 255, 0),
                -1,
            )
        else:
            cv2.circle(
                image,
                (image.shape[1] // 2, image.shape[0] // 2),
                50,
                (0, 255, 255),
            )