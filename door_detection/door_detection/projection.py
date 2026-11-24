from typing import Any

import cv2
import numpy as np


from image_geometry import PinholeCameraModel


class CameraProjector:
    def __init__(self):
        self._camera_model = PinholeCameraModel()
        self._camera_matrix = None
        self._projection_matrix = None
        self._distortion = None

    def from_camera_info(self, camera_info: Any) -> None:
        self._camera_model.fromCameraInfo(camera_info)

    def project_pixel_to_3d_ray(self, x: float, y: float) -> np.ndarray:
        ray = self._camera_model.projectPixelTo3dRay((float(x), float(y)))
        return np.asarray(ray, dtype=np.float64)



def _rotation_matrix_from_quaternion(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = x * x + y * y + z * z + w * w
    if norm == 0.0:
        return np.eye(3)

    scale = 2.0 / norm
    xx, yy, zz = x * x * scale, y * y * scale, z * z * scale
    xy, xz, yz = x * y * scale, x * z * scale, y * z * scale
    wx, wy, wz = w * x * scale, w * y * scale, w * z * scale
    return np.array([
        [1.0 - yy - zz, xy - wz, xz + wy],
        [xy + wz, 1.0 - xx - zz, yz - wx],
        [xz - wy, yz + wx, 1.0 - xx - yy],
    ])


def project_ray_to_plane(
    ray_cam: np.ndarray,
    transform: Any,
    plane_height: float,
    min_ray_z: float | None = None,
    max_ray_z: float | None = None,
) -> np.ndarray | None:
    rotation = transform.transform.rotation
    translation = transform.transform.translation
    rotation_matrix = _rotation_matrix_from_quaternion(
        rotation.x,
        rotation.y,
        rotation.z,
        rotation.w,
    )
    ray_base = rotation_matrix @ ray_cam

    if min_ray_z is not None and ray_base[2] < min_ray_z:
        return None
    if max_ray_z is not None and ray_base[2] > max_ray_z:
        return None

    camera_base = np.asarray(
        [translation.x, translation.y, translation.z],
        dtype=np.float64,
    )
    scale = (plane_height - camera_base[2]) / ray_base[2]
    return camera_base + ray_base * scale