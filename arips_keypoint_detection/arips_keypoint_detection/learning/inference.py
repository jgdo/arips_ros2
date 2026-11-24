"""ONNX keypoint detector without torch (onnxruntime, falls back to cv2.dnn)."""

from pathlib import Path

import cv2
import numpy as np

from . import preprocess as pp


class KeypointDetector:

    def __init__(self, onnx_path, backend: str = 'auto'):
        self.path = Path(onnx_path)
        self.meta = pp.load_meta(self.path)
        self.inputs = self.meta['inputs']
        self.names = self.meta['keypoint_names']
        self.threshold = float(self.meta.get('threshold', 0.3))
        self.stride = int(self.meta['heatmap_stride'])
        self.backend = None
        if backend in ('auto', 'onnxruntime'):
            try:
                import onnxruntime as ort
                self._session = ort.InferenceSession(str(self.path), providers=['CPUExecutionProvider'])
                self.backend = 'onnxruntime'
            except ImportError:
                if backend == 'onnxruntime':
                    raise
        if self.backend is None:
            self._net = cv2.dnn.readNetFromONNX(str(self.path))
            self.backend = 'cv2.dnn'

    def heatmaps(self, tensors: dict) -> np.ndarray:
        """{input name: 1xCxHxW} -> K x Hh x Wh probabilities."""
        if self.backend == 'onnxruntime':
            out = self._session.run(None, {name: tensors[name] for name in self.inputs})[0]
        else:
            for name in self.inputs:
                self._net.setInput(tensors[name], name)
            out = self._net.forward()
        return np.asarray(out[0], np.float32)

    def detect(self, rgb_u8: np.ndarray | None = None, depth_m: np.ndarray | None = None):
        """rgb_u8: RGB uint8 image, depth_m: float depth in metres.

        Returns (points K x 2 in reference image pixels (corner convention), confidences K, heatmaps).
        """
        tensors, m_inv = pp.prepare_inputs(self.meta, rgb_u8, depth_m)
        hm = self.heatmaps(tensors)
        pts, conf = pp.decode_heatmaps(hm, self.stride, m_inv)
        return pts, conf, hm
