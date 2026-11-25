"""ONNX keypoint detector without torch (onnxruntime, falls back to cv2.dnn)."""

from pathlib import Path

import cv2
import numpy as np

from . import preprocess as pp


_GPU_PROVIDERS = ('CUDAExecutionProvider', 'TensorrtExecutionProvider', 'ROCMExecutionProvider',
                  'MIGraphXExecutionProvider', 'OpenVINOExecutionProvider', 'DmlExecutionProvider')


def _ort_device(providers) -> str:
    """'GPU (<provider>)' if the session's first provider is a GPU one, else 'CPU'."""
    first = providers[0] if providers else 'CPUExecutionProvider'
    return f'GPU ({first})' if first in _GPU_PROVIDERS else 'CPU'


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
                # GPU first; onnxruntime silently drops providers it cannot load, so the session's
                # actual providers (-> self.device) tell where it really runs
                available = ort.get_available_providers()
                if 'CUDAExecutionProvider' in available and hasattr(ort, 'preload_dlls'):
                    # load CUDA/cuDNN from the pip nvidia-* packages (onnxruntime-gpu[cuda,cudnn])
                    # if installed; otherwise the system libraries are used
                    try:
                        ort.preload_dlls()
                    except Exception:
                        pass
                providers = [p for p in ('CUDAExecutionProvider',) if p in available]
                providers.append('CPUExecutionProvider')
                self._session = ort.InferenceSession(str(self.path), providers=providers)
                self.backend = 'onnxruntime'
                self.device = _ort_device(self._session.get_providers())
            except ImportError:
                if backend == 'onnxruntime':
                    raise
        if self.backend is None:
            self._net = cv2.dnn.readNetFromONNX(str(self.path))
            self.backend = 'cv2.dnn'
            self.device = 'CPU'
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                self.device = 'GPU (cv2.dnn CUDA)'

    @property
    def on_gpu(self) -> bool:
        return self.device.startswith('GPU')

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
