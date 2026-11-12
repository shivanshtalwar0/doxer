import logging
import os
from dataclasses import dataclass

os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")
logging.getLogger("insightface").setLevel(logging.ERROR)

import threading  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from insightface.app import FaceAnalysis  # noqa: E402

from . import config

_INFER_LOCK = threading.Lock()


def _available_providers() -> list[str]:
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:  # noqa: BLE001
        return ["CPUExecutionProvider"]


def _pick_providers() -> list[str]:
    """Prefer CoreML, then CUDA, then CPU."""
    avail = _available_providers()
    chosen: list[str] = []
    for p in ("CoreMLExecutionProvider", "CUDAExecutionProvider"):
        if p in avail:
            chosen.append(p)
            break
    chosen.append("CPUExecutionProvider")
    return chosen


def _accelerator_name(providers: list[str]) -> str:
    if "CoreMLExecutionProvider" in providers:
        return "coreml"
    if "CUDAExecutionProvider" in providers:
        return "cuda"
    return "cpu"


@dataclass
class DetectedFace:
    bbox: list          # [x1, y1, x2, y2] in original image px
    det_score: float
    embedding: np.ndarray  # unit-normalized float32 (512,)
    thumb_bgr: np.ndarray | None = None


class FaceEngine:
    """InsightFace buffalo_l detection + ArcFace recognition.

    Uses CoreML (Apple Silicon GPU/ANE) when available, falling back to CPU.
    Detection stays on CPU: CoreML is often slower for the variable-size det
    pass, while recognition (fixed 112x112) benefits a lot.
    """

    def __init__(self, det_size=config.DET_SIZE, providers: list[str] | None = None):
        self.providers = providers or _pick_providers()
        self.app = FaceAnalysis(name="buffalo_l", providers=self.providers)
        self.app.prepare(ctx_id=-1, det_size=det_size)

    @staticmethod
    def _crop_thumb(bgr: np.ndarray, bbox: list, size: int = config.THUMB_SIZE) -> np.ndarray:
        h, w = bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        pad = int((x2 - x1) * 0.18)
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return crop
        scale = size / max(crop.shape[:2])
        nh, nw = int(crop.shape[0] * scale), int(crop.shape[1] * scale)
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        return cv2.resize(crop, (nw, nh), interpolation=interp)

    @staticmethod
    def shrink(bgr: np.ndarray, max_side: int = 520) -> np.ndarray:
        """Downscale a whole image for the live preview (keeps it small enough to stream)."""
        if bgr is None or not getattr(bgr, "size", 0):
            return bgr
        h, w = bgr.shape[:2]
        longest = max(h, w)
        if longest <= max_side:
            return bgr
        scale = max_side / float(longest)
        return cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

    def detect(self, bgr: np.ndarray) -> list[DetectedFace]:
        # onnxruntime sessions are not safe to call concurrently; the sweep runs
        # several workers, so serialise inference (it is fast relative to I/O).
        faces: list[DetectedFace] = []
        with _INFER_LOCK:
            raw = self.app.get(bgr)
        for f in raw:
            emb = np.asarray(f.normed_embedding, dtype=np.float32)
            n = float(np.linalg.norm(emb))
            if n < 1e-9:
                continue
            emb /= n
            bbox = [int(v) for v in f.bbox]
            faces.append(
                DetectedFace(
                    bbox=bbox,
                    det_score=float(f.det_score),
                    embedding=emb,
                    thumb_bgr=self._crop_thumb(bgr, bbox),
                )
            )
        return faces
