import logging
import os
from dataclasses import dataclass

os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")
logging.getLogger("insightface").setLevel(logging.ERROR)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from insightface.app import FaceAnalysis  # noqa: E402

from . import config


@dataclass
class DetectedFace:
    bbox: list          # [x1, y1, x2, y2] in original image px
    det_score: float
    embedding: np.ndarray  # unit-normalized float32 (512,)
    thumb_bgr: np.ndarray | None = None


class FaceEngine:
    """InsightFace buffalo_l detection + ArcFace recognition on CPU."""

    def __init__(self, det_size=config.DET_SIZE):
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
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

    def detect(self, bgr: np.ndarray) -> list[DetectedFace]:
        faces: list[DetectedFace] = []
        for f in self.app.get(bgr):
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
