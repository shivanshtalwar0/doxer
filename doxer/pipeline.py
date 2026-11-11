"""Indexing (scrape -> download -> detect faces -> store) and search."""

import base64
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from . import config, db
from .engine import FaceEngine
from .scraper import ScrapeError, download_image, profile_id_for, scrape_profile


class Pipeline:
    def __init__(self):
        db.init()
        self.engine = FaceEngine()
        self._search_lock = threading.Lock()

    # ----------------------------------------------------------------- index

    def add_profile(self, url: str, job_id: str | None = None, full_images: int = 6) -> dict:
        if job_id is None:
            job_id = db.new_job("add")
        try:
            info = scrape_profile(url, full_images=full_images)
            pid = profile_id_for(info["platform"], info["username"])
            self._reset_profile(pid)
            db.upsert_profile(pid, info["platform"], info["username"], url.strip(), info.get("name", ""))

            n_imgs = 0
            n_faces = 0
            avatar_set = False
            img_urls = list(info["images"])
            av = info.get("avatar")
            if av and av not in img_urls:
                img_urls.insert(0, av)  # profile pic is usually the best face shot
            for i, img_url in enumerate(img_urls):
                bgr, w, h = download_image(img_url)
                if bgr is None:
                    continue
                path = config.IMAGES_DIR / pid / f"{i:03d}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
                image_id = db.add_image(pid, img_url, str(path.relative_to(config.DATA)), w, h)

                if not avatar_set and i == 0:
                    db.set_avatar(pid, str(path.relative_to(config.DATA)))
                    avatar_set = True

                faces = self.engine.detect(bgr)
                for j, f in enumerate(faces):
                    thumb = config.IMAGES_DIR / pid / f"{i:03d}_f{j}.jpg"
                    if f.thumb_bgr is not None and f.thumb_bgr.size:
                        cv2.imwrite(str(thumb), f.thumb_bgr)
                    db.add_face(
                        image_id,
                        pid,
                        f.bbox,
                        f.det_score,
                        f.embedding.tobytes(),
                        str(thumb.relative_to(config.DATA)) if f.thumb_bgr is not None else "",
                    )
                    n_faces += 1
                n_imgs += 1

            db.set_job(job_id, "done", f"indexed {n_imgs} images, {n_faces} faces", {"profile": pid})
            return db.get_profile(pid) or {}
        except ScrapeError as e:
            db.set_job(job_id, "error", str(e))
            raise
        except Exception as e:  # noqa: BLE001
            db.set_job(job_id, "error", f"{type(e).__name__}: {e}")
            raise

    def _reset_profile(self, pid: str) -> None:
        db.delete_profile(pid)
        d = config.IMAGES_DIR / pid
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)

    # ----------------------------------------------------------------- search

    def search_image_bytes(self, data: bytes, top_k: int = config.SEARCH_TOP_K) -> dict:
        arr = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("could not decode image")
        return self._search_bgr(bgr, top_k)

    def search_image_path(self, path: str, top_k: int = config.SEARCH_TOP_K) -> dict:
        bgr = cv2.imread(path)
        if bgr is None:
            raise ValueError(f"could not read image: {path}")
        return self._search_bgr(bgr, top_k)

    def _search_bgr(self, bgr: np.ndarray, top_k: int) -> dict:
        with self._search_lock:
            query_faces = self.engine.detect(bgr)
            if not query_faces:
                return {"query_faces": [], "matches": []}

            rows = db.all_faces()
            profiles = {p["id"]: p for p in db.list_profiles()}
            n = len(rows)
            if n == 0:
                return {"query_faces": [], "matches": []}

            M = np.empty((n, 512), dtype=np.float32)
            for i, r in enumerate(rows):
                M[i] = np.frombuffer(r["embedding"], dtype=np.float32)

            best_per_profile: dict[str, dict] = {}
            for q in query_faces:
                sims = M @ q.embedding
                k = min(top_k * 3, n)
                idx = np.argpartition(-sims, k - 1)[:k] if k < n else np.arange(n)
                idx = idx[np.argsort(-sims[idx])][:top_k]
                for i in idx:
                    r = rows[i]
                    s = float(sims[i])
                    if s < config.SEARCH_MIN_SCORE:
                        continue
                    cur = best_per_profile.get(r["profile_id"])
                    if cur is None or s > cur["score"]:
                        best_per_profile[r["profile_id"]] = {
                            "profile": r["profile_id"],
                            "score": s,
                            "face_thumb": r["thumb_path"],
                            "image_url": r["image_url"],
                        }

            matches = sorted(best_per_profile.values(), key=lambda m: -m["score"])[:20]
            for m in matches:
                p = profiles.get(m["profile"], {})
                m.update(
                    {
                        "platform": p.get("platform", ""),
                        "username": p.get("username", ""),
                        "display_name": p.get("display_name", ""),
                        "source_url": p.get("source_url", ""),
                        "avatar": p.get("avatar_path", ""),
                    }
                )

            qf = []
            for j, f in enumerate(query_faces):
                src = f.thumb_bgr if (f.thumb_bgr is not None and f.thumb_bgr.size) else bgr
                ok, enc = cv2.imencode(".jpg", src)
                qf.append(
                    {
                        "bbox": f.bbox,
                        "det_score": round(f.det_score, 3),
                        "thumb_b64": base64.b64encode(enc.tobytes()).decode() if ok and f.thumb_bgr is not None else "",
                    }
                )
            return {"query_faces": qf, "matches": matches}

    # ------------------------------------------------------------------ misc

    def stats(self) -> dict:
        profs = db.list_profiles()
        total_images = sum(p["n_images"] for p in profs)
        total_faces = sum(p["n_faces"] for p in profs)
        return {"profiles": len(profs), "images": total_images, "faces": total_faces}
