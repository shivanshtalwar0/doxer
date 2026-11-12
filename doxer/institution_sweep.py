"""Institution mode: city -> schools/colleges -> websites -> galleries -> faces.

A different search shape from the social sweep. There is no handle to guess:
institutions are found on a map, their websites are crawled for event galleries,
and every face in those galleries is compared against the query photo.

Only high-confidence hits are reported. Gallery photos are group shots at mixed
resolution, so the usual "possible" band would bury the real result in noise;
INSTITUTION_MIN_SCORE gates what surfaces.

A name is contextual, not a filter: the query is matched on faces. The name is
recorded with the hit so a human can confirm the face belongs to the person
they were looking for.
"""

import base64
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import config, db
from .galleries import crawl_site
from .institutions import SEARCH_TERMS, search as search_places
from .pipeline import Pipeline


def _thumb_b64(bgr, max_side: int = 480) -> str:
    if bgr is None or not getattr(bgr, "size", 0):
        return ""
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest > max_side:
        scale = max_side / float(longest)
        bgr = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return base64.b64encode(enc.tobytes()).decode() if ok else ""


class InstitutionRunner:
    """One institution-mode run, recorded as a session."""

    def __init__(self, session_id: str, query_image: str, attrs: dict,
                 pipeline: Pipeline | None = None):
        self.sid = session_id
        self.query_image = query_image
        self.attrs = attrs or {}
        self.pipe = pipeline or Pipeline()
        self.cancelled = False
        self._lock = None
        self._best = 0.0
        self._n_images = 0
        self._n_faces = 0
        self._n_profiles = 0
        self._qfaces = []
        bgr = cv2.imread(str(query_image)) if query_image else None
        if bgr is not None:
            self._qfaces = self.pipe.engine.detect(bgr)

    # ------------------------------------------------------------------ util

    def cancel(self) -> None:
        self.cancelled = True

    def _set(self, stage: str | None = None, message: str | None = None, **counts) -> None:
        fields = {k: v for k, v in counts.items() if v is not None}
        if stage:
            fields["stage"] = stage
        if message is not None:
            fields["message"] = message
        db.update_session(self.sid, **fields)

    def _act(self, label: str, step: str = "", state: str = "run", pct: int = 0) -> None:
        db.log_progress("sweep", self.sid, label, step=step, state=state, pct=pct)

    # ------------------------------------------------------------------- run

    def run(self, city: str, kinds: list[str], name: str = "",
            max_places: int = config.INSTITUTION_MAX_PLACES,
            max_sites: int = config.INSTITUTION_MAX_SITES,
            max_images: int = config.INSTITUTION_MAX_IMAGES,
            workers: int = config.INSTITUTION_WORKERS) -> dict:
        import threading

        self._lock = threading.Lock()
        try:
            return self._run(city, kinds, name, max_places, max_sites, max_images, workers)
        except BaseException as e:  # noqa: BLE001
            db.update_session(self.sid, status="error", stage="done",
                              message=f"{type(e).__name__}: {e}")
            raise

    def _run(self, city, kinds, name, max_places, max_sites, max_images, workers) -> dict:
        kinds = [k for k in kinds if k in SEARCH_TERMS] or ["school", "college"]
        label = " + ".join(kinds)
        self._set("probing", f"finding {label} in {city}")
        self._act(f'looking for {label} in "{city}"'
                  + (f" (face of {name})" if name else ""), "start", "run", 2)

        if not self._qfaces:
            self._set(status="error", stage="done", message="no face detected in the query photo")
            self._act("no face detected in the query photo — nothing to match", "start", "error", 0)
            return {"status": "error", "reason": "no query face"}

        places = search_places(
            city, kinds=kinds, max_places=max_places, want_websites=True,
            on_progress=lambda m: self._act(m, "maps", "run", 10),
        )
        with_sites = [p for p in places if p.get("website")][:max_sites]
        db.update_session(self.sid, n_candidates=len(places))
        self._act(f"{len(places)} institution(s) found, {len(with_sites)} with a website",
                  "maps", "ok", 25)
        if not with_sites:
            self._set(status="done", stage="done",
                      message=f"no {label} websites found in {city}")
            self._act("no websites to crawl", "maps", "warn", 25)
            return {"status": "done", "places": len(places), "sites": 0}

        self._set("indexing", f"crawling {len(with_sites)} website(s)")

        def do_site(job: tuple[int, dict]) -> None:
            i, place = job
            if self.cancelled:
                return
            pct = 25 + int(65 * i / max(1, len(with_sites)))
            self._act(f"[{i}/{len(with_sites)}] {place['name']}: opening site",
                      "site", "run", pct)
            gallery = crawl_site(place["website"], max_images=max_images)
            imgs = gallery.get("images") or []
            if gallery.get("error"):
                self._act(f"{place['name']}: site error ({gallery['error'][:60]})",
                          "site", "warn", pct)
            if not imgs:
                with self._lock:
                    self._n_profiles += 1
                    counts = (self._n_profiles, self._n_images, self._n_faces)
                self._set(n_profiles=counts[0], n_images=counts[1], n_faces=counts[2])
                self._record(place, name, images=0, faces=0, best=0.0, status="no-images")
                self._act(f"{place['name']}: no usable photos", "site", "warn", pct)
                return
            faces_total, best = self._match_images(place, name, imgs)
            with self._lock:
                self._n_profiles += 1
                self._n_images += len(imgs)
                self._n_faces += faces_total
                counts = (self._n_profiles, self._n_images, self._n_faces)
            self._set(n_profiles=counts[0], n_images=counts[1], n_faces=counts[2])
            self._act(f"{place['name']}: {len(imgs)} photos, {faces_total} faces, "
                      f"best {best:.3f}", "site", "ok" if best >= config.INSTITUTION_MIN_SCORE else "run", pct)

        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            list(ex.map(do_site, enumerate(with_sites, 1)))

        self._finalize()
        return {"status": "done", "places": len(places), "sites": len(with_sites),
                "best": self._best}

    # --------------------------------------------------------------- pieces

    def _match_images(self, place: dict, name: str, urls: list[str]) -> tuple[int, float]:
        faces_total = 0
        best = 0.0
        for url in urls:
            if self.cancelled:
                break
            bgr, w, h = self.pipe.download(url)
            if bgr is None:
                continue
            faces = self.pipe.engine.detect(bgr)
            faces_total += len(faces)
            if not faces:
                continue
            img_best = 0.0
            for f in faces:
                for q in self._qfaces:
                    img_best = max(img_best, float(np.dot(f.embedding, q.embedding)))
            best = max(best, img_best)
            db.add_session_check(
                self.sid, profile_id=self._pid(place), image_path=url,
                thumb_b64=_thumb_b64(bgr), n_faces=len(faces), best_score=img_best,
                matched=1 if img_best >= config.INSTITUTION_MIN_SCORE else 0, stage="done",
            )
            self._record_if_better(place, name, img_best,
                                   faces=faces_total, images=len(urls))
        return faces_total, best

    @staticmethod
    def _pid(place: dict) -> str:
        import re

        slug = re.sub(r"[^a-z0-9]+", "_", (place.get("name") or "site").lower()).strip("_")[:48]
        return f"{place.get('kind', 'school')}_{slug}"

    def _record(self, place: dict, name: str, *, images: int, faces: int, best: float,
                status: str) -> None:
        db.add_session_profile(
            self.sid, self._pid(place),
            platform="web",
            username=place.get("website", ""),
            full_name=place.get("name", ""),
            bio=place.get("address", ""),
            city=place.get("city", ""),
            n_images=images,
            n_faces=faces,
            best_score=best,
            rank_score=best,
            status=status,
            source_url=place.get("maps_url", "") or place.get("website", ""),
            discovered_via="maps",
            raw={"website": place.get("website", ""), "kind": place.get("kind", ""),
                 "query_name": name, "phone": place.get("phone", "")},
        )

    def _record_if_better(self, place: dict, name: str, score: float,
                          faces: int = 0, images: int = 0) -> None:
        """Upsert the site's best score so every crawled site is represented."""
        with self._lock:
            improved = score > self._best
            if improved:
                self._best = score
        if improved:
            self._set(best_score=score)
        self._record(place, name, images=images or 1, faces=faces, best=score,
                     status="hit" if score >= config.INSTITUTION_MIN_SCORE else "indexed")

    def _finalize(self) -> None:
        best = self._best
        if best >= config.SCORE_STRONG:
            verdict = "STRONG"
        elif best >= config.INSTITUTION_MIN_SCORE:
            verdict = "probable"
        else:
            verdict = "no match"
        self._set(status="done", stage="done", best_score=best,
                  message=f"done — best {best:.3f} ({verdict})")
        self._act(f"finished — best score {best:.3f} ({verdict})", "done",
                  "ok" if best >= config.INSTITUTION_MIN_SCORE else "warn", 100)
