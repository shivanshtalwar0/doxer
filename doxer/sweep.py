"""Exhaustive sweep: probe(name variants) -> index -> follow network -> match.

Every run is recorded as a *session* so it can be replayed image-by-image in the
UI, and so profiles seen in one run stay searchable in later ones.

Attributes (city, education, school, employer, ...) are used two ways:
  1. as handle constituents (a city often appears in the username), and
  2. as a relevance signal — a profile whose bio/name contains the city or
     school ranks above one that doesn't, which is what keeps a 400-profile
     sweep usable instead of a wall of anonymous hits.
"""

import base64
import threading

import cv2
import numpy as np

from . import config, db
from .candidates import from_query
from concurrent.futures import ThreadPoolExecutor

from .pipeline import Pipeline

ATTR_KEYS = ("city", "education", "school", "college", "employer", "workplace", "hometown")


def _norm(s: str) -> str:
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def attribute_score(attrs: dict, *, full_name: str = "", bio: str = "", username: str = "") -> tuple[float, list[str]]:
    """How well a profile matches the supplied attributes.

    Returns (0..1, matched keys). The denominator counts only attributes the
    caller actually supplied, so a 3-attribute query and a 1-attribute query are
    both scored out of what was asked for.
    """
    haystacks = [_norm(full_name), _norm(bio), _norm(username)]
    hits: list[str] = []
    checks = 0
    for key in ATTR_KEYS:
        val = _norm((attrs or {}).get(key) or "")
        if len(val) < 3:
            continue
        checks += 1
        if any(val in h for h in haystacks):
            hits.append(key)
    if not checks:
        return 0.0, hits
    return len(hits) / checks, hits


# How much attribute evidence can lift a candidate.
#
# Bio text is weak evidence: anyone can write "Muzaffarnagar". So it may only
# reorder profiles that are already plausible on the face -- it must never move
# a profile into a higher band. 0.59 is still "possible"; a 0.05 flat bonus
# would push it to "strong" on the strength of a word in a bio. The lift is
# therefore capped by the actual headroom to the next boundary.
ATTR_BONUS_MIN_FACE = config.SCORE_POSSIBLE
ATTR_BONUS_FRACTION = 0.10        # up to 10% of the room before the next band


def _next_boundary(score: float) -> float | None:
    """The band threshold above `score`, or None if already at the top band."""
    if score < config.SCORE_POSSIBLE:
        return config.SCORE_POSSIBLE
    if score < config.SCORE_STRONG:
        return config.SCORE_STRONG
    return None


def combined_rank(face_score: float, attr_score: float) -> float:
    """Blend face similarity with attribute evidence for ordering results.

    Bounded so that attributes can only reorder within a band, never promote a
    profile across one: ordering evidence must not change the verdict.
    """
    if face_score < ATTR_BONUS_MIN_FACE or attr_score <= 0.0:
        return face_score
    boundary = _next_boundary(face_score)
    if boundary is None:
        return face_score                     # already STRONG: nothing to add
    headroom = boundary - face_score
    return face_score + min(headroom * ATTR_BONUS_FRACTION * attr_score,
                            headroom * 0.5)


def _thumb_b64(bgr) -> str:
    if bgr is None or not getattr(bgr, "size", 0):
        return ""
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(enc.tobytes()).decode() if ok else ""


class _Cancelled(Exception):
    """Raised inside worker callbacks to unwind a cancelled run."""


class SweepRunner:
    """Runs one sweep session. Not reusable; one instance per run."""

    def __init__(self, session_id: str, query_image: str, attrs: dict, pipe: Pipeline | None = None):
        self.sid = session_id
        self.query_image = query_image
        self.attrs = attrs or {}
        self.pipe = pipe or Pipeline()
        self.cancelled = False
        self._lock = threading.Lock()      # guards _seen, counters, best map
        self._db_lock = threading.Lock()   # sqlite writes from worker threads
        self._seen: set[str] = set()
        self._qfaces = []
        bgr = cv2.imread(str(query_image)) if query_image else None
        if bgr is not None:
            self._qfaces = self.pipe.engine.detect(bgr)
        self._best_by_profile: dict[str, float] = {}

    # ------------------------------------------------------------------ utils

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
        """One line in the live activity feed: what is happening right now."""
        db.log_progress("sweep", self.sid, label, step=step, state=state, pct=pct)

    # -------------------------------------------------------------------- run

    def run(
        self,
        query: str,
        platforms: list[str],
        *,
        limit: int | None = None,
        max_photos: int = config.SWEEP_MAX_PHOTOS,
        follow_network: bool = False,
        network_max: int = config.SWEEP_NETWORK_MAX,
        workers: int = config.SWEEP_WORKERS,
        index_workers: int = config.INDEX_WORKERS,
    ) -> dict:
        try:
            return self._run(query, platforms, limit, max_photos, follow_network,
                             network_max, workers, index_workers)
        except _Cancelled:
            return self._finish_cancelled()
        except BaseException as e:  # noqa: BLE001
            db.update_session(self.sid, status="error", stage="done",
                              message=f"{type(e).__name__}: {e}")
            raise

    def _run(self, query, platforms, limit, max_photos, follow_network, network_max,
             workers, index_workers) -> dict:
        cands = from_query(query, city=self.attrs.get("city") or None, limit=limit)
        db.update_session(self.sid, n_candidates=len(cands))
        self._set("probing", f"probing {len(cands)} candidate handles")
        self._act(f'looking for "{query}" — {len(cands)} handle variants to check', "start", "run", 2)
        self._act("platforms: " + ", ".join(platforms), "start", "ok", 3)

        # ---- 1. probe
        hits = self._probe(cands, platforms, workers)
        if self.cancelled:
            return self._finish_cancelled()
        self._act(f"{len(hits)} of {len(cands)} handles exist", "probe_done", "ok", 40)
        self._set("indexing", f"{len(hits)} handles exist — indexing")

        # ---- 2. index every hit, in parallel
        # I/O (page loads, image downloads) dominates here, so a thread pool
        # gives a large speedup; face inference itself is serialised in the
        # engine. Progress is written per image, so the UI updates continuously.
        n_prof = n_img = n_faces = 0
        indexed: list[dict] = []
        lock = self._lock
        icount = [0]

        def index_one(job: tuple[int, dict]) -> dict | None:
            i, r = job
            if self.cancelled:
                return None
            platform = r.get("platform", "instagram")
            uname = r["username"]
            pct = 40 + int(45 * i / max(1, total))
            with lock:
                icount[0] += 1
                started = icount[0]
            try:
                prof = self._index(uname, max_photos, platform=platform)
            except Exception as e:  # noqa: BLE001
                with self._db_lock:
                    db.add_session_profile(self.sid, f"{platform}_{uname}", platform=platform,
                                           username=uname, status="error",
                                           raw={"error": str(e)[:200]})
                self._act(f"{platform} @{uname} failed: {type(e).__name__}", "profile", "warn", pct)
                return None
            if prof is None:
                self._act(f"{platform} @{uname} already seen — skipped", "profile", "ok", pct)
                return None
            with lock:
                indexed.append(prof)
            best = self._best_by_profile.get(prof["pid"], 0.0)
            lift = ""
            if prof.get("attr_score"):
                lift = f", attrs {prof['attr_score']:.2f}"
            self._act(
                f"{platform} @{uname}: {prof['n_images']} photos, {prof['n_faces']} faces, "
                f"best {best:.3f}{lift}  [{started}/{total}]",
                "profile", "ok", pct,
            )
            with lock:
                self._set(n_profiles=len(indexed),
                          n_images=sum(x["n_images"] for x in indexed),
                          n_faces=sum(x["n_faces"] for x in indexed))
            return prof

        total = max(1, len(hits))
        workers = max(1, min(index_workers, len(hits) or 1))
        if workers > 1:
            self._act(f"indexing {len(hits)} profiles with {workers} workers", "index_start", "run", 40)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(index_one, enumerate(hits, 1)))

        if self.cancelled:
            return self._finish_cancelled()
        n_prof = len(indexed)
        n_img = sum(x["n_images"] for x in indexed)
        n_faces = sum(x["n_faces"] for x in indexed)
        self._set(n_profiles=n_prof, n_images=n_img, n_faces=n_faces)

        # ---- 3. follow the network (opt-in)
        if not follow_network and not self.cancelled:
            self._act("network search off — skipping related accounts", "network", "ok", 85)
        if follow_network and not self.cancelled:
            self._set("networking", "following related accounts")
            self._act(f"searching the network for accounts related to {len(indexed)} profile(s)",
                      "network", "run", 85)
            append = self._related_usernames(indexed)[:network_max]
            if not append:
                self._act("network search found no additional accounts", "network", "ok", 86)
            else:
                self._act(f"{len(append)} related account(s) to index", "network", "ok", 86)
            for j, u in enumerate(append, 1):
                if self.cancelled:
                    break
                self._act(f"related {j}/{len(append)}: @{u}", "network", "run", 86 + int(4 * j / max(1, len(append))))
                try:
                    prof = self._index(u, min(max_photos, config.SWEEP_NETWORK_PHOTOS),
                                       via="network", platform="instagram")
                except Exception as e:  # noqa: BLE001
                    db.add_session_profile(self.sid, f"instagram_{u}", platform="instagram",
                                           username=u, status="error",
                                           discovered_via="network",
                                           raw={"error": f"{type(e).__name__}: {e}"[:200]})
                    self._act(f"related @{u} failed: {type(e).__name__}", "network", "warn")
                    continue
                if prof:
                    n_prof += 1
                    n_img += prof["n_images"]
                    n_faces += prof["n_faces"]
                    self._set(n_profiles=n_prof, n_images=n_img, n_faces=n_faces)

        # ---- 4. rank
        self._act("comparing every indexed face against your photo", "match", "run", 92)
        best = self._finalize()
        band = config.band(best) if best >= config.SCORE_POSSIBLE else "no match"
        state = "ok" if best >= config.SCORE_STRONG else "warn"
        self._act(f"finished — best score {best:.3f} ({band})", "done", state, 100)
        return {"status": "done", "n_profiles": n_prof, "n_images": n_img, "n_faces": n_faces, "best": best}

    # ---------------------------------------------------------------- steps

    def _probe(self, cands: list[str], platforms: list[str], workers: int) -> list[dict]:
        hits: list[dict] = []
        total = max(1, len(cands))

        def report(n: int, tot: int, rec: dict) -> None:
            pct = 4 + int(34 * n / max(1, tot))
            u = rec.get("username", "?")
            if rec.get("exists"):
                self._act(f'FOUND @{u} — {rec.get("name") or "no name"}  ({n}/{tot})',
                          "probe", "ok", pct)
            else:
                self._act(f"@{u} — not found  ({n}/{tot})", "probe", "run", pct)

        if "instagram" in platforms:
            from .instagram import LoginRequired, probe_authed

            try:
                self._act(
                    f"checking {total} handles on instagram "
                    f"({workers} workers, ~{config.IG_RATE:.1f} req/s cap)…",
                    "probe", "run", 4,
                )
                def _report(n, tot, rec):
                    if self.cancelled:
                        raise _Cancelled()
                    report(n, tot, rec)

                try:
                    hits += [dict(r, platform="instagram")
                             for r in probe_authed(
                                 cands, workers=workers, on_result=_report,
                                 rate=config.IG_RATE, burst=config.IG_BURST,
                             )
                             if r.get("exists")]
                except _Cancelled:
                    self._act("stopped by you", "cancelled", "warn", 0)
                    raise
            except LoginRequired as e:
                self._set(message=f"instagram needs a login: {e}")
                self._act(f"instagram skipped — needs a login: {e}", "probe", "error", 4)

        others = [p for p in platforms if p != "instagram"]
        if others:
            from .scraper import probe_usernames

            for plat in others:
                self._act(f"checking {total} handles on {plat}…", "probe", "run", 30)
                targets = [(plat, c) for c in cands]

                def report_other(n, tot, batch, _plat=plat):
                    if self.cancelled:
                        raise _Cancelled()
                    pct2 = 30 + int(8 * n / max(1, tot))
                    r = batch[0] if batch else {}
                    u = r.get("username", "?")
                    if r.get("exists"):
                        self._act(f"FOUND @{u} on {_plat}  ({n}/{tot})", "probe", "ok", pct2)
                    else:
                        self._act(f"@{u} — not found on {_plat}  ({n}/{tot})", "probe", "run", pct2)

                # report_other already logs each hit as it is found; just collect.
                for r in probe_usernames(targets, workers=workers, progress=report_other):
                    if r.get("exists"):
                        hits.append(dict(r))
        return hits

    def _scrape(self, platform: str, username: str) -> dict:
        """Fetch one profile's metadata + photo URLs from its own platform."""
        if platform == "instagram":
            from .instagram import scrape_authed

            return scrape_authed(username)
        from .scraper import scrape_profile

        if platform == "facebook":
            return scrape_profile(f"https://www.facebook.com/{username}")
        return scrape_profile(f"https://www.threads.com/@{username}")

    def _index(self, username: str, max_photos: int, via: str = "probe",
               platform: str = "instagram") -> dict | None:
        """Scrape + index one profile, recording a progress row per image."""
        uname = username.lower()
        key = f"{platform}:{uname}"
        with self._lock:
            if key in self._seen:
                return None
            self._seen.add(key)   # attempted; removed again if the scrape fails
        try:
            return self._index_scraped(platform, uname, key, max_photos, via)
        except Exception:
            with self._lock:
                self._seen.discard(key)   # transient failure must stay retryable
            raise

    def _index_scraped(self, platform: str, uname: str, key: str,
                       max_photos: int, via: str) -> dict | None:
        """Body of _index; runs with the handle already claimed in _seen."""
        self._act(f"{platform} @{uname}: loading profile page", "scrape", "run")
        info = self._scrape(platform, uname)
        pid = f"{platform}_{uname}"
        urls = list(info.get("images") or [])
        avatar = info.get("avatar")
        if avatar and avatar not in urls:
            urls.insert(0, avatar)
        urls = urls[:max_photos]

        nm = info.get("name") or ""
        bio = (info.get("bio") or "")[:60]
        self._act(
            f"@{uname}: {nm or 'no name'}"
            + (f" · {info.get('followers', 0):,} followers" if info.get("followers") else "")
            + (f' · "{bio}"' if bio else "")
            + f" · {len(urls)} photos",
            "scrape", "ok",
        )

        n_img = n_faces = 0
        profile_best = 0.0
        n_total = max(1, len(urls))
        # Download ahead of detection: network latency dominates for photos, so
        # a small prefetch window keeps the engine busy instead of waiting.
        with ThreadPoolExecutor(max_workers=config.INDEX_PREFETCH) as prefetch:
            futures = [(k, u, prefetch.submit(self.pipe.download, u))
                       for k, u in enumerate(urls, 1)]
            for k, url, fut in futures:
                if self.cancelled:
                    break
                self._act(f"{platform} @{uname}: photo {k}/{n_total} — downloading", "image", "run")
                try:
                    bgr, w, h = fut.result(timeout=90)
                except Exception:  # noqa: BLE001
                    bgr = None
                if bgr is None:
                    self._act(f"{platform} @{uname}: photo {k}/{n_total} — skipped (not an image)",
                              "image", "warn")
                    continue

                # Push the image to the UI the moment it lands, before detection,
                # so the view shows what is being worked on rather than the result.
                thumb = _thumb_b64(self.pipe.engine.shrink(bgr))
                check_id = db.add_session_check(
                    self.sid, profile_id=pid, image_path=url, thumb_b64=thumb,
                    n_faces=0, best_score=0.0, matched=False, stage="detecting",
                )
                self._act(f"{platform} @{uname}: photo {k}/{n_total} — detecting faces",
                          "image", "run")

                faces = self.pipe.engine.detect(bgr)
                n_img += 1
                n_faces += len(faces)

                img_score = 0.0
                if faces and self._qfaces:
                    for f in faces:
                        for q in self._qfaces:
                            img_score = max(img_score, float(np.dot(f.embedding, q.embedding)))

                self.pipe.store_image(pid, url, bgr, faces, uname, info.get("name", ""),
                                      f"https://www.instagram.com/{uname}/")
                face = faces[0] if faces else None
                db.update_session_check(
                    check_id,
                    thumb_b64=_thumb_b64(face.thumb_bgr if face else bgr),
                    n_faces=len(faces),
                    best_score=img_score,
                    matched=1 if img_score >= config.SCORE_POSSIBLE else 0,
                    stage="done",
                )
                profile_best = max(profile_best, img_score)
                with self._lock:
                    self._best_by_profile[pid] = max(self._best_by_profile.get(pid, 0.0), img_score)
                state = "ok" if config.band(img_score) != "weak" else "run"
                self._act(
                    f"{platform} @{uname}: photo {k}/{n_total} — {len(faces)} face(s), best {img_score:.3f}",
                    "image", state,
                )

        a_score, a_hits = attribute_score(self.attrs, full_name=info.get("name", ""),
                                          bio=info.get("bio", ""), username=uname)
        db.add_session_profile(
            self.sid, pid,
            platform=platform, username=uname,
            full_name=info.get("name", ""), bio=info.get("bio", ""),
            city=self.attrs.get("city", ""),
            education=self.attrs.get("education", "") or self.attrs.get("school", ""),
            followers=int(info.get("followers") or 0),
            n_images=n_img, n_faces=n_faces,
            best_score=profile_best,
            attr_score=a_score,
            rank_score=combined_rank(profile_best, a_score),
            status="indexed" if n_faces else "no-faces",
            avatar_path=(db.get_profile(pid) or {}).get("avatar_path", ""),
            source_url=(f"https://www.instagram.com/{uname}/" if platform == "instagram"
                        else f"https://www.{platform}.com/{uname}"),
            discovered_via=via,
            raw={"attribute_score": a_score, "attribute_hits": a_hits},
        )
        return {"pid": pid, "username": uname, "platform": platform,
                "n_images": n_img, "n_faces": n_faces, "attr_score": a_score,
                "rank_score": combined_rank(profile_best, a_score)}

    def _related_usernames(self, indexed: list[dict]) -> list[str]:
        """Accounts suggested on the profiles we already have ('similar accounts')."""
        from .instagram import related_accounts

        out: list[str] = []
        for prof in indexed:
            if self.cancelled:
                break
            try:
                found = related_accounts(prof["username"])
            except Exception as e:  # noqa: BLE001
                self._act(f"related @{prof['username']} failed: {type(e).__name__}",
                          "network", "warn")
                continue
            if found:
                self._act(f"@{prof['username']} suggests {len(found)} account(s)", "network", "ok")
            for u in found:
                ul = u.lower()
                if f"instagram:{ul}" not in self._seen and ul not in out:
                    out.append(ul)
        return out

    def _finalize(self) -> float:
        # Score from THIS session's checks, not a global per-profile max: a
        # profile indexed better by an earlier run must not inflate (or, when
        # max_photos truncation hides the matching photo, misreport) this one.
        best = db.session_best_score(self.sid)
        db.update_session(self.sid, status="done", stage="done", best_score=best,
                          message=f"done — best score {best:.3f}")
        return best

    def _finish_cancelled(self) -> dict:
        db.update_session(self.sid, status="cancelled", stage="done", message="stopped")
        self._act("stopped", "cancelled", "warn", 0)
        return {"status": "cancelled"}
