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

import cv2
import numpy as np

from . import config, db
from .candidates import from_query
from .pipeline import Pipeline

ATTR_KEYS = ("city", "education", "school", "college", "employer", "workplace", "hometown")


def _norm(s: str) -> str:
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def attribute_score(attrs: dict, *, full_name: str = "", bio: str = "", username: str = "") -> tuple[float, list[str]]:
    """How well a profile matches the supplied attributes. Returns (0..1, matched keys)."""
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


def _thumb_b64(bgr) -> str:
    if bgr is None or not getattr(bgr, "size", 0):
        return ""
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(enc.tobytes()).decode() if ok else ""


class SweepRunner:
    """Runs one sweep session. Not reusable; one instance per run."""

    def __init__(self, session_id: str, query_image: str, attrs: dict, pipe: Pipeline | None = None):
        self.sid = session_id
        self.query_image = query_image
        self.attrs = attrs or {}
        self.pipe = pipe or Pipeline()
        self.cancelled = False
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
        follow_network: bool = True,
        network_max: int = config.SWEEP_NETWORK_MAX,
        workers: int = config.SWEEP_WORKERS,
    ) -> dict:
        try:
            return self._run(query, platforms, limit, max_photos, follow_network, network_max, workers)
        except Exception as e:  # noqa: BLE001
            db.update_session(self.sid, status="error", message=f"{type(e).__name__}: {e}")
            raise

    def _run(self, query, platforms, limit, max_photos, follow_network, network_max, workers) -> dict:
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

        # ---- 2. index every hit
        n_prof = n_img = n_faces = 0
        indexed: list[dict] = []
        total = max(1, len(hits))
        for i, r in enumerate(hits, 1):
            if self.cancelled:
                return self._finish_cancelled()
            pct = 40 + int(45 * i / total)
            self._act(f"scraping profile {i}/{total}: @{r['username']}", "profile", "run", pct)
            try:
                prof = self._index(r["username"], max_photos)
            except Exception as e:  # noqa: BLE001
                db.add_session_profile(self.sid, f"instagram_{r['username']}", username=r["username"],
                                       status="error", raw={"error": str(e)[:200]})
                self._act(f"@{r['username']} failed: {type(e).__name__}", "profile", "warn", pct)
                continue
            if prof is None:
                self._act(f"@{r['username']} already seen — skipped", "profile", "ok", pct)
                continue
            indexed.append(prof)
            n_prof += 1
            n_img += prof["n_images"]
            n_faces += prof["n_faces"]
            best = self._best_by_profile.get(prof["pid"], 0.0)
            self._act(
                f"@{r['username']}: {prof['n_images']} photos, {prof['n_faces']} faces, best {best:.3f}",
                "profile", "ok", pct,
            )
            self._set(n_profiles=n_prof, n_images=n_img, n_faces=n_faces)

        # ---- 3. follow the network
        if follow_network and not self.cancelled:
            self._set("networking", "following related accounts")
            self._act("looking for related accounts", "network", "run", 85)
            append = self._related_usernames(indexed)[:network_max]
            self._act(f"{len(append)} related accounts found", "network", "ok", 86)
            for j, u in enumerate(append, 1):
                if self.cancelled:
                    break
                self._act(f"related {j}/{len(append)}: @{u}", "network", "run", 86 + int(4 * j / max(1, len(append))))
                try:
                    prof = self._index(u, min(max_photos, 12), via="network")
                except Exception:  # noqa: BLE001
                    continue
                if prof:
                    n_prof += 1
                    n_img += prof["n_images"]
                    n_faces += prof["n_faces"]
                    self._set(n_profiles=n_prof, n_images=n_img, n_faces=n_faces)

        # ---- 4. rank
        self._act("comparing every indexed face against your photo", "match", "run", 92)
        best = self._finalize()
        band = "STRONG" if best >= 0.60 else ("possible" if best >= 0.45 else "no match")
        state = "ok" if best >= 0.60 else ("warn" if best >= 0.45 else "ok")
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
                hits += [dict(r, platform="instagram")
                         for r in probe_authed(
                             cands, workers=workers, on_result=report,
                             rate=config.IG_RATE, burst=config.IG_BURST,
                         )
                         if r.get("exists")]
            except LoginRequired as e:
                self._set(message=f"instagram needs a login: {e}")
                self._act(f"instagram skipped — needs a login: {e}", "probe", "error", 4)

        others = [p for p in platforms if p != "instagram"]
        if others:
            from .scraper import probe_usernames

            for plat in others:
                self._act(f"checking {total} handles on {plat}…", "probe", "run", 30)
                targets = [(plat, c) for c in cands]
                for r in probe_usernames(targets, workers=workers):
                    u = r.get("username", "?")
                    if r.get("exists"):
                        hits.append(dict(r))
                        self._act(f"FOUND @{u} on {plat}", "probe", "ok", 38)
        return hits

    def _index(self, username: str, max_photos: int, via: str = "probe") -> dict | None:
        """Scrape + index one profile, recording a progress row per image."""
        from .instagram import scrape_authed

        uname = username.lower()
        if uname in self._seen:
            return None
        self._seen.add(uname)

        self._act(f"@{uname}: loading profile page", "scrape", "run")
        info = scrape_authed(uname)
        pid = f"instagram_{uname}"
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
        for k, url in enumerate(urls, 1):
            if self.cancelled:
                break
            self._act(f"@{uname}: photo {k}/{n_total} — downloading", "image", "run")
            bgr, w, h = self.pipe.download(url)
            if bgr is None:
                self._act(f"@{uname}: photo {k}/{n_total} — skipped (not an image)", "image", "warn")
                continue

            # Push the image to the UI the moment it lands, before detection,
            # so the view shows what is being worked on rather than the result.
            thumb = _thumb_b64(self.pipe.engine.shrink(bgr))
            check_id = db.add_session_check(
                self.sid, profile_id=pid, image_path=url, thumb_b64=thumb,
                n_faces=0, best_score=0.0, matched=False, stage="detecting",
            )
            self._act(f"@{uname}: photo {k}/{n_total} — detecting faces", "image", "run")

            faces = self.pipe.engine.detect(bgr)
            n_img += 1
            n_faces += len(faces)

            img_score = 0.0
            if faces and self._qfaces:
                for f in faces:
                    for q in self._qfaces:
                        s = float(np.dot(f.embedding, q.embedding))
                        img_score = max(img_score, s)

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
            self._best_by_profile[pid] = max(self._best_by_profile.get(pid, 0.0), img_score)
            state = "ok" if img_score >= config.SCORE_POSSIBLE else "run"
            self._act(
                f"@{uname}: photo {k}/{n_total} — {len(faces)} face(s), best {img_score:.3f}",
                "image", state,
            )

        a_score, a_hits = attribute_score(self.attrs, full_name=info.get("name", ""),
                                          bio=info.get("bio", ""), username=uname)
        db.add_session_profile(
            self.sid, pid,
            platform="instagram", username=uname,
            full_name=info.get("name", ""), bio=info.get("bio", ""),
            city=self.attrs.get("city", ""),
            education=self.attrs.get("education", "") or self.attrs.get("school", ""),
            followers=int(info.get("followers") or 0),
            n_images=n_img, n_faces=n_faces,
            best_score=profile_best,
            status="indexed" if n_faces else "no-faces",
            avatar_path=db.get_profile(pid).get("avatar_path", "") if db.get_profile(pid) else "",
            source_url=f"https://www.instagram.com/{uname}/",
            discovered_via=via,
            raw={"attribute_score": a_score, "attribute_hits": a_hits},
        )
        return {"pid": pid, "username": uname, "n_images": n_img, "n_faces": n_faces,
                "attr_score": a_score}

    def _related_usernames(self, indexed: list[dict]) -> list[str]:
        """Accounts suggested on the profiles we already have ('similar accounts')."""
        from .instagram import related_accounts

        out: list[str] = []
        for prof in indexed:
            if self.cancelled:
                break
            try:
                for u in related_accounts(prof["username"]):
                    ul = u.lower()
                    if ul not in self._seen and ul not in out:
                        out.append(ul)
            except Exception:  # noqa: BLE001
                continue
        return out

    def _finalize(self) -> float:
        best = max(self._best_by_profile.values()) if self._best_by_profile else 0.0
        db.update_session(self.sid, status="done", stage="done", best_score=best,
                          message=f"done — best score {best:.3f}")
        return best

    def _finish_cancelled(self) -> dict:
        db.update_session(self.sid, status="cancelled", stage="done", message="cancelled")
        return {"status": "cancelled"}
