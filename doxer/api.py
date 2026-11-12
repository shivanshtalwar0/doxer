import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

STATIC = Path(__file__).parent / "static"

# Paths under data/ that may be served by /img. Everything else -- notably
# doxer.db with its plaintext credentials -- stays off the wire.
SERVABLE_SUBDIRS = ("images", "queries")


def _safe_data_path(rel: str) -> Path | None:
    """Resolve `rel` inside an allow-listed data/ subdirectory, or None."""
    from . import config

    data_root = config.DATA.resolve()
    candidate = (data_root / rel).resolve()
    try:
        candidate.relative_to(data_root)
    except ValueError:
        return None            # escaped data/ via '..' or an absolute path
    parts = candidate.relative_to(data_root).parts
    if not parts or parts[0] not in SERVABLE_SUBDIRS:
        return None
    return candidate


class AddBody(BaseModel):
    url: str
    full_images: int = 6


class LoginBody(BaseModel):
    platform: str = "instagram"
    username: str
    password: str
    visible: bool = False
    wait_seconds: int = 420


class ChallengeAnswer(BaseModel):
    code: str


class SweepBody(BaseModel):
    name: str
    city: str = ""
    education: str = ""
    school: str = ""
    employer: str = ""
    hometown: str = ""
    platforms: list[str] = ["instagram", "threads", "facebook"]
    limit: int | None = None
    max_photos: int = 40
    follow_network: bool = False
    network_max: int = 40
    query_image: str | None = None
    workers: int = 4
    index_workers: int = 4


def create_app(pipeline) -> FastAPI:
    from . import db

    app = FastAPI(title="doxer", version="0.1.0")

    # A server restart orphans any sweep that was running: its thread is gone,
    # so the session would sit in "running" forever and every page load would
    # reattach to a corpse. Mark those at startup.
    try:
        with db._lock, db._conn() as c:
            cur = c.execute(
                "UPDATE sessions SET status='error', stage='done', message=?, updated_at=? "
                "WHERE status='running'",
                ("interrupted by a server restart", __import__("time").time()),
            )
            if cur.rowcount:
                print(f"[doxer] marked {cur.rowcount} orphaned sweep(s) as interrupted")
    except Exception:  # noqa: BLE001
        pass

    @app.get("/api/stats")
    def stats():
        return pipeline.stats()

    @app.get("/api/profiles")
    def profiles():
        return db.list_profiles()

    @app.post("/api/profiles")
    def add_profile(body: AddBody):
        jid = db.new_job("add")

        def run():
            try:
                pipeline.add_profile(body.url, job_id=jid, full_images=body.full_images)
            except Exception as e:  # noqa: BLE001
                db.set_job(jid, "error", str(e))

        threading.Thread(target=run, daemon=True).start()
        return {"job": jid}

    @app.get("/api/jobs/{jid}")
    def job(jid: str):
        j = db.get_job(jid)
        if not j:
            raise HTTPException(404, "no such job")
        return j

    @app.delete("/api/profiles/{pid}")
    def delete_profile(pid: str):
        p = db.get_profile(pid)
        if not p:
            raise HTTPException(404, "no such profile")
        pipeline._reset_profile(pid)
        return {"deleted": pid}

    @app.post("/api/query-image")
    async def query_image(file: UploadFile = File(...)):
        """Store an uploaded query photo so a sweep session can match against it."""
        from . import config

        data = await file.read()
        if len(data) > 16 * 1024 * 1024:
            raise HTTPException(413, "image too large (max 16 MB)")
        import cv2
        import numpy as np

        if cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR) is None:
            raise HTTPException(400, "not a decodable image")
        qdir = config.DATA / "queries"
        qdir.mkdir(parents=True, exist_ok=True)
        import time as _t

        name = f"q_{int(_t.time()*1000)}.jpg"
        target = qdir / name
        target.write_bytes(data)
        # Relative to data/ -- never hand the client an absolute host path that
        # it would post straight back as a query_image.
        return {"path": str(target.relative_to(config.DATA))}

    @app.post("/api/search")
    async def search(file: UploadFile = File(...)):
        data = await file.read()
        if len(data) > 16 * 1024 * 1024:
            raise HTTPException(413, "image too large (max 16 MB)")
        try:
            return pipeline.search_image_bytes(data)
        except ValueError as e:
            raise HTTPException(400, str(e))

    # ------------------------------------------------------------- accounts

    @app.get("/api/accounts")
    def accounts():
        return db.list_accounts()

    @app.post("/api/accounts/login")
    def accounts_login(body: LoginBody):
        """Verify credentials by actually logging in, and persist the cookies.

        With visible=true a real browser window opens so captchas or an emailed
        verification code can be completed. That takes minutes, so it runs on a
        worker thread and the UI polls /api/accounts/challenge + /api/accounts.
        """
        from .instagram import Checkpoint, LoginRequired, login, login_visible
        from .scraper import BrowserPool

        db.upsert_account(body.platform, body.username, body.password, "", "pending")

        if body.visible:
            def run_visible():
                import json as _json

                try:
                    ok, msg, cookies = login_visible(body.username, body.password,
                                                     wait_seconds=body.wait_seconds)
                except Exception as e:  # noqa: BLE001
                    db.upsert_account(body.platform, body.username, "", "", "failed",
                                      f"{type(e).__name__}: {e}")
                    return
                db.upsert_account(body.platform, body.username, "", _json.dumps(cookies),
                                  "ok" if ok else "failed", "" if ok else msg)

            threading.Thread(target=run_visible, daemon=True).start()
            return {"async": True, "message": "browser window opening — watch it, and enter the code here if asked"}

        pool = BrowserPool()
        try:
            ok, msg, cookies = pool.call(login, body.username, body.password, timeout=180)
        except (Checkpoint, LoginRequired) as e:
            db.upsert_account(body.platform, body.username, "", "", "checkpoint", str(e))
            raise HTTPException(400, str(e))
        finally:
            pool.close()
        import json as _json

        db.upsert_account(body.platform, body.username, "", _json.dumps(cookies),
                          "ok" if ok else "failed", "" if ok else msg)
        if not ok:
            raise HTTPException(400, msg)
        return {"ok": True, "message": msg, "cookies": len(cookies)}

    # ------------------------------------------------- login verification

    @app.get("/api/accounts/challenge")
    def current_challenge(platform: str = "instagram"):
        """Is an interactive login waiting for a verification code?"""
        ch = db.pending_challenge(platform)
        if not ch:
            return {"pending": False}
        return {"pending": True, **ch}

    @app.get("/api/accounts/progress")
    def login_progress(platform: str = "instagram", after: int = 0):
        """Live step log for the most recent login attempt."""
        acct = db.get_account(platform)
        user = (acct or {}).get("username", "")
        if not user:
            return {"steps": [], "done": True}
        steps = db.list_progress("login", user, after=after)
        last = steps[-1] if steps else None
        return {
            "username": user,
            "steps": steps,
            "done": bool(last and last["step"] in ("done", "error", "timeout")),
            "pct": last["pct"] if last else 0,
            "status": (acct or {}).get("status", ""),
        }

    @app.post("/api/accounts/challenge/{cid}")
    def submit_challenge(cid: str, body: ChallengeAnswer):
        ch = db.get_challenge(cid)
        if not ch:
            raise HTTPException(404, "no such challenge")
        if not db.answer_challenge(cid, body.code.strip()):
            raise HTTPException(409, "that challenge is no longer waiting")
        return {"ok": True, "id": cid}

    # -------------------------------------------------------------- sweeps

    _runners: dict[str, object] = {}
    _running: dict[str, str] = {}

    @app.get("/api/sessions")
    def sessions():
        return db.list_sessions()

    @app.post("/api/sweep")
    def start_sweep(body: SweepBody):
        from . import config as _cfg
        from .sweep import SweepRunner

        attrs = {k: v for k, v in body.model_dump().items()
                 if k in ("city", "education", "school", "employer", "hometown") and v}
        # Accept only paths inside data/ (relative or absolute), so a crafted
        # query_image cannot make the worker read arbitrary local files.
        qimg = None
        if body.query_image:
            safe = _safe_data_path(body.query_image) or _safe_data_path(
                str(Path(body.query_image).name))
            qimg = str(safe) if safe else None
        if not qimg:
            qimg = str(_cfg.DATA / "query.jpg")
        sid = db.create_session(body.name, attrs, qimg, body.platforms)

        runner = SweepRunner(sid, qimg, attrs, pipe=pipeline)
        _runners[sid] = runner

        def work():
            try:
                runner.run(
                    body.name, body.platforms,
                    limit=body.limit,
                    max_photos=body.max_photos,
                    follow_network=body.follow_network,
                    network_max=body.network_max,
                    workers=max(1, min(8, body.workers)),
                    index_workers=max(1, min(8, body.index_workers)),
                )
            except BaseException as e:  # noqa: BLE001
                # Never leave the session "running" after the worker dies:
                # the UI would poll a dead run forever and look stuck.
                import traceback

                db.update_session(sid, status="error", stage="done",
                                  message=f"{type(e).__name__}: {e}")
                traceback.print_exc()

        threading.Thread(target=work, daemon=True).start()
        return {"session": sid}

    @app.get("/api/sessions/{sid}")
    def session_detail(sid: str):
        s = db.get_session(sid)
        if not s:
            raise HTTPException(404, "no such session")
        s["profiles"] = db.list_session_profiles(sid)
        return s

    @app.get("/api/sessions/{sid}/checks")
    def session_checks(
        sid: str,
        after: int = Query(0, ge=0),
        limit: int = Query(60, ge=1, le=200),
        act_after: int = Query(0, ge=0),
    ):
        """Poll for images examined and activity lines since the given ids."""
        s = db.get_session(sid)
        if not s:
            raise HTTPException(404, "no such session")
        return {
            "stage": s["stage"],
            "status": s["status"],
            "message": s["message"],
            "n_candidates": s["n_candidates"],
            "n_profiles": s["n_profiles"],
            "n_images": s["n_images"],
            "n_faces": s["n_faces"],
            "best_score": s["best_score"],
            "checks": db.list_session_checks(sid, after_id=after, limit=limit),
            "activity": db.list_progress("sweep", sid, after=act_after, limit=80),
        }

    @app.post("/api/sessions/{sid}/cancel")
    def session_cancel(sid: str):
        r = _runners.get(sid)
        if not r:
            raise HTTPException(404, "no such running session")
        r.cancel()
        return {"cancelled": sid}

    @app.delete("/api/sessions/{sid}")
    def session_delete(sid: str):
        if not db.get_session(sid):
            raise HTTPException(404, "no such session")
        db.delete_session(sid)
        _runners.pop(sid, None)
        return {"deleted": sid}

    @app.get("/")
    def index():
        from fastapi.responses import HTMLResponse

        return HTMLResponse(STATIC.joinpath("index.html").read_text())

    @app.get("/static/{name}")
    def static_file(name: str):
        from fastapi.responses import FileResponse

        p = (STATIC / name).resolve()
        try:
            p.relative_to(STATIC.resolve())
        except ValueError:
            raise HTTPException(404, "not found")
        if not p.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(str(p))

    @app.get("/img/{path:path}")
    def img(path: str):
        """Serve scraped images and query photos.

        Only the image subdirectories are servable. data/ also holds doxer.db,
        which contains the account password and cookies in plaintext -- an
        unrestricted /img/doxer.db handed those to anyone who could reach the
        port.
        """
        from fastapi.responses import FileResponse

        from . import config

        p = _safe_data_path(path)
        if p is None or not p.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(str(p))

    return app
