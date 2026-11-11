import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

STATIC = Path(__file__).parent / "static"


class AddBody(BaseModel):
    url: str
    full_images: int = 6


class LoginBody(BaseModel):
    platform: str = "instagram"
    username: str
    password: str


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
    follow_network: bool = True
    network_max: int = 40
    query_image: str | None = None


def create_app(pipeline) -> FastAPI:
    from . import db

    app = FastAPI(title="doxer", version="0.1.0")

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
        (qdir / name).write_bytes(data)
        return {"path": str(qdir / name)}

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
        """Verify credentials by actually logging in, and persist the cookies."""
        from .instagram import Checkpoint, LoginRequired, login
        from .scraper import BrowserPool

        db.upsert_account(body.platform, body.username, body.password, "", "new")
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
        qimg = body.query_image or str(_cfg.DATA / "query.jpg")
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
                )
            except Exception:  # noqa: BLE001
                pass

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
    def session_checks(sid: str, after: int = 0, limit: int = 60):
        """Poll for images examined since `after` — this drives the live feed."""
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
        with db._lock, db._conn() as c:
            c.execute("DELETE FROM session_profiles WHERE session_id=?", (sid,))
            c.execute("DELETE FROM session_checks WHERE session_id=?", (sid,))
            c.execute("DELETE FROM sessions WHERE id=?", (sid,))
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
        if not str(p).startswith(str(STATIC.resolve())) or not p.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(str(p))

    @app.get("/img/{path:path}")
    def img(path: str):
        from fastapi.responses import FileResponse

        from . import config

        p = (config.DATA / path).resolve()
        if not str(p).startswith(str(config.DATA.resolve())) or not p.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(str(p))

    return app
