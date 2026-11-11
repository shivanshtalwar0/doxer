import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

STATIC = Path(__file__).parent / "static"


class AddBody(BaseModel):
    url: str
    full_images: int = 6


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

    @app.post("/api/search")
    async def search(file: UploadFile = File(...)):
        data = await file.read()
        if len(data) > 16 * 1024 * 1024:
            raise HTTPException(413, "image too large (max 16 MB)")
        try:
            return pipeline.search_image_bytes(data)
        except ValueError as e:
            raise HTTPException(400, str(e))

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
