import json
import sqlite3
import threading
import time
import uuid
from typing import Optional

from . import config

_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(config.DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA foreign_keys=ON")
    return c


def init() -> None:
    with _lock, _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                id TEXT PRIMARY KEY,
                platform TEXT NOT NULL,
                username TEXT NOT NULL,
                display_name TEXT DEFAULT '',
                source_url TEXT NOT NULL,
                avatar_path TEXT DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
                url TEXT NOT NULL,
                path TEXT NOT NULL,
                w INTEGER DEFAULT 0,
                h INTEGER DEFAULT 0,
                added_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
                profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
                bbox TEXT NOT NULL,
                det_score REAL NOT NULL,
                embedding BLOB NOT NULL,
                thumb_path TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_faces_profile ON faces(profile_id);
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT DEFAULT '',
                result TEXT DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            """
        )


def new_job(kind: str) -> str:
    jid = uuid.uuid4().hex[:12]
    now = time.time()
    with _lock, _conn() as c:
        c.execute(
            "INSERT INTO jobs(id,kind,status,message,created_at,updated_at) VALUES(?,?,?,?,?,?)",
            (jid, kind, "running", "", now, now),
        )
    return jid


def set_job(jid: str, status: str, message: str = "", result: Optional[dict] = None) -> None:
    with _lock, _conn() as c:
        c.execute(
            "UPDATE jobs SET status=?, message=?, result=?, updated_at=? WHERE id=?",
            (status, message, json.dumps(result or {}), time.time(), jid),
        )


def get_job(jid: str) -> Optional[dict]:
    with _lock, _conn() as c:
        row = c.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
    if not row:
        return None
    d = dict(row)
    try:
        d["result"] = json.loads(d["result"]) if d["result"] else {}
    except Exception:
        d["result"] = {}
    return d


def upsert_profile(pid: str, platform: str, username: str, source_url: str, display_name: str = "") -> None:
    now = time.time()
    with _lock, _conn() as c:
        exists = c.execute("SELECT id FROM profiles WHERE id=?", (pid,)).fetchone()
        if exists:
            c.execute(
                "UPDATE profiles SET platform=?, username=?, display_name=?, source_url=?, updated_at=? WHERE id=?",
                (platform, username, display_name, source_url, now, pid),
            )
        else:
            c.execute(
                "INSERT INTO profiles(id,platform,username,display_name,source_url,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                (pid, platform, username, display_name, source_url, now, now),
            )


def set_avatar(pid: str, path: str) -> None:
    with _lock, _conn() as c:
        c.execute("UPDATE profiles SET avatar_path=? WHERE id=?", (path, pid))


def list_profiles() -> list[dict]:
    with _lock, _conn() as c:
        rows = c.execute(
            """SELECT p.*,
               (SELECT COUNT(*) FROM images i WHERE i.profile_id=p.id) AS n_images,
               (SELECT COUNT(*) FROM faces f WHERE f.profile_id=p.id) AS n_faces
               FROM profiles p ORDER BY p.updated_at DESC"""
        ).fetchall()
    return [dict(r) for r in rows]


def get_profile(pid: str) -> Optional[dict]:
    with _lock, _conn() as c:
        row = c.execute("SELECT * FROM profiles WHERE id=?", (pid,)).fetchone()
    return dict(row) if row else None


def delete_profile(pid: str) -> None:
    with _lock, _conn() as c:
        c.execute("DELETE FROM faces WHERE profile_id=?", (pid,))
        c.execute("DELETE FROM images WHERE profile_id=?", (pid,))
        c.execute("DELETE FROM profiles WHERE id=?", (pid,))


def add_image(profile_id: str, url: str, path: str, w: int = 0, h: int = 0) -> int:
    with _lock, _conn() as c:
        cur = c.execute(
            "INSERT INTO images(profile_id,url,path,w,h,added_at) VALUES(?,?,?,?,?,?)",
            (profile_id, url, path, w, h, time.time()),
        )
        return int(cur.lastrowid)


def add_face(image_id: int, profile_id: str, bbox: list, det_score: float, embedding: bytes, thumb_path: str = "") -> int:
    with _lock, _conn() as c:
        cur = c.execute(
            "INSERT INTO faces(image_id,profile_id,bbox,det_score,embedding,thumb_path) VALUES(?,?,?,?,?,?)",
            (image_id, profile_id, json.dumps(bbox), det_score, embedding, thumb_path),
        )
        return int(cur.lastrowid)


def all_faces() -> list[dict]:
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT f.id, f.profile_id, f.thumb_path, f.bbox, f.embedding, i.url AS image_url FROM faces f JOIN images i ON i.id=f.image_id"
        ).fetchall()
    return [dict(r) for r in rows]


def profile_face_counts() -> dict:
    with _lock, _conn() as c:
        rows = c.execute("SELECT profile_id, COUNT(*) n FROM faces GROUP BY profile_id").fetchall()
    return {r["profile_id"]: r["n"] for r in rows}
