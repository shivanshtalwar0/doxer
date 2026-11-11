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

            -- A sweep run: what we searched for, with which account, and how far it got.
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,              -- free-text name
                attrs TEXT DEFAULT '{}',          -- JSON: city, education, school, workplace, age, ...
                query_image TEXT DEFAULT '',      -- path to the photo being matched
                platforms TEXT DEFAULT '',        -- comma list
                status TEXT NOT NULL,             -- running|done|error|cancelled
                stage TEXT DEFAULT '',            -- probing|indexing|networking|matching|done
                message TEXT DEFAULT '',
                n_candidates INTEGER DEFAULT 0,
                n_profiles INTEGER DEFAULT 0,
                n_images INTEGER DEFAULT 0,
                n_faces INTEGER DEFAULT 0,
                best_score REAL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            -- Everything observed during a session, kept even for non-matches.
            CREATE TABLE IF NOT EXISTS session_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                profile_id TEXT NOT NULL,
                platform TEXT DEFAULT '',
                username TEXT DEFAULT '',
                full_name TEXT DEFAULT '',
                bio TEXT DEFAULT '',
                city TEXT DEFAULT '',
                education TEXT DEFAULT '',
                followers INTEGER DEFAULT 0,
                n_images INTEGER DEFAULT 0,
                n_faces INTEGER DEFAULT 0,
                best_score REAL DEFAULT 0,
                status TEXT DEFAULT 'indexed',    -- found|indexed|no-faces|error
                avatar_path TEXT DEFAULT '',
                source_url TEXT DEFAULT '',
                discovered_via TEXT DEFAULT '',   -- probe|network|search
                raw TEXT DEFAULT '{}',            -- whatever else the page exposed
                added_at REAL NOT NULL,
                UNIQUE(session_id, profile_id)
            );
            CREATE INDEX IF NOT EXISTS idx_sp_session ON session_profiles(session_id);

            -- One row per image examined, so the UI can replay the run image by image.
            CREATE TABLE IF NOT EXISTS session_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                profile_id TEXT DEFAULT '',
                image_path TEXT DEFAULT '',
                face_thumb TEXT DEFAULT '',
                thumb_b64 TEXT DEFAULT '',
                n_faces INTEGER DEFAULT 0,
                best_score REAL DEFAULT 0,
                matched INTEGER DEFAULT 0,
                checked_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sc_session ON session_checks(session_id);

            -- Instagram / Facebook login state.
            CREATE TABLE IF NOT EXISTS accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform TEXT NOT NULL,
                username TEXT NOT NULL,
                password TEXT DEFAULT '',
                cookies TEXT DEFAULT '',          -- JSON list of playwright cookies
                status TEXT DEFAULT 'new',        -- new|ok|failed|checkpoint
                last_error TEXT DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE(platform, username)
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


def list_images(pid: str) -> list[dict]:
    with _lock, _conn() as c:
        rows = c.execute("SELECT * FROM images WHERE profile_id=? ORDER BY id", (pid,)).fetchall()
    return [dict(r) for r in rows]


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


# ------------------------------------------------------------------ sessions

def create_session(query: str, attrs: dict, query_image: str, platforms: list[str]) -> str:
    sid = uuid.uuid4().hex[:12]
    now = time.time()
    with _lock, _conn() as c:
        c.execute(
            """INSERT INTO sessions(id,query,attrs,query_image,platforms,status,stage,created_at,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (sid, query, json.dumps(attrs), query_image, ",".join(platforms), "running", "probing", now, now),
        )
    return sid


def update_session(sid: str, **fields) -> None:
    if not fields:
        return
    allowed = {
        "status", "stage", "message", "n_candidates", "n_profiles", "n_images",
        "n_faces", "best_score",
    }
    sets, vals = [], []
    for k, v in fields.items():
        if k in allowed:
            sets.append(f"{k}=?")
            vals.append(v)
    if not sets:
        return
    sets.append("updated_at=?")
    vals.extend([time.time(), sid])
    with _lock, _conn() as c:
        c.execute(f"UPDATE sessions SET {', '.join(sets)} WHERE id=?", vals)


def get_session(sid: str) -> Optional[dict]:
    with _lock, _conn() as c:
        row = c.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
    if not row:
        return None
    d = dict(row)
    try:
        d["attrs"] = json.loads(d["attrs"]) if d["attrs"] else {}
    except Exception:
        d["attrs"] = {}
    return d


def list_sessions(limit: int = 50) -> list[dict]:
    with _lock, _conn() as c:
        rows = c.execute("SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["attrs"] = json.loads(d["attrs"]) if d["attrs"] else {}
        except Exception:
            d["attrs"] = {}
        out.append(d)
    return out


def add_session_profile(
    session_id: str,
    profile_id: str,
    *,
    platform: str = "",
    username: str = "",
    full_name: str = "",
    bio: str = "",
    city: str = "",
    education: str = "",
    followers: int = 0,
    n_images: int = 0,
    n_faces: int = 0,
    best_score: float = 0.0,
    status: str = "indexed",
    avatar_path: str = "",
    source_url: str = "",
    discovered_via: str = "probe",
    raw: dict | None = None,
) -> None:
    with _lock, _conn() as c:
        c.execute(
            """INSERT INTO session_profiles
               (session_id,profile_id,platform,username,full_name,bio,city,education,followers,
                n_images,n_faces,best_score,status,avatar_path,source_url,discovered_via,raw,added_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(session_id,profile_id) DO UPDATE SET
                 n_images=excluded.n_images, n_faces=excluded.n_faces, best_score=excluded.best_score,
                 status=excluded.status, bio=excluded.bio, full_name=excluded.full_name,
                 followers=excluded.followers""",
            (
                session_id, profile_id, platform, username, full_name, bio, city, education,
                followers, n_images, n_faces, best_score, status, avatar_path, source_url,
                discovered_via, json.dumps(raw or {}), time.time(),
            ),
        )


def list_session_profiles(sid: str, order: str = "best_score DESC") -> list[dict]:
    allowed_orders = {"best_score DESC", "best_score ASC", "added_at DESC", "n_faces DESC", "username ASC"}
    if order not in allowed_orders:
        order = "best_score DESC"
    with _lock, _conn() as c:
        rows = c.execute(f"SELECT * FROM session_profiles WHERE session_id=? ORDER BY {order}", (sid,)).fetchall()
    return [dict(r) for r in rows]


def add_session_check(
    session_id: str,
    *,
    profile_id: str = "",
    image_path: str = "",
    face_thumb: str = "",
    thumb_b64: str = "",
    n_faces: int = 0,
    best_score: float = 0.0,
    matched: bool = False,
) -> int:
    with _lock, _conn() as c:
        cur = c.execute(
            """INSERT INTO session_checks(session_id,profile_id,image_path,face_thumb,thumb_b64,
                                          n_faces,best_score,matched,checked_at)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (session_id, profile_id, image_path, face_thumb, thumb_b64, n_faces, best_score,
             1 if matched else 0, time.time()),
        )
        return int(cur.lastrowid)


def list_session_checks(sid: str, after_id: int = 0, limit: int = 200) -> list[dict]:
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT * FROM session_checks WHERE session_id=? AND id>? ORDER BY id LIMIT ?",
            (sid, after_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def clear_session_checks(sid: str) -> None:
    with _lock, _conn() as c:
        c.execute("DELETE FROM session_checks WHERE session_id=?", (sid,))


# ------------------------------------------------------------------ accounts

def upsert_account(platform: str, username: str, password: str = "", cookies: str = "",
                   status: str = "new", last_error: str = "") -> int:
    now = time.time()
    with _lock, _conn() as c:
        c.execute(
            """INSERT INTO accounts(platform,username,password,cookies,status,last_error,created_at,updated_at)
               VALUES(?,?,?,?,?,?,?,?)
               ON CONFLICT(platform,username) DO UPDATE SET
                 password=CASE WHEN excluded.password<>'' THEN excluded.password ELSE password END,
                 cookies=CASE WHEN excluded.cookies<>'' THEN excluded.cookies ELSE cookies END,
                 status=excluded.status, last_error=excluded.last_error, updated_at=excluded.updated_at""",
            (platform, username, password, cookies, status, last_error, now, now),
        )
        row = c.execute("SELECT id FROM accounts WHERE platform=? AND username=?", (platform, username)).fetchone()
    return int(row["id"]) if row else 0


def get_account(platform: str) -> Optional[dict]:
    """Most recently updated usable account for a platform."""
    with _lock, _conn() as c:
        row = c.execute(
            "SELECT * FROM accounts WHERE platform=? AND status IN ('ok','new') ORDER BY updated_at DESC LIMIT 1",
            (platform,),
        ).fetchone()
    return dict(row) if row else None


def list_accounts() -> list[dict]:
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT id,platform,username,status,last_error,updated_at,"
            "LENGTH(cookies) AS cookie_len FROM accounts ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]
