import contextlib
import json
import sqlite3
import threading
import time
import uuid
from typing import Iterator, Optional

from . import config

_lock = threading.Lock()
_initialised = False


@contextlib.contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    """Open a connection, commit on success, roll back on error, always close.

    `with sqlite3.connect(...)` is a *transaction* manager, not a closing one:
    an exception leaves the connection to the garbage collector, and each open
    WAL connection holds its own file handles. Under the sweep's worker threads
    that accumulated file descriptors.
    """
    c = sqlite3.connect(config.DB_PATH, timeout=config.DB_BUSY_TIMEOUT)
    c.row_factory = sqlite3.Row
    try:
        c.execute("PRAGMA foreign_keys=ON")
        c.execute("PRAGMA busy_timeout=%d" % int(config.DB_BUSY_TIMEOUT * 1000))
        yield c
        c.commit()
    except BaseException:
        c.rollback()
        raise
    finally:
        c.close()


def init() -> None:
    global _initialised
    with _lock, _conn() as c:
        if not _initialised:
            c.execute("PRAGMA journal_mode=WAL")   # persistent; once is enough
            c.execute("PRAGMA synchronous=NORMAL")
            _initialised = True
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
                attr_score REAL DEFAULT 0,
                rank_score REAL DEFAULT 0,
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
                stage TEXT DEFAULT 'done',
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

            -- A verification step an interactive login is blocked on
            -- (email/SMS code, captcha). The browser waits for the answer.
            CREATE TABLE IF NOT EXISTS login_challenges (
                id TEXT PRIMARY KEY,
                platform TEXT NOT NULL,
                username TEXT NOT NULL,
                kind TEXT NOT NULL,
                prompt TEXT DEFAULT '',
                status TEXT NOT NULL,
                answer TEXT DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            -- Step-by-step progress for long interactive operations (login).
            CREATE TABLE IF NOT EXISTS progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scope TEXT NOT NULL,
                ref TEXT NOT NULL,
                step TEXT DEFAULT '',
                label TEXT DEFAULT '',
                state TEXT DEFAULT 'run',
                pct INTEGER DEFAULT 0,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_progress_ref ON progress(scope, ref);
            """
        )
        # --- migrations for tables created by earlier versions
        _migrate(c)


def _migrate(c) -> None:
    """Add columns that older databases are missing (CREATE TABLE IF NOT EXISTS won't)."""
    wanted = {
        "session_checks": [("stage", "TEXT DEFAULT 'done'")],
        "sessions": [],
        "session_profiles": [("attr_score", "REAL DEFAULT 0"),
                             ("rank_score", "REAL DEFAULT 0")],
    }
    for table, cols in wanted.items():
        try:
            existing = {r[1] for r in c.execute(f"PRAGMA table_info({table})")}
        except Exception:  # noqa: BLE001
            continue
        for name, decl in cols:
            if name not in existing:
                c.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


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


def count_images(pid: str) -> int:
    """Image count for a profile -- never materialises the rows."""
    with _lock, _conn() as c:
        row = c.execute("SELECT COUNT(*) FROM images WHERE profile_id=?", (pid,)).fetchone()
    return int(row[0]) if row else 0


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
    attr_score: float = 0.0,
    rank_score: float = 0.0,
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
                n_images,n_faces,best_score,attr_score,rank_score,status,avatar_path,source_url,
                discovered_via,raw,added_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(session_id,profile_id) DO UPDATE SET
                 n_images=excluded.n_images, n_faces=excluded.n_faces, best_score=excluded.best_score,
                 attr_score=excluded.attr_score, rank_score=excluded.rank_score,
                 status=excluded.status, bio=excluded.bio, full_name=excluded.full_name,
                 followers=excluded.followers""",
            (
                session_id, profile_id, platform, username, full_name, bio, city, education,
                followers, n_images, n_faces, best_score, attr_score, rank_score, status,
                avatar_path, source_url, discovered_via, json.dumps(raw or {}), time.time(),
            ),
        )


def list_session_profiles(sid: str, order: str = "rank_score DESC") -> list[dict]:
    allowed_orders = {"rank_score DESC", "best_score DESC", "best_score ASC",
                      "added_at DESC", "n_faces DESC", "username ASC"}
    if order not in allowed_orders:
        order = "rank_score DESC"
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
    stage: str = "done",
) -> int:
    with _lock, _conn() as c:
        cur = c.execute(
            """INSERT INTO session_checks(session_id,profile_id,image_path,face_thumb,thumb_b64,
                                          n_faces,best_score,matched,stage,checked_at)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (session_id, profile_id, image_path, face_thumb, thumb_b64, n_faces, best_score,
             1 if matched else 0, stage, time.time()),
        )
        return int(cur.lastrowid)


def update_session_check(cid: int, **fields) -> None:
    allowed = {"n_faces", "best_score", "matched", "stage", "thumb_b64"}
    sets, vals = [], []
    for k, v in fields.items():
        if k in allowed:
            sets.append(f"{k}=?")
            vals.append(v)
    if not sets:
        return
    vals.append(cid)
    with _lock, _conn() as c:
        c.execute(f"UPDATE session_checks SET {', '.join(sets)} WHERE id=?", vals)


def session_best_score(sid: str) -> float:
    """Highest score this session actually observed (not a global profile max)."""
    with _lock, _conn() as c:
        row = c.execute(
            "SELECT COALESCE(MAX(best_score), 0.0) FROM session_checks WHERE session_id=?",
            (sid,),
        ).fetchone()
    return float(row[0]) if row else 0.0


def delete_session(sid: str) -> None:
    """Remove a session and its children atomically."""
    with _lock, _conn() as c:
        c.execute("DELETE FROM session_profiles WHERE session_id=?", (sid,))
        c.execute("DELETE FROM session_checks WHERE session_id=?", (sid,))
        c.execute("DELETE FROM progress WHERE scope='sweep' AND ref=?", (sid,))
        c.execute("DELETE FROM sessions WHERE id=?", (sid,))


def list_session_checks(sid: str, after_id: int = 0, limit: int = 200) -> list[dict]:
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT * FROM session_checks WHERE session_id=? AND id>? ORDER BY id LIMIT ?",
            (sid, after_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


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


# --------------------------------------------------------- login challenges

def new_challenge(platform: str, username: str, kind: str, prompt: str) -> str:
    cid = uuid.uuid4().hex[:10]
    now = time.time()
    with _lock, _conn() as c:
        c.execute(
            "INSERT INTO login_challenges(id,platform,username,kind,prompt,status,created_at,updated_at)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (cid, platform, username, kind, prompt, "waiting", now, now),
        )
    return cid


def get_challenge(cid: str) -> Optional[dict]:
    with _lock, _conn() as c:
        row = c.execute("SELECT * FROM login_challenges WHERE id=?", (cid,)).fetchone()
    return dict(row) if row else None


def pending_challenge(platform: str = "instagram") -> Optional[dict]:
    """Most recent challenge still waiting for an answer."""
    with _lock, _conn() as c:
        row = c.execute(
            "SELECT * FROM login_challenges WHERE platform=? AND status='waiting'"
            " ORDER BY created_at DESC LIMIT 1",
            (platform,),
        ).fetchone()
    return dict(row) if row else None


def answer_challenge(cid: str, answer: str) -> bool:
    with _lock, _conn() as c:
        cur = c.execute(
            "UPDATE login_challenges SET answer=?, status='submitted', updated_at=? WHERE id=? AND status='waiting'",
            (answer, time.time(), cid),
        )
        return cur.rowcount > 0


def set_challenge_status(cid: str, status: str) -> None:
    with _lock, _conn() as c:
        c.execute("UPDATE login_challenges SET status=?, updated_at=? WHERE id=?", (status, time.time(), cid))


# ------------------------------------------------------- progress reporting

def log_progress(scope: str, ref: str, label: str, step: str = "", state: str = "run", pct: int = 0) -> int:
    with _lock, _conn() as c:
        cur = c.execute(
            "INSERT INTO progress(scope,ref,step,label,state,pct,created_at) VALUES(?,?,?,?,?,?,?)",
            (scope, ref, step, label, state, max(0, min(100, int(pct))), time.time()),
        )
        return int(cur.lastrowid)


def list_progress(scope: str, ref: str, after: int = 0, limit: int = 200) -> list[dict]:
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT * FROM progress WHERE scope=? AND ref=? AND id>? ORDER BY id LIMIT ?",
            (scope, ref, after, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def clear_progress(scope: str, ref: str) -> None:
    with _lock, _conn() as c:
        c.execute("DELETE FROM progress WHERE scope=? AND ref=?", (scope, ref))
