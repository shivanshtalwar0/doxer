"""Authenticated Instagram access via a persistent Chromium profile.

Logged-out Instagram serves a generic wall instead of a real 404, so username
probing without a session is unreliable (it reports nonexistent handles as
real). Logging in fixes both probing and scraping depth: the authenticated
profile page exposes bio, follower count, full name, and the full post grid.

Login is done once; cookies are persisted in the DB and re-injected on later
runs, so a sweep doesn't re-authenticate every time (repeated logins from a new
IP are the main checkpoint trigger).
"""

import json
import random
import re
import threading
import time
from urllib.parse import quote

from . import config, db
from .scraper import BrowserPool, ScrapeError

IG = "https://www.instagram.com"

_UA = config.UA


class LoginRequired(ScrapeError):
    pass


class Checkpoint(ScrapeError):
    pass


# ------------------------------------------------------------------ login


def _restore_cookies(browser, cookies: list[dict]) -> None:
    if not cookies:
        return
    ctx = browser.new_context(user_agent=_UA, viewport={"width": 1366, "height": 900})
    ctx.add_cookies(cookies)
    return ctx


def login(browser, username: str, password: str, progress=None) -> tuple[bool, str, list[dict]]:
    """Perform a login. Returns (ok, message, cookies).

    Instagram's login form is a React app; we fill it by its accessible labels
    which survive most redesigns, and then wait for either the feed or an
    error/checkpoint message.
    """
    if progress:
        progress("opening instagram…")
    ctx = browser.new_context(user_agent=_UA, viewport={"width": 1366, "height": 900})
    page = ctx.new_page()
    try:
        page.goto(f"{IG}/accounts/login/", wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_timeout(random.randint(1200, 2400))

        # Accept the cookie banner if present.
        for sel in ("button:has-text('Allow all')", "button:has-text('Accept')", "button:has-text('Only allow essential')"):
            try:
                el = page.query_selector(sel)
                if el and el.is_visible():
                    el.click()
                    page.wait_for_timeout(600)
                    break
            except Exception:  # noqa: BLE001
                pass

        if progress:
            progress("filling credentials…")
        user_box = page.query_selector("input[name='username'], input[aria-label*='ser name' i], input[aria-label*='hone' i]")
        pass_box = page.query_selector("input[name='password'], input[type='password']")
        if not user_box or not pass_box:
            return False, "could not find the login form (Instagram may be showing a captcha or a different layout)", []

        user_box.click()
        page.wait_for_timeout(random.randint(200, 500))
        user_box.fill(username)
        page.wait_for_timeout(random.randint(150, 400))
        pass_box.click()
        page.wait_for_timeout(random.randint(150, 400))
        pass_box.fill(password)
        page.wait_for_timeout(random.randint(300, 700))

        if progress:
            progress("submitting…")
        pass_box.press("Enter")

        # Wait for navigation off the login page, or an error message.
        deadline = time.time() + 45
        while time.time() < deadline:
            page.wait_for_timeout(1000)
            url = page.url
            if "accounts/login" not in url and "challenge" not in url:
                break
            body = (page.inner_text("body") or "").lower() if _has_body(page) else ""
            if "incorrect" in body or "didn't match" in body:
                return False, "Instagram rejected the password", []
            if "challenge" in url or "suspicious" in body or "confirm it's you" in body:
                raise Checkpoint("Instagram wants a security check — log in manually in a browser and retry")
            if "try again later" in body:
                return False, "Instagram is rate limiting this IP — wait and retry", []

        page.wait_for_timeout(random.randint(1500, 3000))

        # Dismiss the "Save your login info?" / notifications nags.
        for sel in ("button:has-text('Not now')", "button:has-text('Not Now')", "div[role='button']:has-text('Not Now')"):
            try:
                el = page.query_selector(sel)
                if el and el.is_visible():
                    el.click()
                    page.wait_for_timeout(500)
            except Exception:  # noqa: BLE001
                pass

        cookies = ctx.cookies()
        ig_cookies = [c for c in cookies if "instagram" in (c.get("domain") or "")]
        logged_in = any(c["name"] == "sessionid" for c in ig_cookies)

        if not logged_in:
            body = (page.inner_text("body") or "").lower() if _has_body(page) else ""
            if "incorrect" in body:
                return False, "Instagram rejected the password", []
            if "challenge" in page.url:
                raise Checkpoint("Instagram wants a security check — log in manually in a browser, then retry")
            return False, "login did not produce a session cookie", ig_cookies

        if progress:
            progress("logged in")
        return True, "logged in", ig_cookies
    finally:
        try:
            page.close()
        except Exception:  # noqa: BLE001
            pass


def _has_body(page) -> bool:
    try:
        return bool(page.query_selector("body"))
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------- extraction

_COUNT_SUFFIX = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}


def parse_count(s: str) -> int:
    s = (s or "").strip().lower().replace(",", "")
    m = re.match(r"([\d.]+)\s*([kmb])?", s)
    if not m:
        return 0
    try:
        v = float(m.group(1))
    except ValueError:
        return 0
    return int(v * _COUNT_SUFFIX.get(m.group(2) or "", 1))


def profile_from_page(page) -> dict:
    """Pull name, bio, follower count and image URLs off a profile page."""
    name = bio = ""
    followers = 0
    imgs: list[str] = []

    try:
        name = (page.title() or "").split("|")[0].split("•")[0].strip()[:80]
    except Exception:  # noqa: BLE001
        pass

    # Bio: og:description looks like "12K Followers, 300 Following, 45 Posts - See Instagram photos and videos from Name (@user)"
    try:
        metas = page.eval_on_selector_all(
            "meta[property='og:description']", "els => els.map(e => e.content)"
        )
        desc = metas[0] if metas else ""
        m = re.search(r"([\d.,]+\s*[KkMmBb]?)\s*Followers", desc)
        if m:
            followers = parse_count(m.group(1))
        m2 = re.search(r"—?\s*(.*)$", desc)
    except Exception:  # noqa: BLE001
        pass

    # The real bio lives in the header <section>; take its text minus the stats.
    for sel in ("header section", "main header", "section:first-of-type"):
        try:
            txt = page.inner_text(sel)
        except Exception:  # noqa: BLE001
            continue
        if not txt:
            continue
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        skip = {"follow", "following", "message", "posts", "reels", "tagged", "similar accounts"}
        keep = []
        for l in lines:
            low = l.lower()
            if low in skip or re.fullmatch(r"[\d.,]+\s*[kmbKM]?", l):
                continue
            keep.append(l)
        if keep:
            bio = " | ".join(keep[:4])[:500]
            break

    try:
        srcs = page.eval_on_selector_all(
            "img[src*='cdninstagram']", "els => els.map(e => e.currentSrc || e.src)"
        )
        imgs = [s for s in srcs if s]
    except Exception:  # noqa: BLE001
        pass

    avatar = None
    try:
        og = page.eval_on_selector_all("meta[property='og:image']", "els => els.map(e => e.content)")
        avatar = og[0] if og else None
    except Exception:  # noqa: BLE001
        pass

    return {
        "name": name,
        "bio": bio,
        "followers": followers,
        "images": list(dict.fromkeys(imgs)),
        "avatar": avatar,
    }


def _probe_authed(browser, usernames: list[str], cookies: list[dict], progress=None) -> list[dict]:
    ctx = browser.new_context(user_agent=_UA, viewport={"width": 1366, "height": 900})
    if cookies:
        ctx.add_cookies(cookies)
    page = ctx.new_page()
    out = []
    try:
        for u in usernames:
            rec = {"username": u, "exists": False, "name": "", "followers": 0}
            try:
                resp = page.goto(f"{IG}/{quote(u)}/", wait_until="domcontentloaded", timeout=45_000)
                page.wait_for_timeout(random.randint(500, 1100))
                title = (page.title() or "").strip().lower()
                status = resp.status if resp is not None else None
                body_hint = ""
                try:
                    body_hint = (page.inner_text("main") or "")[:200].lower()
                except Exception:  # noqa: BLE001
                    pass
                dead = (
                    status == 404
                    or "isn't available" in title
                    or "page not found" in title
                    or "sorry, this page" in body_hint
                )
                if not dead:
                    info = profile_from_page(page)
                    rec.update(exists=True, name=info["name"], followers=info["followers"])
            except Exception as e:  # noqa: BLE001
                rec["error"] = str(e)[:120]
            out.append(rec)
            if progress:
                progress(rec)
    finally:
        try:
            page.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            ctx.close()
        except Exception:  # noqa: BLE001
            pass
    return out


# ----------------------------------------------------------------- public API


def ensure_login(progress=None) -> list[dict]:
    """Return usable cookies for instagram, logging in if needed."""
    acct = db.get_account("instagram")
    if not acct:
        raise LoginRequired("no Instagram account configured — add one in the UI")
    try:
        cookies = json.loads(acct.get("cookies") or "[]")
    except Exception:  # noqa: BLE001
        cookies = []
    if acct.get("status") == "ok" and cookies:
        return cookies

    pool = BrowserPool()
    try:
        ok, msg, cookies = pool.call(login, acct["username"], acct.get("password") or "", timeout=180)
        if ok:
            db.upsert_account("instagram", acct["username"], "", json.dumps(cookies), "ok", "")
            return cookies
        db.upsert_account("instagram", acct["username"], "", json.dumps(cookies), "failed", msg)
        raise LoginRequired(msg)
    finally:
        pool.close()


def probe_authed(usernames: list[str], workers: int = 2, progress=None) -> list[dict]:
    """Probe usernames with an authenticated browser."""
    cookies = ensure_login(progress=progress)
    if not usernames:
        return []
    workers = max(1, min(workers, len(usernames)))
    chunks = [usernames[i::workers] for i in range(workers)]
    results: list[dict] = []
    lock = threading.Lock()

    def run(chunk):
        pool = BrowserPool()
        try:
            batch = pool.call(_probe_authed, chunk, cookies, timeout=90 * len(chunk) + 60)
            with lock:
                results.extend(batch)
        finally:
            pool.close()

    threads = [threading.Thread(target=run, args=(c,), daemon=True) for c in chunks if c]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    order = {u: i for i, u in enumerate(usernames)}
    results.sort(key=lambda r: order.get(r["username"], 0))
    return results


def scrape_authed(username: str) -> dict:
    """Fetch a profile's metadata + image URLs while logged in."""
    cookies = ensure_login()
    pool = BrowserPool()
    try:
        return pool.call(_scrape_authed, username, cookies, timeout=180)
    finally:
        pool.close()


def related_accounts(username: str, limit: int = 20) -> list[str]:
    """Usernames Instagram suggests on a profile ("similar accounts")."""
    cookies = ensure_login()
    pool = BrowserPool()
    try:
        return pool.call(_related, username, cookies, limit, timeout=120)
    finally:
        pool.close()


def _related(browser, username: str, cookies: list[dict], limit: int) -> list[str]:
    ctx = browser.new_context(user_agent=_UA, viewport={"width": 1366, "height": 900})
    if cookies:
        ctx.add_cookies(cookies)
    page = ctx.new_page()
    try:
        page.goto(f"{IG}/{quote(username)}/", wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_timeout(random.randint(1000, 1800))
        for _ in range(2):
            page.mouse.wheel(0, 1200)
            page.wait_for_timeout(random.randint(500, 900))
        hrefs = page.eval_on_selector_all(
            "a[href^='/']", "els => els.map(e => e.getAttribute('href'))"
        ) or []
        skip = {"p", "reel", "reels", "explore", "accounts", "direct", "stories", "tv", "about",
                "legal", "help", "api", "privacy", "terms", "session", "challenge"}
        out: list[str] = []
        for h in hrefs:
            if not h or not h.startswith("/") or h.startswith("//"):
                continue
            seg = h.strip("/").split("/")[0].split("?")[0]
            if not seg or seg in skip or seg.startswith("_") or "@" in str(seg):
                continue
            if re.fullmatch(r"[A-Za-z0-9._]{3,30}", seg) and seg.lower() not in {u.lower() for u in out}:
                if seg.lower() != username.lower():
                    out.append(seg)
            if len(out) >= limit:
                break
        return out
    finally:
        try:
            page.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            ctx.close()
        except Exception:  # noqa: BLE001
            pass


def related_background(username: str) -> list[str]:
    """Kept for symmetry; see _related."""
    return related_accounts(username)
    ctx = browser.new_context(user_agent=_UA, viewport={"width": 1366, "height": 900})
    if cookies:
        ctx.add_cookies(cookies)
    page = ctx.new_page()
    try:
        page.goto(f"{IG}/{quote(username)}/", wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_timeout(random.randint(1200, 2200))
        # Scroll to load more of the grid.
        for _ in range(3):
            page.mouse.wheel(0, 1400)
            page.wait_for_timeout(random.randint(700, 1400))
        info = profile_from_page(page)
        info.update({"platform": "instagram", "username": username.lower()})
        return info
    finally:
        try:
            page.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            ctx.close()
        except Exception:  # noqa: BLE001
            pass
