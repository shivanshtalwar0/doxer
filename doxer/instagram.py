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
from .throttle import Throttle

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


_CODE_HINTS = (
    "enter the code", "enter code", "confirmation code", "security code",
    "we sent", "we've sent", "sent a code", "sent an email", "check your email",
    "verify your email", "verification code", "6-digit", "six-digit",
    "enter the 6", "code we sent", "sent to your email",
)
_CAPTCHA_HINTS = ("captcha", "i'm not a robot", "not a robot", "verify you're human")


def _detect_challenge(page) -> tuple[str, str] | None:
    """Inspect the page: is it asking for a verification code or a captcha?

    Returns (kind, prompt) with kind in {"email_code","sms_code","captcha"} or None.
    """
    try:
        text = (page.inner_text("body") or "").lower()
    except Exception:  # noqa: BLE001
        return None
    if not text:
        return None
    text = " ".join(text.split())

    if any(h in text for h in _CODE_HINTS):
        kind = "email_code"
        # Instagram says which channel was used; SMS is the other common one.
        if "phone" in text or "sms" in text or "text message" in text:
            kind = "sms_code"
        try:
            # The prompt sentence usually names the destination.
            import re as _re

            m = _re.search(r"([^.]*(?:sent|code|verify)[^.]*)", text)
            prompt = (m.group(1).strip() if m else "verification code required")[:200]
        except Exception:  # noqa: BLE001
            prompt = "verification code required"
        return kind, prompt

    if any(h in text for h in _CAPTCHA_HINTS):
        return "captcha", "solve the captcha in the browser window"

    return None


def _find_code_inputs(page) -> list:
    """Inputs that look like a code entry (single field or 6 separate boxes)."""
    try:
        boxes = page.query_selector_all(
            "input[name*='code' i], input[autocomplete='one-time-code'], "
            "input[inputmode='numeric'], input[type='tel'], input[maxlength='1'], input[maxlength='6']"
        )
    except Exception:  # noqa: BLE001
        return []
    return [b for b in (boxes or []) if _visible(b)]


def _fill_code(page, code: str) -> bool:
    """Type a verification code into whatever shape the form uses."""
    code = "".join(ch for ch in code if ch.isdigit() or ch.isalpha())
    if not code:
        return False
    boxes = _find_code_inputs(page)
    if not boxes:
        return False
    # Six single-character boxes: type one digit into each.
    singles = [b for b in boxes if (b.get_attribute("maxlength") or "") == "1"]
    if len(singles) >= 4:
        for b, ch in zip(singles, code):
            b.click()
            b.type(ch, delay=60)
        return True
    target = max(boxes, key=lambda b: int(b.get_attribute("maxlength") or 0) or 6)
    target.click()
    target.fill(code)
    return True


def login_visible(username: str, password: str, wait_seconds: int = 180, progress=None) -> tuple[bool, str, list[dict]]:
    """Open a real browser window, attempt the login, and let the human finish.

    Instagram often gates automated logins with a captcha or an email/SMS code.
    This shows the window, fills in what it can, then:
      - detects a verification-code prompt and writes a login_challenges row so
        the web UI can ask the user for the code,
      - polls that row and types the code into the browser as soon as it arrives,
      - keeps polling for the session cookie throughout.

    Every step is written to the `progress` table so the UI can show live,
    timestamped stages instead of a silent spinner.
    """
    from playwright.sync_api import sync_playwright

    from . import db

    def rep(label: str, step: str = "", state: str = "run", pct: int = 0) -> None:
        db.log_progress("login", username, label, step=step, state=state, pct=pct)
        if progress:
            progress(label)

    rep("starting browser", "browser", pct=5)
    cid: str | None = None
    started = time.time()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, args=["--start-maximized"])
        rep("browser window opened", "browser", "ok", 10)
        ctx = browser.new_context(viewport=None, user_agent=_UA)
        page = ctx.new_page()

        def cookies_now():
            return [c for c in ctx.cookies() if "instagram" in (c.get("domain") or "")]

        def have_session(cookies) -> bool:
            return any(c["name"] == "sessionid" and c.get("value") for c in cookies)

        try:
            rep("loading instagram.com/accounts/login", "navigate", pct=15)
            page.goto(f"{IG}/accounts/login/", wait_until="domcontentloaded", timeout=60_000)
            page.wait_for_timeout(2500)
            rep("login page loaded", "navigate", "ok", 22)

            user_box = page.query_selector(
                "input[name='email'], input[name='username'], "
                "input[aria-label*='ser name' i], input[aria-label*='mail' i]"
            )
            pass_box = page.query_selector("input[name='pass'], input[name='password'], input[type='password']")
            if not (user_box and pass_box):
                rep("could not find the login form", "form", "error", 22)
                return False, "could not find the login form (captcha or layout changed)", cookies_now()

            rep("filling username and password", "form", pct=28)
            try:
                user_box.click()
                user_box.fill(username)
                page.wait_for_timeout(400)
                pass_box.click()
                pass_box.fill(password)
                page.wait_for_timeout(400)
                rep("credentials entered", "form", "ok", 35)
                pass_box.press("Enter")
                rep("submitted — waiting for instagram to respond", "submit", pct=40)
            except Exception as e:  # noqa: BLE001
                rep(f"could not fill the form: {e}", "form", "warn", 35)

            deadline = time.time() + wait_seconds
            last_seen: tuple[str, str] | None = None
            last_beat = 0.0
            pct = 40
            while time.time() < deadline:
                if have_session(cookies_now()):
                    if cid:
                        db.set_challenge_status(cid, "expired")
                    rep("logged in — session cookie saved", "done", "ok", 100)
                    return True, "logged in", cookies_now()

                chal = db.get_challenge(cid) if cid else None

                if chal and chal["status"] == "waiting":
                    # heartbeat so the UI shows it is alive, not stuck
                    if time.time() - last_beat > 8:
                        last_beat = time.time()
                        rep("waiting for you to enter the code", "wait_code", "warn", pct)
                    page.wait_for_timeout(1000)
                    continue

                if chal and chal["status"] == "submitted":
                    rep(f"entering code {chal['answer'][:2]}•••• into the browser", "fill_code", pct=pct)
                    ok = _fill_code(page, chal["answer"])
                    if ok:
                        rep("code entered — submitting", "fill_code", "ok", pct + 4)
                        page.wait_for_timeout(700)
                        try:
                            page.keyboard.press("Enter")
                        except Exception:  # noqa: BLE001
                            pass
                    else:
                        rep("could not find the code box on the page", "fill_code", "warn", pct)
                    cid = None
                    last_beat = 0.0
                    page.wait_for_timeout(2500)
                    rep("checking that the code was accepted", "verify_code", pct=pct + 5)
                    continue

                found = _detect_challenge(page)
                if found and found != last_seen:
                    kind, prompt = found
                    last_seen = found
                    if kind in ("email_code", "sms_code"):
                        cid = db.new_challenge("instagram", username, kind, prompt)
                        pct = 55
                        rep(f"verification code required — {prompt}", "need_code", "warn", pct)
                        page.wait_for_timeout(1000)
                        continue
                    if kind == "captcha":
                        rep("captcha shown — solve it in the browser window", "captcha", "warn", pct)

                if last_seen is None and _is_incorrect_password(page):
                    rep("instagram rejected the password", "error", "error", pct)
                    return False, "Instagram rejected the password", cookies_now()

                if found is None and last_seen is None and time.time() - last_beat > 12:
                    last_beat = time.time()
                    rep("waiting for instagram…", "wait", pct=pct)

                page.wait_for_timeout(1500)

            if cid:
                db.set_challenge_status(cid, "expired")
            mins = int((time.time() - started) / 60)
            rep(f"timed out after {mins} min without a session", "timeout", "error", pct)
            return False, f"timed out after {wait_seconds}s without a session cookie", cookies_now()
        finally:
            try:
                browser.close()
            except Exception:  # noqa: BLE001
                pass


def _is_incorrect_password(page) -> bool:
    try:
        text = (page.inner_text("body") or "").lower()
    except Exception:  # noqa: BLE001
        return False
    return "incorrect" in text or "password you entered is incorrect" in text



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
        # Instagram's login form names its fields "email" and "pass" (not
        # "username"/"password"); the first is a username-or-email field.
        user_box = page.query_selector(
            "input[name='email'], input[name='username'], "
            "input[aria-label*='ser name' i], input[aria-label*='hone' i], input[aria-label*='mail' i]"
        )
        pass_box = page.query_selector("input[name='pass'], input[name='password'], input[type='password']")
        if not user_box or not pass_box:
            # last resort: positional (first visible text input, first password input)
            boxes = [b for b in (page.query_selector_all("input") or []) if _visible(b)]
            for b in boxes:
                typ = (b.get_attribute("type") or "text").lower()
                if typ == "password" and not pass_box:
                    pass_box = b
                elif typ in ("text", "email", "tel") and not user_box:
                    user_box = b
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


def _visible(el) -> bool:
    try:
        return bool(el and el.is_visible())
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


# Instagram encodes what an image is in its CDN path:
#   /v/t51.2885-19/  profile photo (avatar)
#   /v/t51.71878-15/ or t51.82787-15/  post / reel media
#   /v/t51.89012-19/, t58.9792-6/ ...  UI chrome (icons, badges, emoji)
# Only media is worth downloading; chrome produces empty "0 faces" entries.
_MEDIA_HINTS = ("t51.71878-15", "t51.82787-15", "t39.30808-6", "t51.2885-15")
_AVATAR_HINTS = ("t51.2885-19",)


def _is_media_url(url: str) -> bool:
    """True only for post/reel media.

    Unknown variants are rejected too: the observed chrome variants
    (t51.89012-19, t58.9792-6) were slipping through a permissive fallback and
    burning downloads on icons.
    """
    return any(h in url.lower() for h in _MEDIA_HINTS)


def _is_avatar_url(url: str) -> bool:
    return any(h in url.lower() for h in _AVATAR_HINTS)


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
            "img[src*='cdninstagram'], img[src*='fbcdn.net']", "els => els.map(e => e.currentSrc || e.src)"
        )
        imgs = [s for s in srcs if s and _is_media_url(s)]
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


def _page_title(page, timeout_ms: int = 4000) -> str:
    """Best available page title, lowercased.

    document.title is unreliable on profile pages: it briefly shows the real
    name in the server HTML, then hydration overwrites it with the generic
    "Instagram". og:title comes from the server HTML and stays correct, so
    prefer that and only fall back to document.title.
    """
    deadline = time.time() + timeout_ms / 1000.0
    best = ""
    while time.time() < deadline:
        try:
            metas = page.eval_on_selector_all("meta[property='og:title']", "els => els.map(e => e.content)")
        except Exception:  # noqa: BLE001
            metas = []
        cand = (metas[0] if metas else "").strip()
        if not cand:
            try:
                cand = (page.title() or "").strip()
            except Exception:  # noqa: BLE001
                cand = ""
        low = cand.lower()
        if low and not low.startswith("instagram photos and videos") and low != "instagram":
            return low
        best = low or best
        page.wait_for_timeout(150)
    return best


def _probe_authed(browser, usernames: list[str], cookies: list[dict], progress=None, throttle=None, stop=None) -> list[dict]:
    ctx = browser.new_context(user_agent=_UA, viewport={"width": 1366, "height": 900})
    if cookies:
        ctx.add_cookies(cookies)
    page = ctx.new_page()
    out = []
    try:
        for u in usernames:
            if stop and stop():
                break
            rec = {"username": u, "exists": False, "name": "", "followers": 0}
            if throttle:
                throttle.wait()
            try:
                resp = page.goto(f"{IG}/{quote(u)}/", wait_until="domcontentloaded", timeout=45_000)
                # Give the SPA a moment to swap in the real title; without this
                # we read "Instagram" and report a wrong name.
                title = _page_title(page)
                page.wait_for_timeout(random.randint(120, 280))
                status = resp.status if resp is not None else None
                body_hint = ""
                try:
                    body_hint = (page.inner_text("main") or "")[:200].lower()
                except Exception:  # noqa: BLE001
                    pass

                # Instagram pushing back: pause everyone briefly rather than
                # plough on and get the account checkpointed.
                if status == 429 or "too many requests" in body_hint or "try again later" in body_hint:
                    rec["throttled"] = True
                    rec["error"] = "rate limited"
                    if throttle:
                        throttle.penalise(25)
                    out.append(rec)
                    if progress:
                        progress(rec)
                    continue

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


def probe_authed(usernames: list[str], workers: int = 3, progress=None, on_result=None,
                 rate: float = 1.2, burst: int = 3) -> list[dict]:
    """Probe usernames with authenticated browsers, safely parallelised.

    Each worker owns its own browser process: playwright's sync API is not
    thread-safe across a shared instance (concurrent navigations abort). A
    shared token bucket caps sustained request rate across all workers, so more
    workers raise throughput without raising sustained load on Instagram.

    on_result(done, total, rec) fires as each handle resolves.
    """
    cookies = ensure_login(progress=progress)
    if not usernames:
        return []
    workers = max(1, min(workers, len(usernames)))
    chunks = [usernames[i::workers] for i in range(workers)]
    results: list[dict] = []
    lock = threading.Lock()
    done = [0]
    total = len(usernames)
    throttle = Throttle(rate=rate, burst=burst)

    errors: list[BaseException] = []

    def run(chunk):
        pool = BrowserPool()
        try:
            batch = pool.call(
                _probe_authed, chunk, cookies,
                timeout=120 * len(chunk) + 90,
                progress=_emit_wrapper(progress, on_result, lock, done, total),
                throttle=throttle,
                stop=lambda: bool(errors),
            )
            with lock:
                results.extend(batch)
        except BaseException as e:  # noqa: BLE001
            # remember it: a callback raising (e.g. cancellation) must not be
            # swallowed by the worker thread, or the caller hangs until every
            # chunk finishes on its own.
            with lock:
                errors.append(e)
        finally:
            pool.close()

    threads = [threading.Thread(target=run, args=(c,), daemon=True) for c in chunks if c]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    order = {u: i for i, u in enumerate(usernames)}
    results.sort(key=lambda r: order.get(r["username"], 0))
    return results


def _emit_wrapper(progress, on_result, lock, done, total):
    def emit(rec):
        with lock:
            done[0] += 1
            n = done[0]
        if progress:
            progress(rec)
        if on_result:
            on_result(n, total, rec)
    return emit


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


def _post_images(browser, hrefs: list[str], cookies: list[dict], limit: int = 24) -> list[str]:
    """Collect og:image (full-size photo) from post pages."""
    out: list[str] = []
    ctx = browser.new_context(user_agent=_UA, viewport={"width": 1366, "height": 900})
    if cookies:
        ctx.add_cookies(cookies)
    page = ctx.new_page()
    try:
        for href in hrefs[:limit]:
            if not href.startswith("http"):
                continue
            try:
                page.goto(href, wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_timeout(random.randint(300, 700))
                metas = page.eval_on_selector_all("meta[property='og:image']", "els => els.map(e => e.content)")
                if metas and metas[0] and metas[0] not in out:
                    out.append(metas[0])
            except Exception:  # noqa: BLE001
                continue
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


def _scrape_authed(browser, username: str, cookies: list[dict]) -> dict:
    """Load a profile page and extract name/bio/followers/photo URLs."""
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

        # The grid only carries small thumbnails, and its variants are a mix of
        # photos and UI chrome with no reliable marker. Post pages expose the
        # real photo via og:image, so prefer those when available.
        try:
            hrefs = page.eval_on_selector_all(
                "a[href*='/p/'], a[href*='/reel/']", "els => els.map(e => e.href)"
            ) or []
        except Exception:  # noqa: BLE001
            hrefs = []
        full = _post_images(browser, list(dict.fromkeys(hrefs)), cookies, limit=24)
        if full:
            info["images"] = full + [u for u in info["images"] if u not in full]
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
