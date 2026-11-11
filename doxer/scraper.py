"""Profile scrapers for Instagram / Threads / Facebook public pages.

Strategy: render the public profile page in headless Chromium, harvest image
URLs (avatar + grid/post thumbnails), then optionally visit individual post
pages to grab full-size og:image URLs. Everything is best-effort — Meta
changes markup often and may serve login walls; direct image URLs always work.
"""

import concurrent.futures
import io
import os
import queue
import re
import threading
import time

os.environ.setdefault(
    "PLAYWRIGHT_BROWSERS_PATH", str(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".browsers"))
)

import cv2  # noqa: E402
import numpy as np
import requests

from . import config


class ScrapeError(RuntimeError):
    pass


# ---------------------------------------------------------------- browser pool

class BrowserPool:
    """Runs playwright's sync API on one dedicated thread (it owns an event loop)."""

    def __init__(self, headless: bool = True):
        self._q: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self._t = threading.Thread(target=self._run, args=(headless,), daemon=True)
        self._t.start()
        if not self._ready.wait(60):
            raise ScrapeError("browser failed to start (is chromium installed?)")

    def _run(self, headless: bool) -> None:
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=headless)
                self._ready.set()
                while True:
                    item = self._q.get()
                    if item is None:
                        break
                    fn, args, kw, fut = item
                    try:
                        fut.set_result(fn(browser, *args, **kw))
                    except Exception as e:  # noqa: BLE001
                        fut.set_exception(e)
        except Exception as e:  # noqa: BLE001
            if not self._ready.is_set():
                self._ready.set()
                # surface via a sentinel future so callers get the error
            if isinstance(e, Exception):
                pass

    def call(self, fn, *args, timeout: int = 90, **kwargs):
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._q.put((fn, args, kwargs, fut))
        return fut.result(timeout=timeout)

    def close(self) -> None:
        try:
            self._q.put(None)
        except Exception:  # noqa: BLE001
            pass


_pool: BrowserPool | None = None
_pool_lock = threading.Lock()


def get_pool() -> BrowserPool:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = BrowserPool()
        return _pool


# ------------------------------------------------------------------- scraping

def parse_profile_url(url: str) -> tuple[str, str]:
    """Return (platform, username) for a social profile URL, or ('image', '')."""
    u = url.strip()
    if not re.match(r"^https?://", u):
        u = "https://" + u
    m = re.match(r"https?://(?:www\.)?(instagram\.com)/([A-Za-z0-9._]+)/?", u)
    if m:
        user = m.group(2).lower()
        if user in {"p", "reel", "reels", "stories", "explore", "accounts", "direct", "tv"}:
            raise ScrapeError(f"that looks like a post/stories link, not a profile: {url}")
        return "instagram", user
    m = re.match(r"https?://(?:www\.)?(threads\.com)/@([A-Za-z0-9._]+)/?", u)
    if m:
        return "threads", m.group(1).lower()
    m = re.match(r"https?://(?:www\.)?(facebook\.com)/(?:profile\.php\?id=(\d+)|([A-Za-z0-9._-]+))/?", u)
    if m:
        uid = m.group(1) or m.group(2)
        if uid in {"sharer.php", "photo", "photo.php", "watch", "groups", "pages", "login", "home"}:
            raise ScrapeError("that looks like a Facebook feature link, not a profile: " + url)
        return "facebook", uid
    # any other http(s) URL is treated as a direct image
    if re.match(r"^https?://\S+\.(jpe?g|png|webp|gif)(\?\S*)?$", u, re.I):
        return "image", ""
    raise ScrapeError("unrecognized URL — expected an instagram.com/<user>, threads.com/@<user>, facebook.com/<name> profile or a direct image link")


def _collect(page, selectors: list[str]) -> list[dict]:
    out = []
    seen = set()
    for sel in selectors:
        try:
            items = page.eval_on_selector_all(
                sel, "els => els.map(e => ({src: e.currentSrc || e.src, w: e.naturalWidth || 0}))"
            )
        except Exception:  # noqa: BLE001
            continue
        for it in items or []:
            src = (it.get("src") or "").strip()
            if src and src not in seen and "://" in src:
                seen.add(src)
                out.append(it)
    return out


def _first_meta(page, prop: str) -> str | None:
    """First meta[property=...] content (this playwright build has no eval_on_selector_first)."""
    try:
        out = page.eval_on_selector_all(f"meta[property='{prop}']", "els => els.map(e => e.content)")
        return out[0] if out else None
    except Exception:  # noqa: BLE001
        return None


def _scrape_instagram(browser, username: str, full_images: int):
    url = f"https://www.instagram.com/{username}/"
    page = browser.new_page()
    try:
        page.set_viewport_size({"width": 1366, "height": 900})
        page.goto(url, wait_until="domcontentloaded", timeout=config.PAGE_TIMEOUT)
        try:
            page.wait_for_selector("img[src*='cdninstagram']", timeout=20_000)
        except Exception:  # noqa: BLE001
            pass
        page.wait_for_timeout(2500)

        avatar = None
        try:
            avatar = _first_meta(page, "og:image")
        except Exception:  # noqa: BLE001
            pass
        name = ""
        try:
            name = (page.title() or "").split("|")[0].strip()
        except Exception:  # noqa: BLE001
            pass

        imgs = _collect(page, ["img[src*='cdninstagram']"])
        post_hrefs = []
        try:
            post_hrefs = page.eval_on_selector_all(
                "a[href*='/p/'], a[href*='/reel/']", "els => els.map(e => e.href)"
            ) or []
        except Exception:  # noqa: BLE001
            pass
    finally:
        page.close()

    full = []
    for href in list(dict.fromkeys(post_hrefs))[:full_images]:
        if not href.startswith("http"):
            continue
        p2 = browser.new_page()
        try:
            p2.goto(href, wait_until="domcontentloaded", timeout=config.PAGE_TIMEOUT)
            try:
                og = _first_meta(p2, "og:image")
                if og:
                    full.append({"src": og, "w": 0})
            except Exception:  # noqa: BLE001
                pass
        finally:
            p2.close()

    merged = list(dict.fromkeys([i["src"] for i in full] + [i["src"] for i in imgs]))
    return {
        "platform": "instagram",
        "username": username,
        "name": name,
        "avatar": avatar,
        "images": merged[:60],
    }


def _probe_one(page, url: str) -> dict:
    """Load a profile URL and report whether it exists (login walls count as existing)."""
    out: dict = {"exists": False, "status": None, "name": "", "error": None}
    try:
        resp = page.goto(url, wait_until="domcontentloaded", timeout=40_000)
        page.wait_for_timeout(600)
        title = (page.title() or "").strip()
        tl = title.lower()
        out["status"] = resp.status if resp is not None else None
        if resp is not None and resp.status == 404:
            return out
        if "isn't available" in tl or "page not found" in tl or "page isn't available" in tl:
            return out
        if not title:
            return out
        out["exists"] = True
        out["name"] = title.split("|")[0].strip()[:80]
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)[:120]
    return out


def _probe_batch(browser, targets: list[tuple[str, str]]) -> list[dict]:
    """Probe [(platform, username), ...] in one page. Sequential, order preserved."""
    page = browser.new_page()
    try:
        page.set_viewport_size({"width": 1366, "height": 900})
        out = []
        for platform, username in targets:
            url = f"https://www.instagram.com/{username}/" if platform == "instagram" else f"https://www.threads.net/@{username}/"
            r = _probe_one(page, url)
            out.append({"platform": platform, "username": username, "url": url, **r})
        return out
    finally:
        page.close()


def probe_usernames(targets: list[tuple[str, str]], workers: int = 3, progress=None) -> list[dict]:
    """Check whether usernames exist on their platforms.

    Returns [{platform, username, url, exists, status, name, error}]. Splits the
    work across `workers` independent browser instances for speed.
    """
    if not targets:
        return []
    workers = max(1, min(workers, len(targets)))
    chunks = [targets[i::workers] for i in range(workers)]
    results: list[dict] = []
    lock = threading.Lock()
    done = [0]

    def worker(chunk: list[tuple[str, str]]) -> None:
        pool = BrowserPool()
        try:
            batch = pool.call(_probe_batch, chunk, timeout=120 * len(chunk) + 60)
            with lock:
                results.extend(batch)
                done[0] += len(batch)
                if progress:
                    progress(done[0], len(targets), batch)
        finally:
            pool.close()

    threads = [threading.Thread(target=worker, args=(c,), daemon=True) for c in chunks if c]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    order = {u: i for i, (_, u) in enumerate(targets)}
    results.sort(key=lambda r: order.get(r["username"], 0))
    return results


def _scrape_threads(browser, username: str):
    url = f"https://www.threads.com/@{username}"
    page = browser.new_page()
    try:
        page.set_viewport_size({"width": 1366, "height": 900})
        page.goto(url, wait_until="domcontentloaded", timeout=config.PAGE_TIMEOUT)
        try:
            page.wait_for_selector("img[src*='cdninstagram'], img[src*='fbsbx']", timeout=20_000)
        except Exception:  # noqa: BLE001
            pass
        page.wait_for_timeout(2500)

        avatar = None
        try:
            avatar = _first_meta(page, "og:image")
        except Exception:  # noqa: BLE001
            pass
        name = ""
        try:
            name = (page.title() or "").split("|")[0].strip()
        except Exception:  # noqa: BLE001
            pass

        imgs = _collect(page, ["img[src*='cdninstagram']", "img[src*='fbsbx']"])
    finally:
        page.close()
    return {
        "platform": "threads",
        "username": username,
        "name": name,
        "avatar": avatar,
        "images": [i["src"] for i in imgs][:60],
    }


def _scrape_facebook(browser, uid: str):
    url = f"https://www.facebook.com/{uid}"
    page = browser.new_page()
    try:
        page.set_viewport_size({"width": 1366, "height": 900})
        page.goto(url, wait_until="domcontentloaded", timeout=config.PAGE_TIMEOUT)
        try:
            page.wait_for_selector("img[src*='scontent'], img[src*='fbsbx']", timeout=20_000)
        except Exception:  # noqa: BLE001
            pass
        page.wait_for_timeout(2500)

        avatar = None
        try:
            avatar = _first_meta(page, "og:image")
        except Exception:  # noqa: BLE001
            pass
        name = ""
        try:
            name = (page.title() or "").split("|")[0].strip()
        except Exception:  # noqa: BLE001
            pass

        imgs = _collect(page, ["img[src*='scontent']", "img[src*='fbsbx']"])
    finally:
        page.close()
    return {
        "platform": "facebook",
        "username": uid,
        "name": name,
        "avatar": avatar,
        "images": [i["src"] for i in imgs][:60],
    }


# ------------------------------------------------------------------ download

_session = requests.Session()
_session.headers.update({"User-Agent": config.UA, "Accept-Language": "en-US,en;q=0.9"})


def download_image(url: str) -> tuple[np.ndarray | None, int, int]:
    """Download an image; returns (bgr, w, h). Returns (None, 0, 0) on failure."""
    try:
        r = _session.get(url, timeout=config.DOWNLOAD_TIMEOUT, stream=True)
        if r.status_code != 200:
            return None, 0, 0
        ctype = r.headers.get("content-type", "")
        if "image" not in ctype and "octet-stream" not in ctype:
            return None, 0, 0
        buf = io.BytesIO()
        total = 0
        for chunk in r.iter_content(65536):
            total += len(chunk)
            if total > config.MAX_IMAGE_BYTES:
                return None, 0, 0
            buf.write(chunk)
        buf.seek(0)
        arr = np.frombuffer(buf.read(), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None, 0, 0
        h, w = img.shape[:2]
        if min(h, w) < 64:  # too small for a face
            return None, 0, 0
        return img, w, h
    except Exception:  # noqa: BLE001
        return None, 0, 0


def scrape_profile(url: str, full_images: int = 6) -> dict:
    """Scrape a profile URL. Returns {platform, username, name, avatar, images:[urls]}."""
    platform, username = parse_profile_url(url)
    if platform == "image":
        return {"platform": "image", "username": "direct-image", "name": "", "avatar": None, "images": [url.strip()]}

    pool = get_pool()
    if platform == "instagram":
        return pool.call(_scrape_instagram, username, full_images, timeout=180)
    if platform == "threads":
        return pool.call(_scrape_threads, username, timeout=120)
    return pool.call(_scrape_facebook, username, timeout=120)


def profile_id_for(platform: str, username: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "_", f"{platform}_{username}".lower()).strip("_")
    return safe or "profile"
