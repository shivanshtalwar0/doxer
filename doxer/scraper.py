"""Profile scrapers for Instagram / Threads / Facebook public pages.

Strategy: render the public profile page in headless Chromium, harvest image
URLs (avatar + grid/post thumbnails), then optionally visit individual post
pages to grab full-size og:image URLs. Everything is best-effort — Meta
changes markup often and may serve login walls; direct image URLs always work.
"""

import concurrent.futures
import contextlib
import io
import os
import queue
import random
import re
import threading
import weakref

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
        self._startup_error: BaseException | None = None
        self._t = threading.Thread(target=self._run, args=(headless,), daemon=True)
        self._t.start()
        if not self._ready.wait(config.BROWSER_START_TIMEOUT):
            raise ScrapeError("browser failed to start (is chromium installed?)")
        # A failed launch used to be swallowed here, so every later call()
        # blocked for its full timeout instead of reporting the real cause.
        if self._startup_error is not None:
            raise ScrapeError(f"browser failed to start: {self._startup_error}")

    def _run(self, headless: bool) -> None:
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=headless,
                    # Local school/college sites frequently have expired or
                    # misconfigured certificates. Chrome must not refuse to
                    # render them: we only read public gallery images.
                    args=["--ignore-certificate-errors"],
                )
                self._ready.set()
                while True:
                    item = self._q.get()
                    if item is None:
                        break
                    fn, args, kw, fut = item
                    try:
                        result = fn(browser, *args, **kw)
                    except BaseException as e:  # noqa: BLE001
                        if not fut.done():
                            fut.set_exception(e)
                        continue
                    if not fut.done():      # caller may have timed out
                        fut.set_result(result)
        except BaseException as e:  # noqa: BLE001
            # Record it: __init__ re-raises, so a dead browser surfaces as
            # ScrapeError instead of every call() timing out.
            self._startup_error = e
            self._ready.set()
        finally:
            # Fail anything still queued so callers get an error, not a timeout.
            while True:
                try:
                    item = self._q.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    continue
                _, _, _, fut = item
                if not fut.done():
                    fut.set_exception(ScrapeError("browser pool closed"))

    def call(self, fn, *args, timeout: int = 90, **kwargs):
        if not self._t.is_alive() or self._startup_error is not None:
            raise ScrapeError(f"browser pool is not running: {self._startup_error}")
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._q.put((fn, args, kwargs, fut))
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise ScrapeError(f"browser call timed out after {timeout}s") from None

    def close(self) -> None:
        try:
            self._q.put(None)
        except Exception:  # noqa: BLE001
            pass


# One browser per THREAD, not one per process.
#
# Playwright's sync API binds its objects to the thread that created them, and
# the sync driver refuses to be entered from two threads at once. A single
# process-wide pool meant every caller thread -- uvicorn's request pool and each
# sweep index worker -- shared one driver, which is exactly the race that
# produced "net::ERR_ABORTED" during earlier parallel testing.
#
# A per-thread browser gives each worker a driver of its own. Browsers are
# closed by close_thread_browser(); the registry holds weak references so a
# finished thread's entry disappears instead of pinning a dead Chromium.
_local = threading.local()
_registry: "weakref.WeakValueDictionary[int, BrowserPool]" = weakref.WeakValueDictionary()
_registry_lock = threading.Lock()


def get_pool() -> BrowserPool:
    """The calling thread's browser, created on first use."""
    pool = getattr(_local, "pool", None)
    if pool is not None:
        return pool
    pool = BrowserPool()
    _local.pool = pool
    with _registry_lock:
        _registry[threading.get_ident()] = pool
    return pool


def close_thread_browser() -> None:
    """Close and forget this thread's browser. Safe to call twice."""
    pool = getattr(_local, "pool", None)
    if pool is None:
        return
    _local.pool = None
    with _registry_lock:
        _registry.pop(threading.get_ident(), None)
    pool.close()


def close_all_browsers() -> None:
    """Close every live browser this process created (used at shutdown)."""
    with _registry_lock:
        pools = list(_registry.values())
        _registry.clear()
    for pool in pools:
        pool.close()


@contextlib.contextmanager
def browser_scope() -> "Iterator[BrowserPool]":
    """Borrow this thread's browser and close it on the way out.

    Long-lived workers (sweep index threads, probe workers) should wrap their
    whole run in this so Chromium is not left running after the work ends.
    """
    try:
        yield get_pool()
    finally:
        close_thread_browser()


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
    m = re.match(r"https?://(?:www\.|m\.)?threads\.(?:com|net)/@([A-Za-z0-9._]+)", u)
    if m:
        # group(1) is the handle; the old pattern captured the hostname instead,
        # so every Threads profile was scraped as the literal user "threads.com".
        return "threads", m.group(1).lower()
    m = re.match(
        r"https?://(?:www\.|m\.|web\.)?facebook\.com/(?:profile\.php\?id=(\d+)|([A-Za-z0-9._-]+))",
        u,
    )
    if m:
        # group(1) = numeric id from profile.php, group(2) = vanity username
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
            page.wait_for_selector("img[src*='cdninstagram'], img[src*='fbcdn.net']", timeout=20_000)
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

        imgs = _collect(page, ["img[src*='cdninstagram'], img[src*='fbcdn.net']"])
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
        # Facebook serves a login wall with title "Facebook" for missing pages,
        # and Threads shows a generic "Threads • Log in" shell for the same.
        if tl in ("", "facebook", "log in", "log into facebook"):
            return out
        if tl.startswith("threads") and ("log in" in tl or tl.strip() == "threads"):
            return out
        if not title:
            return out
        out["exists"] = True
        name = title.split("|")[0].split("•")[0].strip()[:80]
        out["name"] = name
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)[:120]
    return out


def _profile_url(platform: str, username: str) -> str:
    if platform == "instagram":
        return f"https://www.instagram.com/{username}/"
    if platform == "facebook":
        return f"https://www.facebook.com/{username}"
    return f"https://www.threads.net/@{username}/"


def _probe_batch(browser, targets: list[tuple[str, str]], on_each=None, stop=None) -> list[dict]:
    """Probe [(platform, username), ...] in one page. Sequential, order preserved."""
    page = browser.new_page()
    try:
        page.set_viewport_size({"width": 1366, "height": 900})
        out = []
        for platform, username in targets:
            if stop and stop():
                break
            url = _profile_url(platform, username)
            r = _probe_one(page, url)
            rec = {"platform": platform, "username": username, "url": url, **r}
            out.append(rec)
            if on_each:
                on_each(rec)
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

    errors: list[BaseException] = []

    def worker(chunk: list[tuple[str, str]]) -> None:
        def on_each(rec: dict) -> None:
            # report per handle, not per finished chunk, so the UI updates live
            with lock:
                results.append(rec)
                done[0] += 1
                n = done[0]
            if progress:
                progress(n, len(targets), [rec])

        try:
            with browser_scope() as pool:
                pool.call(_probe_batch, chunk, timeout=120 * len(chunk) + 60, on_each=on_each,
                          stop=lambda: bool(errors))
        except BaseException as e:  # noqa: BLE001
            with lock:
                errors.append(e)

    threads = [threading.Thread(target=worker, args=(c,), daemon=True) for c in chunks if c]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
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
            page.wait_for_selector("img[src*='cdninstagram'], img[src*='fbcdn.net'], img[src*='fbsbx']", timeout=20_000)
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

        imgs = _collect(page, ["img[src*='cdninstagram'], img[src*='fbcdn.net']", "img[src*='fbsbx']"])
    finally:
        page.close()
    return {
        "platform": "threads",
        "username": username,
        "name": name,
        "avatar": avatar,
        "images": [i["src"] for i in imgs][:60],
    }


# Facebook encodes image classes in its CDN path / host:
#   static.xx.fbcdn.net            UI assets (icons, sprites)
#   scontent-*.xx.fbcdn.net/t39.*  post photos (and t1.* thumbnails)
#   t39.30808-1 / t1.*             profile pictures
_FB_STATIC_HOSTS = ("static.xx.fbcdn.net", "static.facebook.com")
_FB_PHOTO_HINTS = ("/t39.", "/t1.", "/t15.", "/t51.")


def _is_fb_photo(url: str) -> bool:
    low = url.lower()
    if any(h in low for h in _FB_STATIC_HOSTS):
        return False
    return "fbcdn.net" in low and any(h in low for h in _FB_PHOTO_HINTS)


def _fb_page_info(page) -> dict:
    """Name, follower count and 'About' text from a Facebook page."""
    name = ""
    followers = 0
    bio = ""
    try:
        name = (page.title() or "").split("|")[0].strip()[:80]
    except Exception:  # noqa: BLE001
        pass
    try:
        body = page.inner_text("body") or ""
    except Exception:  # noqa: BLE001
        body = ""
    import re as _re

    # "28 mln. volgers • 52 volgend" / "1.2M followers"
    m = _re.search(r"([\d.,]+)\s*([KkMmBb]|mln|tys)?\.?\s*(?:followers|volgers|Abonnenten|seguidores)", body)
    if m:
        val = m.group(1).replace(".", "").replace(",", ".")
        try:
            n = float(val)
        except ValueError:
            n = 0.0
        unit = (m.group(2) or "").lower()
        mult = 1
        if unit in ("k", "tys"):
            mult = 1_000
        elif unit in ("m", "mln"):
            mult = 1_000_000
        elif unit == "b":
            mult = 1_000_000_000
        followers = int(n * mult)
    for marker in ("About", "Info", "Over"):
        if marker in body:
            seg = body.split(marker, 1)[1].strip().splitlines()[:3]
            bio = " | ".join(x.strip() for x in seg if x.strip())[:300]
            break
    return {"name": name, "followers": followers, "bio": bio}


def _scrape_facebook(browser, uid: str):
    """Scrape a Facebook page/profile: name, followers, photos."""
    url = f"https://www.facebook.com/{uid}"
    ctx = browser.new_context(viewport={"width": 1366, "height": 900}, user_agent=config.UA)
    page = ctx.new_page()
    try:
        resp = page.goto(url, wait_until="domcontentloaded", timeout=config.PAGE_TIMEOUT)
        status = resp.status if resp is not None else None
        page.wait_for_timeout(3000)

        title = (page.title() or "").strip()
        # Facebook answers a missing page with HTTP 200 and a login wall, so the
        # status code proves nothing. A real page's title carries its name;
        # a walled/missing one is just "Facebook".
        title_low = title.lower()
        walled = title_low in ("", "facebook", "log in", "log into facebook")
        if status == 404 or walled or "page isn" in title_low or "not found" in title_low:
            raise ScrapeError(f"facebook page not found or not public: {uid}")

        # Load more of the timeline so the grid has real photos, not just icons.
        for _ in range(3):
            try:
                page.mouse.wheel(0, 1600)
                page.wait_for_timeout(random.randint(800, 1400))
            except Exception:  # noqa: BLE001
                break

        info = _fb_page_info(page)

        try:
            avatar = _first_meta(page, "og:image")
        except Exception:  # noqa: BLE001
            avatar = None

        try:
            srcs = page.eval_on_selector_all(
                "img[src*='fbcdn.net']", "els => els.map(e => e.currentSrc || e.src)"
            ) or []
        except Exception:  # noqa: BLE001
            srcs = []
        photos = [s for s in dict.fromkeys(srcs) if s and _is_fb_photo(s)]

        # Prefer full-size photos from the photo viewer pages.
        try:
            links = page.eval_on_selector_all(
                "a[href*='/photo/?fbid='], a[href*='/photo.php?fbid=']", "els => els.map(e => e.href)"
            ) or []
        except Exception:  # noqa: BLE001
            links = []
        full = _fb_photo_images(ctx, list(dict.fromkeys(links))[:24])
        images = full + [u for u in photos if u not in full]
    finally:
        try:
            page.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            ctx.close()
        except Exception:  # noqa: BLE001
            pass

    return {
        "platform": "facebook",
        "username": uid,
        "name": info["name"],
        "followers": info["followers"],
        "bio": info["bio"],
        "avatar": avatar,
        "images": images[:60],
    }


def _fb_photo_images(ctx, links: list[str]) -> list[str]:
    """og:image from Facebook photo pages — these are full resolution."""
    out: list[str] = []
    if not links:
        return out
    page = ctx.new_page()
    try:
        for href in links:
            try:
                page.goto(href, wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_timeout(random.randint(250, 600))
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
    return out


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
