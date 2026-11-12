"""Crawl an institution's website for photos that may contain faces.

School and college sites are wildly inconsistent: WordPress galleries, Facebook
page embeds, raw <img> grids, sliders, PDFs of annual-day photos. This module
takes the pragmatic path -- load the home page, follow a handful of obvious
gallery-ish pages, and harvest every reasonably sized image -- because the face
detector is the real filter downstream.

It deliberately does NOT crawl the whole site: bounded page count, same-host
only, no form submissions.
"""

import re
import urllib.parse

from . import config

# Links whose text/href suggests photos.
_GALLERY_HINTS = re.compile(
    r"(galler|photo|image|album|event|activity|annual|function|campus|"
    r"life|media|news|picture|tour|infrastructure)",
    re.I,
)

# Image paths that are almost always chrome rather than people.
_JUNK_IMAGE = re.compile(
    r"(logo|icon|favicon|sprite|banner-ad|placeholder|spinner|loader|avatar-default|"
    r"flag|badge|arrow|button|bg-|background|pattern|texture|watermark|"
    r"\b(round|circle|square|line|dot|star|check|tick|quote|shape|ornament|divider)\b|"
    r"(?:^|/)(?:\d{1,2}|logo\d*)\.(?:png|svg|webp|jpe?g)$)",
    re.I,
)

# Gallery plugins and CDNs that host real photos.
_GALLERY_SRC = re.compile(
    r"(wp-content/uploads|gallery|album|photos?/|images?/|media/|"
    r"cdninstagram|fbcdn|cloudfront|imgix|\.(jpe?g|png|webp))",
    re.I,
)

_SKIP_EXT = re.compile(r"\.(css|js|json|xml|svg|ico|woff2?|ttf|eot|pdf|zip|mp4|webm)(\?|$)", re.I)


def same_host(url: str, base: str) -> bool:
    try:
        a = urllib.parse.urlparse(url).netloc.lower().removeprefix("www.")
        b = urllib.parse.urlparse(base).netloc.lower().removeprefix("www.")
        return a == b
    except Exception:  # noqa: BLE001
        return False


def looks_like_gallery_link(href: str, text: str = "") -> bool:
    if _SKIP_EXT.search(href or ""):
        return False
    return bool(_GALLERY_HINTS.search(href or "") or _GALLERY_HINTS.search(text or ""))


def looks_like_photo(url: str) -> bool:
    """Cheap pre-filter; the detector decides for real."""
    if not url or _SKIP_EXT.search(url):
        return False
    if _JUNK_IMAGE.search(urllib.parse.urlparse(url).path):
        return False
    return bool(_GALLERY_SRC.search(url))


def _collect_assets(page) -> tuple[list[str], list[tuple[str, str]]]:
    """(image urls, gallery-ish (href, text) pairs) from the loaded page."""
    try:
        images = page.eval_on_selector_all(
            "img",
            """els => els.map(e => ({
                 src: e.currentSrc || e.src || '',
                 w: e.naturalWidth || 0,
                 h: e.naturalHeight || 0
               })).filter(i => i.src)""",
        )
    except Exception:  # noqa: BLE001
        images = []

    try:
        links = page.eval_on_selector_all(
            "a[href]",
            "els => els.map(e => ({href: e.href, text: (e.textContent || '').trim().slice(0, 60)}))",
        )
    except Exception:  # noqa: BLE001
        links = []
    return images or [], [(l["href"], l.get("text", "")) for l in (links or []) if l.get("href")]


def _good_images(images: list[dict], base: str, limit: int) -> list[str]:
    out: list[str] = []
    for img in images:
        src = img.get("src", "")
        w, h = img.get("w", 0), img.get("h", 0)
        # Tiny images are decorations; unloaded lazy images report 0x0 but are
        # still worth queuing if their URL looks like real content.
        if w and h and (w < 160 or h < 160):
            continue
        if not looks_like_photo(src) and not (w >= 300 and h >= 300):
            continue
        if src not in out:
            out.append(src)
        if len(out) >= limit:
            break
    return out


def crawl_site(website: str, max_images: int = config.INSTITUTION_MAX_IMAGES,
               max_pages: int = config.INSTITUTION_VISIT_GALLERY_PAGES) -> dict:
    """Return {website, images:[urls], pages_visited, error}."""
    if not website:
        return {"website": "", "images": [], "pages_visited": 0, "error": "no website"}
    from .scraper import browser_scope

    try:
        with browser_scope() as pool:
            return pool.call(_crawl, website, max_images, max_pages,
                             timeout=config.INSTITUTION_PAGE_TIMEOUT // 1000 * (max_pages + 2) + 30)
    except Exception as e:  # noqa: BLE001
        return {"website": website, "images": [], "pages_visited": 0,
                "error": f"{type(e).__name__}: {e}"}


def _crawl(browser, website: str, max_images: int, max_pages: int) -> dict:
    ctx = browser.new_context(
        viewport={"width": 1400, "height": 900}, user_agent=config.UA, locale="en-US"
    )
    page = ctx.new_page()
    pages_visited = 0
    images: list[str] = []
    try:
        try:
            page.goto(website, wait_until="domcontentloaded",
                      timeout=config.INSTITUTION_PAGE_TIMEOUT)
            page.wait_for_timeout(2000)
        except Exception as e:  # noqa: BLE001
            return {"website": website, "images": [], "pages_visited": 0,
                    "error": f"{type(e).__name__}: {e}"}

        # Lazy-loaded galleries need a scroll to populate.
        for _ in range(3):
            try:
                page.mouse.wheel(0, 2200)
                page.wait_for_timeout(900)
            except Exception:  # noqa: BLE001
                break

        base = page.url
        imgs, links = _collect_assets(page)
        pages_visited += 1
        images.extend(_good_images(imgs, base, max_images))

        # Follow gallery-ish pages, same host only.
        targets: list[str] = []
        for href, text in links:
            if len(targets) >= max_pages:
                break
            if not same_host(href, base):
                continue
            if not looks_like_gallery_link(href, text):
                continue
            if href not in targets and href != base:
                targets.append(href)

        for href in targets:
            if len(images) >= max_images:
                break
            try:
                page.goto(href, wait_until="domcontentloaded",
                          timeout=config.INSTITUTION_PAGE_TIMEOUT)
                page.wait_for_timeout(1500)
                for _ in range(2):
                    page.mouse.wheel(0, 2000)
                    page.wait_for_timeout(700)
                imgs, _ = _collect_assets(page)
                pages_visited += 1
                for src in _good_images(imgs, href, max_images - len(images)):
                    if src not in images:
                        images.append(src)
            except Exception:  # noqa: BLE001
                continue
        return {"website": website, "images": images[:max_images],
                "pages_visited": pages_visited, "error": ""}
    finally:
        try:
            page.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            ctx.close()
        except Exception:  # noqa: BLE001
            pass
