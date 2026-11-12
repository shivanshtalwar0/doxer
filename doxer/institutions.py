"""Find schools/colleges in a city via Google Maps, with their websites.

Google Maps is the only source here that reliably lists small local
institutions *and* gives a website for most of them. OpenStreetMap/Overpass was
tried first and returned 17 places for Muzaffarnagar with a single website --
far too thin to be useful.

Maps is scraped with a real browser because it is a heavy JS app that renders
its results as you scroll. Consent walls are dismissed when present.
"""

import re
import urllib.parse

from . import config
from .scraper import ScrapeError

MAPS = "https://www.google.com/maps/search/"

# What to ask Maps for. "school" covers CBSE/ICSE/convent/play schools; "college"
# covers degree colleges, polytechnics and institutes.
SEARCH_TERMS = {
    "school": ["school", "public school", "convent school", "higher secondary school"],
    "college": ["college", "degree college", "institute", "university"],
}

# Maps result names that are not the institution we want (aggregators, ads).
_JUNK_NAME = re.compile(r"^(sponsored|results|ad)\b", re.I)


def search_url(query: str) -> str:
    return MAPS + urllib.parse.quote_plus(query)


def _dismiss_consent(page) -> None:
    for sel in (
        "button[aria-label*='Accept all' i]",
        "button[aria-label*='Accept' i]",
        "form[action*='consent'] button",
        "button:has-text('Accept all')",
        "button:has-text('I agree')",
    ):
        try:
            el = page.query_selector(sel)
            if el and el.is_visible():
                el.click()
                page.wait_for_timeout(600)
                return
        except Exception:  # noqa: BLE001
            continue


def _collect_place_links(page) -> list[tuple[str, str]]:
    """(name, place_url) pairs currently rendered in the results feed."""
    out: list[tuple[str, str]] = []
    try:
        items = page.eval_on_selector_all(
            "a[href*='/maps/place/']",
            "els => els.map(e => ({name: (e.getAttribute('aria-label') || e.textContent || '').trim(),"
            " href: e.href}))",
        )
    except Exception:  # noqa: BLE001
        return out
    for it in items or []:
        name, href = it.get("name", ""), it.get("href", "")
        if not name or not href:
            continue
        if _JUNK_NAME.match(name):
            continue
        out.append((name, href))
    return out


def _scroll_feed(page, rounds: int = 8) -> None:
    for _ in range(rounds):
        try:
            page.evaluate(
                "() => { const f = document.querySelector(\"div[role='feed']\");"
                " if (f) f.scrollTop = f.scrollHeight; }"
            )
        except Exception:  # noqa: BLE001
            return
        page.wait_for_timeout(1200)


def _place_details(page, name: str) -> dict:
    """Open a place panel and read website / address / phone."""
    details = {"name": name, "website": "", "address": "", "phone": ""}
    try:
        h1 = page.query_selector("h1")
        if h1:
            heading = (h1.text_content() or "").strip()
            if heading and heading.lower() not in ("results", "result"):
                details["name"] = heading
    except Exception:  # noqa: BLE001
        pass

    try:
        found = page.evaluate(
            """() => {
              const out = {website:'', address:'', phone:''};
              const site = [...document.querySelectorAll("a[data-item-id]")]
                .find(a => (a.getAttribute('data-item-id') || '').includes('authority'));
              if (site) out.website = site.getAttribute('href') || '';
              const addr = document.querySelector("[data-item-id^='address']");
              if (addr) out.address = (addr.textContent || '').trim();
              const phone = document.querySelector("[data-item-id^='phone']");
              if (phone) out.phone = (phone.textContent || '').trim();
              return out;
            }"""
        )
        for key in ("website", "address", "phone"):
            if found.get(key):
                details[key] = found[key]
    except Exception:  # noqa: BLE001
        pass

    # Google sometimes hides the website behind a "Website" button anchor.
    if not details["website"]:
        try:
            href = page.eval_on_selector_all(
                "a[aria-label*='website' i]", "els => els.map(e => e.href)"
            ) or []
            if href:
                details["website"] = href[0]
        except Exception:  # noqa: BLE001
            pass
    return details


def _normalise_website(url: str) -> str:
    """Strip Maps redirect wrappers and tracking, keep the real site."""
    if not url:
        return ""
    if "google.com/url" in url:
        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        url = (parsed.get("q") or parsed.get("url") or [""])[0]
    if not url.startswith(("http://", "https://")):
        return ""
    host = urllib.parse.urlparse(url).netloc.lower()
    if not host or "google." in host:
        return ""
    return url


def search(
    city: str,
    kinds: list[str] | tuple[str, ...] = ("school", "college"),
    max_places: int = config.INSTITUTION_MAX_PLACES,
    want_websites: bool = True,
    on_progress=None,
) -> list[dict]:
    """Return [{name, kind, city, website, address, phone, maps_url}].

    One browser pass over the Maps results for each kind, scrolling to load more,
    then a second pass to open each place and read its website.
    """
    city = (city or "").strip()
    if not city:
        raise ScrapeError("a city is required")
    kinds = [k for k in kinds if k in SEARCH_TERMS] or ["school", "college"]

    from .scraper import browser_scope

    places: dict[str, dict] = {}
    with browser_scope() as pool:
        pool.call(_scrape_maps, city, kinds, max_places, want_websites, on_progress,
                  places, timeout=config.INSTITUTION_SEARCH_TIMEOUT + 20 * max_places)
    return list(places.values())


def _scrape_maps(browser, city, kinds, max_places, want_websites, on_progress, sink: dict) -> None:
    ctx = browser.new_context(
        viewport={"width": 1400, "height": 900}, user_agent=config.UA, locale="en-US"
    )
    page = ctx.new_page()
    try:
        for kind in kinds:
            for term in SEARCH_TERMS[kind]:
                if len(sink) >= max_places:
                    break
                query = f"{term} in {city}"
                try:
                    page.goto(search_url(query), wait_until="domcontentloaded",
                              timeout=config.PAGE_TIMEOUT)
                except Exception as e:  # noqa: BLE001
                    if on_progress:
                        on_progress(f"Maps search failed for {query!r}: {type(e).__name__}")
                    continue
                page.wait_for_timeout(3000)
                _dismiss_consent(page)
                _scroll_feed(page, rounds=6)

                links = _collect_place_links(page)
                if on_progress:
                    on_progress(f"{query!r} -> {len(links)} place(s)")
                for name, href in links:
                    key = name.strip().lower()
                    if not key or key in sink or len(sink) >= max_places:
                        continue
                    sink[key] = {
                        "name": name.strip(), "kind": kind, "city": city,
                        "website": "", "address": "", "phone": "", "maps_url": href,
                    }

        if not want_websites:
            return
        # Second pass: open each place for its website.
        done = 0
        for entry in list(sink.values()):
            if not entry.get("maps_url"):
                continue
            try:
                page.goto(entry["maps_url"], wait_until="domcontentloaded",
                          timeout=config.PAGE_TIMEOUT)
                page.wait_for_timeout(1800)
                details = _place_details(page, entry["name"])
                entry["name"] = details["name"] or entry["name"]
                entry["website"] = _normalise_website(details["website"])
                entry["address"] = details["address"] or entry["address"]
                entry["phone"] = details["phone"] or entry["phone"]
            except Exception:  # noqa: BLE001
                pass
            done += 1
            if on_progress and done % 5 == 0:
                with_site = sum(1 for e in sink.values() if e["website"])
                on_progress(f"resolved {done}/{len(sink)} places, {with_site} with a website")
    finally:
        try:
            page.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            ctx.close()
        except Exception:  # noqa: BLE001
            pass
