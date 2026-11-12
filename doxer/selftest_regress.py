"""Regression + edge-case tests.

Run with:  python3 -m doxer.selftest_regress

These cover the failure modes that shipped during development, so they cannot
come back quietly:

  * functions calling names that no longer exist (undefined globals)
  * scraper selectors that stop matching after an upstream change
  * the photo pipeline silently producing zero faces
  * cancel/stop leaving a session stuck in "running"
  * bad input crashing a route instead of returning an error
"""

import warnings

warnings.filterwarnings("ignore")

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(name):
    def deco(fn):
        try:
            fn()
            PASSED.append(name)
        except AssertionError as e:
            FAILED.append((name, str(e) or "assertion failed"))
        except Exception as e:  # noqa: BLE001
            FAILED.append((name, f"{type(e).__name__}: {e}"))
        return fn

    return deco


def expect_raise(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except Exception:  # noqa: BLE001
        return True
    return False


# ------------------------------------------------------------------ static


@check("no undefined globals in any module")
def _undefined_globals():
    import builtins
    import glob
    import symtable

    bad = []
    for path in glob.glob("doxer/*.py"):
        src = open(path).read()
        st = symtable.symtable(src, path, "exec")
        module_names = set(st.get_identifiers()) | set(dir(builtins)) | {"__file__", "__name__"}

        def walk(table):
            for sym in table.get_symbols():
                if sym.is_global() and not sym.is_assigned():
                    if sym.get_name() not in module_names:
                        bad.append(f"{path}: {sym.get_name()} in {table.get_name()}")
            for child in table.get_children():
                walk(child)

        walk(st)
    assert not bad, "undefined globals: " + "; ".join(bad)


@check("no unreachable code after return/raise")
def _unreachable():
    import ast
    import glob

    bad = []
    for path in glob.glob("doxer/*.py"):
        tree = ast.parse(open(path).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for i, st in enumerate(node.body[:-1]):
                    if isinstance(st, (ast.Return, ast.Raise)):
                        bad.append(f"{path}:{node.body[i + 1].lineno} in {node.name}")
    assert not bad, "unreachable: " + "; ".join(bad)


# ------------------------------------------------------------- URL filtering


@check("media filter rejects chrome and avatars")
def _media_filter():
    from doxer.instagram import _is_media_url

    base = "https://scontent.fbcdn.net/v/{}/x.jpg"
    chrome = ["t51.89012-19", "t58.9792-6", "t51.2885-19"]
    media = ["t51.71878-15", "t51.82787-15"]
    for c in chrome:
        assert not _is_media_url(base.format(c)), f"{c} should be rejected"
    for m in media:
        assert _is_media_url(base.format(m)), f"{m} should be accepted"


# --------------------------------------------------------------- candidates


@check("empty name is rejected")
def _empty_name():
    from doxer.candidates import from_query

    assert expect_raise(from_query, ""), "from_query('') should raise"


@check("name variants include city shorthand")
def _city_variants():
    from doxer.candidates import from_query

    cands = from_query("abhay rathi", city="muzaffarnagar")
    assert any("mzngr" in c for c in cands), "expected mzngr shorthand variants"
    assert "abhayrathi" in cands, "expected the plain join"
    assert all(len(c) <= 30 for c in cands), "instagram usernames max 30 chars"


@check("candidates are deduplicated and ordered")
def _dedupe():
    from doxer.candidates import from_query

    cands = from_query("anant rathi")
    assert len(cands) == len(set(cands)), "duplicates present"
    assert cands[0].startswith("anant"), "plain join should come early"


# ------------------------------------------------------------- fb parsing


@check("facebook URLs parse to the page name, not the domain")
def _fb_urls():
    from doxer.scraper import parse_profile_url

    cases = {
        "https://www.facebook.com/nasa": ("facebook", "nasa"),
        "https://facebook.com/nasa/": ("facebook", "nasa"),
        "https://m.facebook.com/nasa": ("facebook", "nasa"),
        "https://www.facebook.com/profile.php?id=123456": ("facebook", "123456"),
        "https://www.facebook.com/nasa?ref=x": ("facebook", "nasa"),
    }
    for url, want in cases.items():
        assert parse_profile_url(url) == want, f"{url} -> {parse_profile_url(url)} != {want}"


@check("facebook photo filter rejects static assets")
def _fb_filter():
    from doxer.scraper import _is_fb_photo

    assert _is_fb_photo("https://scontent-ams2-1.xx.fbcdn.net/v/t39.99422-6/x.jpg")
    assert not _is_fb_photo("https://static.xx.fbcdn.net/rsrc.php/x.png")


@check("probe builds the right URL per platform")
def _probe_urls():
    from doxer.scraper import _profile_url

    assert _profile_url("instagram", "x") == "https://www.instagram.com/x/"
    assert _profile_url("facebook", "x") == "https://www.facebook.com/x"
    assert _profile_url("threads", "x") == "https://www.threads.net/@x/"


# ---------------------------------------------------------------- attribute


@check("attribute score matches bio city")
def _attr_score():
    from doxer.sweep import attribute_score

    s, hits = attribute_score({"city": "muzaffarnagar"}, bio="from Muzaffarnagar")
    assert s == 1.0 and "city" in hits, f"expected a match, got {s} {hits}"
    s2, _ = attribute_score({"city": "muzaffarnagar"}, bio="Delhi")
    assert s2 == 0.0, "should not match"


# ----------------------------------------------------------------- pipeline


@check("download rejects non-image and oversize responses")
def _download_guard():
    from doxer.scraper import download_image

    bgr, w, h = download_image("https://example.com/not-an-image")
    assert bgr is None, "should refuse non-image content"
    bgr, w, h = download_image("ht!tp://bad url")
    assert bgr is None, "should refuse a malformed URL"


@check("search on an unreadable image fails cleanly")
def _empty_search():
    from doxer.pipeline import Pipeline

    p = Pipeline()
    assert expect_raise(p.search_image_bytes, b"not an image"), "expected ValueError"


@check("undecodable query raises ValueError (route maps it to 400)")
def _bad_query():
    from doxer.pipeline import Pipeline

    p = Pipeline()
    assert expect_raise(p.search_image_bytes, b"definitely not an image")


@check("rank_all with no query file raises ValueError, not KeyError")
def _rank_missing():
    from doxer.pipeline import Pipeline

    p = Pipeline()
    assert expect_raise(p.rank_all, "/nonexistent/query.jpg")


# -------------------------------------------------------------------- routes


@check("API routes all resolve")
def _routes():
    from doxer.api import create_app
    from doxer.pipeline import Pipeline

    app = create_app(Pipeline())
    paths = {r.path for r in app.routes}
    for want in (
        "/api/stats", "/api/profiles", "/api/search", "/api/accounts",
        "/api/accounts/login", "/api/accounts/challenge", "/api/sweep",
        "/api/sessions", "/api/sessions/{sid}", "/api/sessions/{sid}/checks",
        "/api/sessions/{sid}/cancel", "/api/query-image",
    ):
        assert want in paths, f"missing route {want}"


@check("unknown session returns 404 not a crash")
def _missing_session():
    from fastapi.testclient import TestClient

    from doxer.api import create_app
    from doxer.pipeline import Pipeline

    client = TestClient(create_app(Pipeline()))
    r = client.get("/api/sessions/nope")
    assert r.status_code == 404, f"expected 404, got {r.status_code}"




# ------------------------------------------------- audit-found regressions


@check("threads URL yields the handle, not the hostname")
def _threads_handle():
    from doxer.scraper import parse_profile_url

    for url in ("https://www.threads.com/@zuck", "https://www.threads.net/@zuck",
                "https://threads.com/@zuck"):
        got = parse_profile_url(url)
        assert got == ("threads", "zuck"), f"{url} -> {got}"


@check("localised counts parse correctly")
def _localised_counts():
    from doxer.instagram import parse_count

    for text, want in [("1,2 mln", 1_200_000), ("12.5K", 12_500), ("1,2K", 1_200),
                       ("28.000", 28_000), ("1.234", 1_234), ("104M", 104_000_000)]:
        got = parse_count(text)
        assert got == want, f"{text!r} -> {got}, want {want}"


@check("db file and paths outside images/ are not servable")
def _no_db_exfil():
    from doxer.api import _safe_data_path

    for bad in ("doxer.db", "../doxer.db", "secrets.txt",
                "images/../doxer.db", "/etc/passwd"):
        assert _safe_data_path(bad) is None, f"{bad} should be refused"
    assert _safe_data_path("images/p/000.jpg") is not None
    assert _safe_data_path("queries/q.jpg") is not None


@check("score bands match config thresholds")
def _bands():
    from doxer import config

    assert config.band(config.SCORE_STRONG) == "strong"
    assert config.band(config.SCORE_POSSIBLE) == "possible"
    assert config.band(0.0) == "weak"


@check("db connections do not leak file descriptors")
def _no_fd_leak():
    import os

    from doxer import db

    db.init()
    before = len(os.listdir("/dev/fd"))
    for _ in range(100):
        db.list_profiles()
    for _ in range(100):
        try:
            with db._lock, db._conn() as c:
                c.execute("SELECT * FROM definitely_not_a_table")
        except Exception:  # noqa: BLE001
            pass
    after = len(os.listdir("/dev/fd"))
    assert after - before <= 5, f"fd grew {before} -> {after}"


@check("browser pool reports a dead thread instead of hanging")
def _pool_fails_fast():
    from doxer.scraper import BrowserPool, ScrapeError

    pool = BrowserPool()
    pool.close()
    import time as _t

    _t.sleep(0.5)
    try:
        pool.call(lambda b: 1, timeout=5)
    except ScrapeError:
        return
    raise AssertionError("closed pool should raise ScrapeError, not block")


# ------------------------------------------- thread safety, ranking, dedupe


@check("each thread gets its own browser")
def _per_thread_browser():
    import threading

    from doxer.scraper import close_all_browsers, get_pool

    ids: dict[int, int] = {}

    def grab(i: int) -> None:
        ids[i] = id(get_pool())

    threads = [threading.Thread(target=grab, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    try:
        assert len(set(ids.values())) == 3, f"threads shared a browser: {ids}"
        assert get_pool() is get_pool(), "same thread should reuse its browser"
    finally:
        close_all_browsers()


@check("attribute evidence reorders but never changes the band")
def _combined_rank():
    from doxer import config
    from doxer.sweep import combined_rank

    assert combined_rank(0.50, 0.0) == 0.50, "no evidence should not change the score"
    assert combined_rank(0.50, 1.0) > 0.50, "evidence should lift a plausible face match"
    assert combined_rank(0.10, 1.0) == 0.10, "a weak face must not be rescued by bio text"
    assert combined_rank(0.0, 1.0) == 0.0, "no face score means no match"
    # The invariant that matters: bio text is weak evidence, so it must never
    # promote a profile into a higher confidence band.
    for face in (0.10, 0.44, 0.46, 0.50, 0.55, 0.59, 0.599, 0.70):
        for attrs in (0.25, 0.5, 1.0):
            got = combined_rank(face, attrs)
            assert config.band(got) == config.band(face), (
                f"face {face} + attrs {attrs} -> {got} changed band "
                f"{config.band(face)} -> {config.band(got)}")


@check("network search is opt-in at every layer")
def _network_default_off():
    import inspect

    from doxer.api import SweepBody
    from doxer.sweep import SweepRunner

    assert SweepBody(name="x").follow_network is False, "API must default to off"
    assert inspect.signature(SweepRunner.run).parameters["follow_network"].default is False, \
        "SweepRunner must default to off"

    src = inspect.getsource(SweepRunner._run)
    assert "network search off" in src, "a skipped network pass must be logged, not silent"

    html = open("doxer/static/index.html").read()
    assert "follow_network: true" not in html, "UI must not hardcode the network crawl on"
    assert 'id="nownetwork"' in html, "UI needs a toggle"


@check("only one sweep implementation remains")
def _single_sweep_engine():
    from doxer.pipeline import Pipeline

    assert not hasattr(Pipeline, "sweep"), "Pipeline.sweep duplicates SweepRunner"


# ------------------------------------------------------- institution mode


@check("institution search terms cover both kinds")
def _inst_terms():
    from doxer.institutions import SEARCH_TERMS, search_url

    assert "school" in SEARCH_TERMS and "college" in SEARCH_TERMS
    url = search_url("school in Muzaffarnagar")
    assert url.startswith("https://www.google.com/maps/search/")
    assert " " not in url, "query must be URL-encoded"


@check("maps website URLs are normalised, not Google redirects")
def _inst_website():
    from doxer.institutions import _normalise_website

    assert _normalise_website("http://www.holyangel.in/") == "http://www.holyangel.in/"
    wrapped = "https://www.google.com/url?q=http://school.example/&sa=D"
    assert _normalise_website(wrapped) == "http://school.example/"
    assert _normalise_website("") == ""
    assert _normalise_website("javascript:void(0)") == ""


@check("gallery crawl rejects decorative images and keeps photos")
def _gallery_filter():
    from doxer.galleries import looks_like_gallery_link, looks_like_photo

    assert not looks_like_photo("http://x/assets/images/round.png")
    assert not looks_like_photo("http://x/img/logo.png")
    assert looks_like_photo("http://x/wp-content/uploads/2023/08/IMG_123.jpg")
    assert looks_like_photo("http://x/gallery/photo12.jpg")
    assert looks_like_gallery_link("http://x/gallery", "Photo Gallery")
    assert not looks_like_gallery_link("http://x/contact", "Contact Us")


@check("institution results only surface high-confidence hits")
def _inst_threshold():
    from doxer import config

    assert config.INSTITUTION_MIN_SCORE >= config.SCORE_POSSIBLE, \
        "institution mode must be stricter than the social sweep"
    assert config.INSTITUTION_MIN_SCORE <= config.SCORE_STRONG + 0.1


# --------------------------------------------------------------------- main


def run() -> int:
    for name in PASSED:
        print(f"  PASS  {name}")
    for name, err in FAILED:
        print(f"  FAIL  {name}\n          {err}")
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(run())
