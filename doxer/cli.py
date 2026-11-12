import argparse
import json
import sys

from . import config, db
from .pipeline import Pipeline


def _print_profile(p: dict) -> None:
    print(f"  {p['id']:<40} {p['platform']:<10} {p['n_images']:>3} imgs  {p['n_faces']:>3} faces")


def main() -> int:
    ap = argparse.ArgumentParser(prog="doxer", description="Scrape social profiles, index faces, match a query photo.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="scrape a profile URL (instagram/threads/facebook) or a direct image URL and index its faces")
    p_add.add_argument("url")
    p_add.add_argument("--full-images", type=int, default=6, help="how many post pages to visit for full-size images (default 6)")
    p_add.add_argument("--json", action="store_true")

    p_search = sub.add_parser("search", help="match a local image against the indexed faces")
    p_search.add_argument("image")
    p_search.add_argument("--top", type=int, default=10)
    p_search.add_argument("--json", action="store_true")

    p_sweep = sub.add_parser("sweep", help="probe many name-variant usernames, index every hit, then rank them against a query photo")
    p_sweep.add_argument("name", help="person's name, e.g. 'abhay rathi'")
    p_sweep.add_argument("--last", action="append", default=[], help="extra surname variants, e.g. --last raj")
    p_sweep.add_argument("--first", action="append", default=[], help="extra first-name variants, e.g. --first anant")
    p_sweep.add_argument("--city", default=None, help="city to append as a handle suffix, e.g. muzaffarnagar")
    p_sweep.add_argument("--platform", action="append", choices=["instagram", "threads", "facebook"],
                         dest="platforms", help="platforms to probe (default: instagram + threads)")
    p_sweep.add_argument("--full-images", type=int, default=2, help="post pages to visit per hit (default 2)")
    p_sweep.add_argument("--workers", type=int, default=3, help="parallel browsers for probing (default 3)")
    p_sweep.add_argument("--limit", type=int, default=None, help="cap the number of candidates probed")
    p_sweep.add_argument("--query", default=None, help="query photo to rank against (default data/query.jpg)")
    p_sweep.add_argument("--dry-run", action="store_true", help="only print the generated candidates")
    p_sweep.add_argument("--json", action="store_true")

    sub.add_parser("list", help="list indexed profiles")
    sub.add_parser("stats", help="show index statistics")

    p_rm = sub.add_parser("remove", help="delete a profile and its faces")
    p_rm.add_argument("profile_id")

    p_serve = sub.add_parser("serve", help="run the web UI + API")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8321)
    p_serve.add_argument("--reload", action="store_true")

    sub.add_parser("install-browser", help="download the headless Chromium used for scraping (one-time)")

    args = ap.parse_args()

    if args.cmd == "install-browser":
        import os
        import subprocess

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ, PLAYWRIGHT_BROWSERS_PATH=os.path.join(root, ".browsers"))
        print("downloading Chromium into", os.path.join(root, ".browsers"), "…")
        return subprocess.call([sys.executable, "-m", "playwright", "install", "chromium"], env=env)

    if args.cmd == "serve":
        import uvicorn

        from .api import create_app

        pipe = Pipeline()
        app = create_app(pipe)
        print(f"doxer UI:  http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
        return 0

    pipe = Pipeline()

    if args.cmd == "add":
        try:
            prof = pipe.add_profile(args.url, full_images=args.full_images)
        except Exception as e:  # noqa: BLE001
            print(f"error: {e}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(prof, indent=2))
        else:
            _print_profile(prof)
        return 0

    if args.cmd == "search":
        try:
            res = pipe.search_image_path(args.image, top_k=args.top)
        except Exception as e:  # noqa: BLE001
            print(f"error: {e}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            qn = len(res["query_faces"])
            print(f"found {qn} face(s) in query image")
            if not res["matches"]:
                print("no matches above threshold")
                return 0
            for i, m in enumerate(res["matches"], 1):
                band = config.band(m["score"])
                name = m.get("display_name") or m["username"]
                print(f"  {i:>2}. {m['platform']:<10} @{m['username']:<28} {m['score']:.3f}  {band}")
        return 0

    if args.cmd == "sweep":
        from .candidates import from_query

        cands = from_query(
            args.name,
            city=args.city,
            extra_firsts=args.first,
            extra_lasts=args.last,
            limit=args.limit,
        )
        if args.dry_run:
            for c in cands:
                print(c)
            print(f"-- {len(cands)} candidates", file=sys.stderr)
            return 0

        platforms = tuple(args.platforms) if args.platforms else ("instagram", "threads")
        print(f"probing {len(cands)} candidates on {', '.join(platforms)}…", file=sys.stderr)

        def progress(done: int, total: int, batch: list) -> None:
            for r in batch:
                if r.get("exists"):
                    extra = f" -> indexed {r['indexed']}" if r.get("indexed") else ""
                    print(f"  FOUND {r['platform']:<10} @{r['username']}{extra}", file=sys.stderr)
            if total:
                print(f"  …{done}/{total} probed", file=sys.stderr)

        result = pipe.sweep(
            cands,
            platforms=platforms,
            full_images=args.full_images,
            workers=args.workers,
            on_progress=progress,
        )
        best = pipe.rank_all(args.query) if any(m for m in result["best"]) else result["best"]

        if args.json:
            print(json.dumps({"result": result, "ranking": best}, indent=2, default=str))
            return 0

        print(f"\nprobed {result['probed']}  found {len(result['found'])}  indexed {len(result['indexed'])}")
        for e in result["errors"]:
            print(f"  index error {e['url']}: {e['error']}", file=sys.stderr)
        if not best:
            print("nothing indexed with a detectable face")
            return 0
        print("\nbest score per profile (no floor):")
        for m in best:
            print(f"  {m['band'].upper():<9} {m['score']:.3f}  @{m['username']:<26} {m['platform']:<10} {m['face_thumb']}")
        if not result["strong"]:
            print(f"\nnothing reached STRONG (>={config.SCORE_STRONG:.2f}) — treat all of the above as non-matches.")
        return 0

    if args.cmd == "list":
        profs = db.list_profiles()
        if not profs:
            print("no profiles indexed yet — use `doxer add <profile-url>`")
            return 0
        for p in profs:
            _print_profile(p)
        return 0

    if args.cmd == "stats":
        s = pipe.stats()
        print(f"profiles: {s['profiles']}   images: {s['images']}   faces: {s['faces']}")
        return 0

    if args.cmd == "remove":
        p = db.get_profile(args.profile_id)
        if not p:
            print(f"error: no profile '{args.profile_id}'", file=sys.stderr)
            return 1
        pipe._reset_profile(args.profile_id)
        print(f"removed {args.profile_id}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
