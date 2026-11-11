import argparse
import json
import sys

from . import db
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
        import sys

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
                band = "STRONG" if m["score"] >= 0.6 else ("possible" if m["score"] >= 0.45 else "weak")
                name = m.get("display_name") or m["username"]
                print(f"  {i:>2}. {m['platform']:<10} @{m['username']:<28} {m['score']:.3f}  {band}")
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
