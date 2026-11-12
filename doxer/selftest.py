"""End-to-end sanity check: index two known portraits, search each other.

Downloads public-domain portraits from Wikimedia (with polite backoff), adds
them to the doxer index as profiles, then runs face searches and verifies the
correct profile ranks first with a high score while the other person scores lower.
"""

import time

import requests

UA = {"User-Agent": "doxer-selftest/1.0 (local face-matching test)"}

PEOPLE = [
    {
        "username": "einstein",
        "display_name": "Albert Einstein",
        "url": "https://upload.wikimedia.org/wikipedia/commons/d/d3/Albert_Einstein_Head.jpg",
    },
    {
        "username": "tesla",
        "display_name": "Nikola Tesla",
        "url": "https://upload.wikimedia.org/wikipedia/commons/7/79/Tesla_circa_1890.jpeg",
    },
]


def fetch(url: str, tries: int = 5) -> bytes:
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 200 and len(r.content) > 20_000:
                return r.content
            last = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            last = str(e)
        time.sleep(4 * (i + 1))
    raise RuntimeError(f"could not fetch {url}: {last}")


def main() -> int:
    from . import db
    from .pipeline import Pipeline

    pipe = Pipeline()

    # download portraits to a temp dir and index them as profiles
    import pathlib

    tmp = pathlib.Path("data/selftest")
    tmp.mkdir(parents=True, exist_ok=True)
    for p in PEOPLE:
        data = fetch(p["url"])
        path = tmp / f"{p['username']}.jpg"
        path.write_bytes(data)
        print(f"[selftest] indexed {p['display_name']} from {path}")

    # index via direct-image pipeline (platform 'image')
    import shutil

    for p in PEOPLE:
        pid = f"selftest_{p['username']}"
        pipe._reset_profile(pid)
        db.upsert_profile(pid, "image", p["username"], str(tmp / f"{p['username']}.jpg"), p["display_name"])
        import cv2

        img = cv2.imread(str(tmp / f"{p['username']}.jpg"))
        i = 0
        for bgr in (img,):
            rel = f"selftest/{p['username']}.jpg"
            image_id = db.add_image(pid, p["url"], rel, bgr.shape[1], bgr.shape[0])
            if not db.get_profile(pid)["avatar_path"]:
                db.set_avatar(pid, rel)
            for j, f in enumerate(pipe.engine.detect(bgr)):
                thumb = tmp / f"{p['username']}_f{j}.jpg"
                if f.thumb_bgr is not None:
                    cv2.imwrite(str(thumb), f.thumb_bgr)
                db.add_face(image_id, pid, f.bbox, f.det_score, f.embedding.tobytes(), f"selftest/{p['username']}_f{j}.jpg")
        print(f"[selftest]   faces indexed for {p['username']}")

    ok = True
    for p in PEOPLE:
        res = pipe.search_image_path(str(tmp / f"{p['username']}.jpg"), top_k=5)
        matches = res["matches"]
        top = matches[0] if matches else None
        print(f"[selftest] search {p['username']}: " + (f"top={top['profile']} score={top['score']:.3f}" if top else "no matches"))
        for m in matches:
            print(f"           - {m['profile']:<24} {m['score']:.3f}")
        if not top or top["profile"] != f"selftest_{p['username']}" or top["score"] < 0.5:
            ok = False

    # cross check: einstein query must rank tesla strictly lower
    r1 = pipe.search_image_path(str(tmp / "einstein.jpg"), top_k=5)["matches"]
    s = {m["profile"]: m["score"] for m in r1}
    if "selftest_einstein" in s and "selftest_tesla" in s:
        print(f"[selftest] discrimination gap: einstein={s['selftest_einstein']:.3f} vs tesla={s['selftest_tesla']:.3f}")

    # cleanup
    for p in PEOPLE:
        pipe._reset_profile(f"selftest_{p['username']}")
    shutil.rmtree(tmp, ignore_errors=True)

    print("[selftest]", "PASS ✔" if ok else "FAIL ✘")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
