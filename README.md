# doxer

Scrape profile photos from **Instagram / Threads / Facebook** (or any direct image URL),
detect and index every face with InsightFace, then match a query photo against all
indexed faces and rank who shows up where.

```
   instagram.com/username ──▶ scrape grid + posts ──▶ download images
                                                        │
                                              InsightFace detection
                                              (buffalo_l, 512-d ArcFace)
                                                        │
                                    SQLite: embeddings per detected face
                                                        │
   your photo ──▶ detect faces ──▶ cosine similarity ──▶ ranked matches
```

## Install

```bash
cd doxer
pip install -r requirements.txt
python3 -m doxer install-browser     # one-time: headless Chromium for scraping
```

First run downloads the InsightFace `buffalo_l` models (~280 MB) into
`~/.insightface/models/`. CPU-only; ~1 s per image for detect + embed.

## CLI

```bash
python3 -m doxer add "https://www.instagram.com/nasa/" --full-images 6
#   scrape profile → download up to ~30 grid images (+ N post pages) → index faces

python3 -m doxer search path/to/photo.jpg --top 8
#   detect faces in the photo, rank all indexed profiles by best cosine score

python3 -m doxer list                 # indexed profiles + image/face counts
python3 -m doxer stats                # totals
python3 -m doxer remove <profile_id>  # drop a profile's index
python3 -m doxer serve --port 8321    # web UI + JSON API
```

Adding the same URL again refreshes that profile (old images/faces are reset).

## Web UI

`python3 -m doxer serve` → http://127.0.0.1:8321

- Add a profile by URL, watch the scrape/index job progress
- Indexed profiles with avatars and face counts
- Drag-and-drop (or paste URL) a photo → detected query faces + ranked match grid
  with score bars

JSON API: `GET /api/stats`, `GET|POST /api/profiles`, `GET /api/jobs/{id}`,
`DELETE /api/profiles/{id}`, `POST /api/search` (multipart `file`).

## Matching semantics & thresholds

Cosine similarity of L2-normalized 512-d ArcFace embeddings (i.e. dot product).
Each query face is compared against **every** indexed face; a profile's score is
the best face-to-face hit, and source URLs are reported so you can see *which*
photo produced the match.

| Score   | Band    | Meaning                                  |
|---------|---------|------------------------------------------|
| ≥ 0.60  | STRONG  | very likely the same person              |
| 0.45–0.60 | possible | could be, check the photo             |
| < 0.45  | weak    | probably different (below search floor)  |

Calibration points measured on this build:

- same photo vs itself → **1.000**
- Tesla ~1890 portrait indexed, *different* 1898 Sarony photo as query → **0.638 STRONG**
- unrelated face (Einstein) queried against Tesla index → below threshold

## Data layout

```
data/
  doxer.db          # SQLite (WAL): profiles, images, faces(+embedding), jobs
  images/<pid>/     # downloaded source photos + <n>_f<j>.jpg face crops
.browsers/          # Playwright Chromium (headless shell)
```

## Notes & caveats

- **Consent-based by design**: doxer only matches against profiles *you* index —
  it does not crawl the whole internet for a stranger's face. Indexing someone
  you don't own is on you (Instagram's ToS doesn't love scraping; be gentle).
- Headless Chromium runs with a normal desktop UA; some private/login-walled
  profiles will only yield what's publicly visible. Direct image URLs always work:
  `python3 -m doxer add "https://.../photo.jpg"`.
- Grid thumbnails are ~640 px — plenty for matching, but post-page fetches
  (`--full-images`) give higher-res crops when you need them.
- One browser thread pool per process; the server keeps it alive for the UI.
