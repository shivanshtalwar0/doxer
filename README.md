# doxer

Scrape profile photos from **Instagram / Threads / Facebook** (or any direct image URL),
detect and index every face with InsightFace, then match a query photo against all
indexed faces and rank who shows up where.

Two ways to drive it:

- **Sweep** (the main workflow, in the web UI): give a name, optional attributes
  (city, school, college, employer, hometown) and a photo. doxer generates every
  plausible handle, probes them on all selected platforms **in parallel** with a
  logged-in Instagram session, indexes every photo it finds, follows related
  accounts, and matches every face — streaming each image to the screen as it is
  checked.
- **Single profile**: add one URL and search against whatever is indexed.

```
   name + attributes ──▶ candidate handles ──▶ probe (IG auth / Threads / FB)
                                                       │
                                            scrape bio, name, photos
                                                       │
                                    InsightFace detection + SQLite
                                                       │
   your photo ──▶ detect faces ──▶ cosine similarity ──▶ ranked per profile
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

### Sweeping a name

When you know a person's name but not their handle, `sweep` generates the
username variants people actually use (plain joins, `.`/`_` separators, reversed
order, initial+surname, numeric suffixes, filler words, city tags) and probes
each one for existence before paying to scrape:

```bash
python3 -m doxer sweep "abhay rathi" --city muzaffarnagar
python3 -m doxer sweep "abhay" --first anant --last rathi --last raj --city muzaffarnagar
python3 -m doxer sweep "abhay rathi" --dry-run      # just print the candidates
```

Every handle that exists gets indexed, then the whole index is ranked against
the query photo. `sweep` reports the best score per profile **with no score
floor**, so a cold index (everything below ~0.2) is visibly different from "we
found the person".

Note the CLI sweep uses the unauthenticated prober and is therefore prone to
false positives (see the login caveat below). **Use the web UI sweep for real
work** — it logs in first and records the run as a session.

| flag | meaning |
|---|---|
| `--first` / `--last` | extra name spellings (`--last raj` covers both surnames) |
| `--city` | city appended as a suffix, e.g. `...mzngr`, `...muzaffarnagar` |
| `--platform` | `instagram` (default), `threads` (default), `facebook` |
| `--limit` | cap candidates probed |
| `--workers` | parallel browsers (default 3) |
| `--full-images` | post pages visited per hit (default 2) |
| `--query` | photo to rank against (default `data/query.jpg`) |
| `--dry-run` | print candidates and exit |

City abbreviations live in `doxer/candidates.py` (`DEFAULT_CITY_SHORT`) — add a
town there to get its shorthands.

## School / college mode

A second tab, for a different question: *is this face on any school or college
website in this city?*

```bash
# web UI -> "School / college search" tab
#   first name + optional last name
#   city
#   Schools / Colleges / Both
#   a face photo -> Find institutions -> Crawl galleries
```

Chain: Google Maps search for schools/colleges in the city -> their websites ->
gallery/photo pages -> every image -> every face -> cosine vs your photo.

Notes on the design:

- **Maps, not OpenStreetMap.** Overpass was tried first: 17 places for
  Muzaffarnagar, one with a website. Maps returns 60+ places and resolves a
  website for most.
- **The name is context, not a filter.** Matching is by face; the name is
  recorded with the hit so you can confirm it.
- **Results are the scored images, not the schools.** A school's headline
  score hides which photo actually matched, so the results grid shows every
  image that contained a face, best first, with its score and institution.
  Filter chips (all / 30%+ / 45%+ / 55%+) narrow it; click a tile to open the
  original.
- Gallery crawling is bounded (pages per site, images per site) and stays on
  the institution's own host. Expired certificates on small-school sites are
  tolerated: the crawl only reads public images.

## Web UI

`python3 -m doxer serve` → http://127.0.0.1:8321

1. **Instagram access** — enter the account once. doxer logs in with a headless
   Chromium, keeps the `sessionid` cookie in SQLite, and reuses it on later runs.
   Credentials and cookies never leave `data/doxer.db`.
2. **Who are you looking for** — name plus optional city / school / college /
   employer / hometown, platform chips, and the photo to match.
3. **Run exhaustive sweep** — the right column goes live: a stage rail
   (probing → indexing → network → matching → done), the image currently being
   examined with a scan sweep and its best score, a filmstrip of every image
   checked so far, a rolling log, and a results table sorted by score.

Attributes do double duty: the city is used to build handles
(`abhayrathimzngr`) *and* to tag profiles whose bio or name mentions it, which is
what keeps a large sweep readable.

Every run is stored as a **session** (`sessions`, `session_profiles`,
`session_checks`), so old runs can be reopened and profiles found once stay
searchable later.

JSON API: `GET /api/stats`, `GET|POST /api/profiles`, `DELETE /api/profiles/{id}`,
`POST /api/search`, `GET|POST /api/accounts`, `POST /api/accounts/login`,
`POST /api/sweep`, `GET /api/sessions`, `GET /api/sessions/{id}`,
`GET /api/sessions/{id}/checks?after=<id>` (live feed), `POST /api/sessions/{id}/cancel`.

> **Login caveat.** Logged out, Instagram answers most profile requests with a
> generic page instead of a real 404, so unauthenticated probing reports
> nonexistent handles as real. That is why the sweep wants a login. Do not use a
> primary account: automated logins can trigger a checkpoint, and handle
> churn/limits are enforced per account.

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
  doxer.db          # SQLite (WAL)
  images/<pid>/     # downloaded source photos + <n>_f<j>.jpg face crops
  queries/          # uploaded query photos for sweep sessions
.browsers/          # Playwright Chromium (headless shell)
```

Tables in `doxer.db`:

| table | holds |
|---|---|
| `profiles` / `images` / `faces` | the searchable index; `faces.embedding` is a 2048-byte float32 blob |
| `sessions` | one row per sweep run: name, attributes JSON, stage, counters, best score |
| `session_profiles` | every profile a run saw — name, bio, followers, city, education, scores, how it was discovered |
| `session_checks` | one row per image examined (with a base64 thumbnail) — this is what the live UI streams |
| `accounts` | Instagram login: username, password, cookies, status |
| `jobs` | background add-profile jobs |

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
