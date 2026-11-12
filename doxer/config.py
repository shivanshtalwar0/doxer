import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
IMAGES_DIR = DATA / "images"
DB_PATH = DATA / "doxer.db"

for p in (DATA, IMAGES_DIR):
    p.mkdir(parents=True, exist_ok=True)

DET_SIZE = (640, 640)          # face detector input size
THUMB_SIZE = 320               # px for saved face crops
MAX_IMAGE_BYTES = 9 * 1024 * 1024
DOWNLOAD_TIMEOUT = 25
PAGE_TIMEOUT = 45_000
BROWSER_START_TIMEOUT = 60      # seconds to wait for chromium to launch
DB_BUSY_TIMEOUT = 30            # seconds sqlite waits for a write lock
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

# ONNX runtime: set DOXER_ACCEL=coreml|cuda|cpu to force an execution provider.
# CoreML measured no faster than CPU on M-series for buffalo_l (decode+detect
# dominate), so CPU is the default.
ACCEL = os.environ.get("DOXER_ACCEL", "cpu").lower()

# Sweep defaults
SWEEP_MAX_PHOTOS = 60          # photos indexed per profile when exhaustive
# Concurrency: separate browser processes per worker (playwright's sync API is
# not thread-safe across one instance). Measured 1521ms/handle at 1 worker,
# ~470ms at 3, ~330ms at 5. 4 is a good speed/safety balance.
SWEEP_WORKERS = 4              # parallel browsers for probing
INDEX_WORKERS = 4              # profiles indexed concurrently (I/O bound)
INDEX_PREFETCH = 6             # images downloaded ahead of face detection
IG_RATE = 1.2                  # sustained requests/sec across all workers
IG_BURST = 3                   # allowed spike above that rate
SWEEP_NETWORK_PHOTOS = 12      # photos indexed per related account
SWEEP_NETWORK_MAX = 40         # related accounts followed per hit
SWEEP_NETWORK_DEPTH = 1        # how many hops outward to follow related accounts

# Institution (school/college) mode
# Only high-confidence matches are surfaced here; gallery photos are often
# group shots at low resolution, so a "possible" band is not worth showing.
INSTITUTION_MIN_SCORE = 0.55
INSTITUTION_MAX_PLACES = 80        # Maps results to collect per search
INSTITUTION_MAX_SITES = 60         # websites to crawl concurrently
INSTITUTION_MAX_IMAGES = 25        # gallery images per site
INSTITUTION_WORKERS = 4            # parallel site crawlers
INSTITUTION_PAGE_TIMEOUT = 25_000  # ms per page inside an institution site
INSTITUTION_SEARCH_TIMEOUT = 300   # seconds for the whole Maps discovery pass
INSTITUTION_VISIT_GALLERY_PAGES = 6  # extra gallery pages followed per site

# cosine-similarity bands used in the UI
SCORE_STRONG = 0.60
SCORE_POSSIBLE = 0.45
SEARCH_MIN_SCORE = 0.35
SEARCH_TOP_K = 12


def band(score: float) -> str:
    """Thresholds live here so the UI, CLI and sweep can never disagree."""
    if score >= SCORE_STRONG:
        return "strong"
    if score >= SCORE_POSSIBLE:
        return "possible"
    return "weak"
