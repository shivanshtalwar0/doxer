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
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

# cosine-similarity bands used in the UI
SCORE_STRONG = 0.60
SCORE_POSSIBLE = 0.45
SEARCH_MIN_SCORE = 0.35
SEARCH_TOP_K = 12
