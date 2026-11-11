"""Generate plausible social-media usernames from a person's name.

Instagram usernames are usually some join of first + last name, optionally with a
separator, a numeric suffix, a filler word ("official", "real"), or a city. This
module enumerates that space so `doxer sweep` can probe it cheaply before paying
for a full scrape.

    from doxer.candidates import generate
    generate(firsts=["abhay"], lasts=["rathi"], cities=["muzaffarnagar"])
"""

import itertools
import re

# Numeric suffixes people actually append (birth years, lucky numbers, sequences).
DEFAULT_NUMBERS = [
    "7", "07", "8", "08", "9", "09", "10", "11", "12", "13",
    "14", "15", "16", "17", "18", "19", "20", "21", "22", "23",
    "24", "25", "26", "99", "100", "101", "123", "007", "111", "786",
]

# Filler words that decorate a taken handle.
DEFAULT_WORDS = ["official", "real", "the", "in", "india", "ig", "photos", "13"]

# City tokens — often appended for people from smaller towns ("...mzngr").
DEFAULT_CITY_SHORT = {
    "muzaffarnagar": ["mzngr", "mzn", "muzaffar", "muzaffarnagr"],
}

IG_MAX_USERNAME = 30

_SPLIT = re.compile(r"[^a-z0-9]+")


def _clean(s: str) -> str:
    return _SPLIT.sub("", (s or "").lower())


def slugify(name: str) -> list[str]:
    """Split a display name into lowercase alphanumeric tokens."""
    return [t for t in _SPLIT.split((name or "").lower()) if t]


def generate(
    firsts: list[str],
    lasts: list[str] | None = None,
    *,
    cities: list[str] | None = None,
    numbers: list[str] | None = None,
    words: list[str] | None = None,
    separators: tuple[str, ...] = ("", ".", "_"),
    max_len: int = IG_MAX_USERNAME,
    limit: int | None = None,
) -> list[str]:
    """Enumerate candidate usernames, most-likely-shapes first, deduplicated.

    Ordering is deliberate: plain `firstlast` and `first.last`/`first_last` come
    first (most common), then reversed forms, then numeric and word suffixes.
    """
    lasts = [l for l in (lasts or []) if _clean(l)]
    numbers = DEFAULT_NUMBERS if numbers is None else numbers
    words = DEFAULT_WORDS if words is None else words
    cities = cities or []

    cands: list[str] = []

    def add(c: str) -> None:
        c = _clean_handle(c)
        if 3 <= len(c) <= max_len:
            cands.append(c)

    def _clean_handle(s: str) -> str:
        # keep separators for dotted/underscored handles, strip anything else
        s = (s or "").lower().strip().strip("._")
        return re.sub(r"[^a-z0-9._]", "", s)

    # expand city tokens to both long form and memorised abbreviations
    city_tokens: list[str] = []
    for c in cities:
        key = _clean(c)
        long = key
        city_tokens.append(long)
        for short in DEFAULT_CITY_SHORT.get(long, []):
            city_tokens.append(short)

    for f in firsts:
        fc = _clean(f)
        if not fc:
            continue
        if lasts:
            for l in lasts:
                lc = _clean(l)
                # 1. plain joins in every order
                for a, b in itertools.permutations((fc, lc)):
                    add(a + b)
                # 2. separator variants (both orders)
                for sep in separators:
                    for a, b in itertools.permutations((fc, lc)):
                        add(a + sep + b)
                # 3. first initial + last  (e.g. arathi)
                add(fc[0] + lc)
                # 4. numeric suffixes
                for n in numbers:
                    add(fc + lc + n)
                    add(fc + "_" + lc + n)
                    add(fc + "." + lc + n)
                # 5. filler words
                for w in words:
                    add(fc + lc + w)
                    add(fc + lc + "_" + w)
                    add("the" + fc + lc)
                # 6. city-suffixed
                for c in city_tokens:
                    add(fc + lc + c)
                    add(fc + lc + "_" + c)
        # 7. first-name-only forms (people often drop the surname entirely)
        for n in numbers:
            add(fc + n)
        for w in words:
            add(fc + "_" + w)
            add(fc + w)
        add(fc)
        for c in city_tokens:
            add(fc + c)
            add(fc + "_" + c)
            add(fc + "." + c)

    # dedupe, preserving first-seen (= most likely) order
    seen: set[str] = set()
    out: list[str] = []
    for c in cands:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out[:limit] if limit else out


def from_query(
    query: str,
    *,
    city: str | None = None,
    extra_firsts: list[str] | None = None,
    extra_lasts: list[str] | None = None,
    **kwargs,
) -> list[str]:
    """Build a candidate list from a free-text name like "abhay rathi".

    Two tokens -> first + last. One token -> treated as first name only. Extra
    alt spellings can be supplied to cover nickname/surname variants.
    """
    tokens = slugify(query)
    firsts = list(extra_firsts or [])
    lasts = list(extra_lasts or [])
    if len(tokens) >= 2:
        firsts = [tokens[0]] + firsts
        lasts = [tokens[-1]] + lasts
    elif len(tokens) == 1:
        firsts = [tokens[0]] + firsts
    else:
        raise ValueError("query had no usable letters or digits")
    return generate(firsts=firsts, lasts=lasts or None, cities=[city] if city else None, **kwargs)
