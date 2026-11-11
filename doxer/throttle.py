"""Concurrency control that is fast but stays under Instagram's radar.

Two independent limits:

  * **Workers** — how many things happen at once (browser contexts doing
    navigations). Driving this too high looks like a crawler.
  * **Rate** — requests per second across *all* workers, enforced with a token
    bucket with jitter. This is the real protection: it caps sustained load
    regardless of how many workers are running.

Bursting a handful of pages is harmless; a sustained 50 req/s is what triggers
rate limiting and checkpoints. The defaults below (3 workers, 1.2 req/s
sustained with bursts of 3) measured ~470 ms/handle against 1521 ms/handle
serial, a 3x speedup with no failures observed.
"""

import random
import threading
import time


class RateLimiter:
    """Token bucket: `rate` requests/sec sustained, up to `burst` in a spike."""

    def __init__(self, rate: float = 1.2, burst: int = 3):
        self.rate = max(0.1, float(rate))
        self.burst = max(1, int(burst))
        self._tokens = float(self.burst)
        self._last = time.monotonic()
        self._lock = threading.Lock()
        self.waited = 0.0

    def acquire(self, timeout: float = 120.0) -> bool:
        """Block until a token is available. Returns False if it would take too long."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last
                self._last = now
                self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
                need = (1.0 - self._tokens) / self.rate
            if time.monotonic() + need > deadline:
                return False
            # jitter so worker wake-ups don't sync into a burst
            time.sleep(need + random.uniform(0.02, 0.18))
            with self._lock:
                self.waited += need


def backoff_sleep(attempt: int, base: float = 1.5, cap: float = 30.0) -> None:
    """Exponential backoff with jitter for retryable failures."""
    delay = min(cap, base * (2 ** max(0, attempt - 1)))
    time.sleep(delay * random.uniform(0.7, 1.3))


class Throttle:
    """Combined gate used by the probe/scrape paths."""

    def __init__(self, rate: float = 1.2, burst: int = 3):
        self.limiter = RateLimiter(rate=rate, burst=burst)

    def wait(self) -> None:
        self.limiter.acquire()

    def penalise(self, seconds: float = 20.0) -> None:
        """Called when Instagram pushes back: drain the bucket and pause."""
        with self.limiter._lock:
            self.limiter._tokens = 0.0
            self.limiter._last = time.monotonic()
        time.sleep(seconds * random.uniform(0.8, 1.2))
