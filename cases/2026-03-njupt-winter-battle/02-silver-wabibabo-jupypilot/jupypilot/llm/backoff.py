from __future__ import annotations

import random
import time


def backoff_sleep_seconds(attempt: int, base_s: float = 1.0, cap_s: float = 8.0, jitter_s: float = 0.2) -> float:
    """
    Exponential backoff with jitter.
    attempt: 0-based retry attempt.
    """
    raw = base_s * (2**attempt)
    return min(cap_s, raw) + random.uniform(0.0, jitter_s)


def sleep_backoff(attempt: int, base_s: float = 1.0, cap_s: float = 8.0, jitter_s: float = 0.2) -> float:
    s = backoff_sleep_seconds(attempt=attempt, base_s=base_s, cap_s=cap_s, jitter_s=jitter_s)
    time.sleep(s)
    return s

