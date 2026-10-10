"""
rate_limiter.py
------------------
Protects the single shared Gemini API key from crashing/getting hammered
when many shops use the app at the same time.

Two layers of protection:

1. CONCURRENCY LIMIT — only N generations are allowed to actually call the
   API at once (default 3). If more shops click "Generate" simultaneously,
   the extra ones wait in a queue instead of all hitting Gemini at once and
   triggering a wave of 429 rate-limit errors. This uses an in-process
   semaphore, which works because Streamlit Cloud/Render run your app as a
   single process handling all shop sessions in threads.

2. GLOBAL DAILY CAP — an org-wide ceiling across ALL shops combined, on top
   of each shop's own per-shop daily_limit. Per-shop limits alone can't
   stop 20 shops each slightly under their own limit from adding up to a
   huge combined bill — this catches that. Set via MAX_DAILY_GENERATIONS
   in .env (0 = unlimited).

Set MAX_CONCURRENT_GENERATIONS explicitly in .env to override the default.

Both are best-effort protections for a small/medium deployment, not a
substitute for proper Gemini billing alerts — set those up too in Google
AI Studio / Cloud Console.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager

import auth

# Default: 3 concurrent calls to the Gemini API at once.
_DEFAULT_CONCURRENCY = 3

MAX_CONCURRENT_GENERATIONS = int(os.getenv("MAX_CONCURRENT_GENERATIONS", str(_DEFAULT_CONCURRENCY)))
MAX_DAILY_GENERATIONS = int(os.getenv("MAX_DAILY_GENERATIONS", "0"))  # 0 = unlimited

_semaphore = threading.Semaphore(MAX_CONCURRENT_GENERATIONS)
_waiting_count_lock = threading.Lock()
_waiting_count = 0


def current_queue_length() -> int:
    """Roughly how many requests are currently waiting for a free slot."""
    with _waiting_count_lock:
        return _waiting_count


def global_remaining_quota() -> int:
    return auth.remaining_quota(auth.GLOBAL_KEY, MAX_DAILY_GENERATIONS)


@contextmanager
def generation_slot():
    """Blocks until a concurrency slot is free, then yields. Use as:

        with generation_slot():
            ... call the API ...

    Keeps at most MAX_CONCURRENT_GENERATIONS calls in flight across all
    shop sessions at once. current_queue_length() only counts requests
    still WAITING for a slot, not ones already running.
    """
    global _waiting_count
    with _waiting_count_lock:
        _waiting_count += 1
    try:
        _semaphore.acquire()
    finally:
        with _waiting_count_lock:
            _waiting_count -= 1
    try:
        yield
    finally:
        _semaphore.release()
