"""
Global demo query counter.

Tracks total queries across all concurrent Streamlit sessions using a
module-level integer protected by a threading.Lock. The counter persists
for the lifetime of the Streamlit server process. A redeployment resets it.

Usage:
    from src.demo_counter import QUERY_LIMIT, get_count, try_increment
"""

import threading

QUERY_LIMIT = 10

_lock = threading.Lock()
_count = 0


def get_count() -> int:
    return _count


def queries_remaining() -> int:
    return max(0, QUERY_LIMIT - _count)


def try_increment() -> bool:
    """Attempt to use one query slot. Returns True if allowed, False if limit reached."""
    global _count
    with _lock:
        if _count >= QUERY_LIMIT:
            return False
        _count += 1
        return True
