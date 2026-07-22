"""Session-scoped pending actions produced by the North runtime bridge."""
from __future__ import annotations

from threading import RLock
from typing import Any

_lock = RLock()
_pending: dict[str, dict[str, Any]] = {}


def register(session_key: str, action: dict[str, Any]) -> None:
    with _lock:
        _pending[session_key] = dict(action)


def get(session_key: str) -> dict[str, Any] | None:
    with _lock:
        action = _pending.get(session_key)
        return dict(action) if action else None


def pop(session_key: str) -> dict[str, Any] | None:
    with _lock:
        action = _pending.pop(session_key, None)
        return dict(action) if action else None


def clear(session_key: str) -> None:
    with _lock:
        _pending.pop(session_key, None)
