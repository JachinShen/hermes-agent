"""Authoritative, transactional session workdir switching.

Project records are only a path registry.  This module never mutates Project
ownership, folders, or active_id.  A service instance owns an explicit gateway
context and per-session locks; no request state is stored at module scope.
"""
from __future__ import annotations

import os
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class SwitchReceipt:
    success: bool
    session_id: str
    path: str | None = None
    previous_path: str | None = None
    error: str | None = None
    rollback_error: str | None = None


@dataclass
class SessionWorkdirContext:
    session_lookup: Callable[[str], dict[str, Any] | None]
    registered_paths: Callable[[], set[str]]
    persist_cwd: Callable[[dict[str, Any], str], None]
    terminal_snapshot: Callable[[str], Any]
    terminal_apply: Callable[[str, str], None]
    terminal_restore: Callable[[str, Any], None]
    git_metadata: Callable[[dict[str, Any], str], None]
    emit_session_info: Callable[[str, dict[str, Any]], None]
    session_info: Callable[[dict[str, Any]], dict[str, Any]]
    locks: dict[str, threading.RLock] = field(default_factory=dict)
    locks_guard: threading.Lock = field(default_factory=threading.Lock)


class SessionWorkdirService:
    def __init__(self, context: SessionWorkdirContext):
        self.context = context

    def _lock_for(self, session_id: str) -> threading.RLock:
        with self.context.locks_guard:
            return self.context.locks.setdefault(str(session_id), threading.RLock())

    @staticmethod
    def _canonical(path: str) -> str:
        return os.path.realpath(os.path.abspath(os.path.expanduser(path)))

    @staticmethod
    def _git_value(path: str, *args: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", "-C", path, "rev-parse", "--path-format=absolute", *args],
                capture_output=True, text=True, timeout=5, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode or not proc.stdout.strip():
            return None
        return SessionWorkdirService._canonical(proc.stdout.strip())

    def _validate(self, session: dict[str, Any], target: str) -> None:
        if not os.path.isabs(target):
            raise ValueError("workdir must be an absolute path")
        if not os.path.isdir(target):
            raise ValueError(f"workdir does not exist: {target}")
        registered = {
            self._canonical(str(path))
            for path in self.context.registered_paths()
        }
        # Exact registered folders are valid even when the session has no cwd.
        # Registration is not ownership and is never used to select a Project.
        if target in registered:
            return

        current_raw = str(session.get("cwd") or "").strip()
        if not current_raw:
            raise ValueError("unregistered worktree requires a trusted session cwd")
        current = self._canonical(current_raw)
        if not os.path.isdir(current):
            raise ValueError("trusted session cwd does not exist")

        target_root = self._git_value(target, "--show-toplevel")
        current_common = self._git_value(current, "--git-common-dir")
        target_common = self._git_value(target, "--git-common-dir")
        if target_root != target:
            raise ValueError("unregistered target must be a Git worktree root")
        if not current_common or target_common != current_common:
            raise ValueError("target is not a sibling worktree of the session cwd")

    def switch(self, session_id: str, path: str) -> SwitchReceipt:
        sid = str(session_id)
        raw_path = str(path)
        # Reject relative input before canonicalization: resolving it against
        # the gateway process cwd would silently turn an invalid request into a
        # different absolute workspace.
        if not os.path.isabs(raw_path):
            return SwitchReceipt(False, sid, error="workdir must be an absolute path")
        session = self.context.session_lookup(sid)
        if session is None:
            return SwitchReceipt(False, sid, error="no active session found")
        # Aliases (gateway table key, durable session_key, agent.session_id)
        # must all serialize on the durable key, not on the alias supplied by
        # the caller.
        real_key = str(session.get("session_key") or "")
        if not real_key:
            return SwitchReceipt(False, sid, error="session has no session_key")
        with self._lock_for(real_key):
            # Re-fetch under the per-session lock and reject replacement/races.
            current = self.context.session_lookup(sid)
            if current is None:
                return SwitchReceipt(False, sid, error="no active session found")
            if current is not session or str(current.get("session_key") or "") != real_key:
                return SwitchReceipt(False, sid, error="session changed during workspace switch")
            target = self._canonical(raw_path)
            previous = str(current.get("cwd") or "") or None
            terminal_key = real_key
            terminal_old = None
            try:
                self._validate(current, target)
                terminal_old = self.context.terminal_snapshot(terminal_key)
                self.context.terminal_apply(terminal_key, target)
                # DB is the last fallible step.  Once it succeeds, only the
                # in-memory commit and best-effort metadata remain.
                self.context.persist_cwd(current, target)
            except Exception as exc:
                rollback_error = None
                if terminal_old is not None:
                    try:
                        self.context.terminal_restore(terminal_key, terminal_old)
                    except Exception as rollback_exc:
                        rollback_error = f"terminal rollback failed: {rollback_exc}"
                return SwitchReceipt(
                    False, sid, previous_path=previous, error=str(exc),
                    rollback_error=rollback_error,
                )

            current["cwd"] = target
            current["explicit_cwd"] = True
            try:
                self.context.git_metadata(current, target)
            except Exception:
                pass
            try:
                self.context.emit_session_info(
                    str(current.get("gateway_id") or sid),
                    self.context.session_info(current),
                )
            except Exception:
                pass
            return SwitchReceipt(True, sid, path=target, previous_path=previous)
