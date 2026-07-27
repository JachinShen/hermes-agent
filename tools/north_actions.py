"""Session-scoped pending actions produced by the North runtime bridge.

Also provides pure-intent project_list and project_switch for the TUI-only
North workspace switch control-plane. These are read-only: they parse the
Hermes projects.db, validate paths, and return structured JSON — they do NOT
set_active, change cwd, or mutate any DB state.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock
from typing import Any, Optional

_lock = RLock()
_pending: dict[str, dict[str, Any]] = {}

# ── Project tools (TUI-only pure intent) ───────────────────────────────
# These are exposed ONLY through agent-tui.yaml. The shared agent.yaml
# must NOT include them.  They are read-only queries against the Hermes
# profile's projects.db and do not mutate any DB or filesystem state.


def _get_hermes_home() -> str:
    """Return the active HERMES_HOME for project DB queries.

    Override via get_hermes_home() for test isolation.  Falls back to
    the canonical hermes_constants function then the environment.
    """
    try:
        from hermes_constants import get_hermes_home
        return str(get_hermes_home())
    except ImportError:
        return str(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))


def _projects_db_path(hermes_home: str) -> Path:
    return Path(hermes_home) / "projects.db"


def _resolve_project(conn, token: str):
    """Resolve a project by id, slug, or name (case-insensitive)."""
    from hermes_cli import projects_db as pdb

    token = (token or "").strip()
    if not token:
        return None
    projects = pdb.list_projects(conn, include_archived=True)
    for proj in projects:
        if token in (proj.id, proj.slug) or proj.name == token:
            return proj
    low = token.lower()
    for proj in projects:
        if proj.slug.lower() == low or proj.name.lower() == low:
            return proj
    return None


def _primary_path(proj) -> Optional[str]:
    """Return the project's primary filesystem path."""
    if getattr(proj, "primary_path", None):
        return proj.primary_path
    for folder in getattr(proj, "folders", []) or []:
        if getattr(folder, "is_primary", False):
            return folder.path
    folders = getattr(proj, "folders", []) or []
    return str(folders[0].path) if folders else None


def project_list_intent() -> str:
    """Read-only project listing. Returns JSON with projects array.

    Pure intent — no side effects, no DB mutations, no cwd changes.
    """
    try:
        from hermes_cli import projects_db as pdb
    except ImportError:
        return json.dumps({"success": False, "error": "projects_db unavailable"})

    hermes_home = _get_hermes_home()
    db_path = _projects_db_path(hermes_home)
    if not db_path.is_file():
        return json.dumps({"success": False, "error": "no projects database"})

    try:
        with pdb.connect_closing() as conn:
            active = pdb.get_active_id(conn)
            projects = pdb.list_projects(conn)

        return json.dumps({
            "active_id": active,
            "note": (
                "active_id identifies the logical project, not a specific worktree. "
                "Multiple folders can belong to one project. After project_switch, "
                "the Hermes success receipt selected_path is authoritative."
            ),
            "projects": [
                {
                    "id": p.id,
                    "slug": getattr(p, "slug", ""),
                    "name": p.name,
                    "primary_path": _primary_path(p),
                    "folders": [
                        {
                            "path": str(folder.path),
                            "is_primary": bool(folder.is_primary),
                            "added_at": getattr(folder, "added_at", None),
                        }
                        for folder in p.folders
                    ],
                    "active": p.id == active,
                }
                for p in projects
            ],
        })
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)})


def project_switch_intent(project: str) -> str:
    """Read-only project switch intent.

    Looks up the project by id/slug/name in the Hermes projects.db,
    verifies that the primary_path exists on disk, and returns a structured
    JSON response.

    **Pure intent** — does NOT call set_active, does NOT change cwd, does
    NOT mutate any DB state.  The caller (TUI gateway server.py) is
    responsible for applying the actual switch.

    Returns:
        On success: ``{"success": true, "workspace_switch": {"project_id": ...,
                     "project_name": ..., "path": ...}}``
        On failure: ``{"success": false, "error": "..."}``
    """
    try:
        from hermes_cli import projects_db as pdb
    except ImportError:
        return json.dumps({"success": False, "error": "projects_db unavailable"})

    token = (project or "").strip()
    if not token:
        return json.dumps({"success": False, "error": "project identifier is required"})

    hermes_home = _get_hermes_home()
    db_path = _projects_db_path(hermes_home)

    if not db_path.is_file():
        return json.dumps({"success": False, "error": "no projects database"})

    try:
        with pdb.connect_closing() as conn:
            proj = _resolve_project(conn, token)
            selected_path = None
            if proj is None and Path(token).expanduser().is_absolute():
                requested = str(Path(token).expanduser().resolve())
                for candidate in pdb.list_projects(conn):
                    for folder in candidate.folders:
                        canonical = str(Path(folder.path).expanduser().resolve())
                        if canonical == requested:
                            proj = candidate
                            selected_path = canonical
                            break
                    if proj is not None:
                        break
            if proj is None:
                return json.dumps({"success": False, "error": f"no project matching '{project}'"})

            target_path = selected_path or _primary_path(proj)
            if not target_path:
                return json.dumps({"success": False, "error": "project has no primary path"})

            resolved_path = str(Path(target_path).expanduser().resolve())
            if not Path(resolved_path).is_dir():
                return json.dumps({
                    "success": False,
                    "error": f"project primary path does not exist: {resolved_path}",
                })

            return json.dumps({
                "success": True,
                "workspace_switch": {
                    "project_id": proj.id,
                    "project_name": proj.name,
                    "path": resolved_path,
                },
            })
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)})


# ── Pending action registry (existing North bridge) ────────────────────


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
