#!/usr/bin/env python3
"""Project tools — the agent's INTENTIONAL handle on first-class Projects.

Projects (per-profile ``projects.db``) are the named workspaces the desktop
sidebar groups sessions into. Creating / switching a project is a deliberate act
expressed as explicit tools — never a side effect of a terminal ``cd``.

Exposed only on GUI sessions: the tools live in the `project` toolset (kept off
``_HERMES_CORE_TOOLS``) which the desktop/TUI gateway folds into its resolved
toolsets, so no CLI/messaging/cron schema carries them. The GUI also wires
``set_project_workspace_callback`` so a create/switch re-anchors the live
session's cwd and the sidebar follows the move; the DB write is the durable part.
"""

import json
import os
from typing import Callable, Optional

from tools.registry import registry

# Set by the GUI gateway (tui_gateway) at session wiring. Receives
# ``(task_id, primary_path, project_name)`` and re-anchors that session's
# workspace + refreshes the sidebar. ``None`` in CLI / messaging contexts — the
# DB write still happens; there's just no live GUI session to move.
_workspace_callback: Optional[Callable[[str, str, str], bool]] = None


def set_project_workspace_callback(fn: Optional[Callable[[str, str, str], bool]]) -> None:
    global _workspace_callback
    _workspace_callback = fn


def _primary_path(proj) -> Optional[str]:
    if getattr(proj, "primary_path", None):
        return proj.primary_path
    for folder in proj.folders:
        if folder.is_primary:
            return folder.path
    return proj.folders[0].path if proj.folders else None


def _apply_workspace(task_id: Optional[str], path: Optional[str], name: str) -> bool:
    cb = _workspace_callback
    if not (cb and task_id and path):
        return True
    try:
        result = cb(task_id, path, name)
        return result is not False
    except Exception:
        return False


def _resolve(conn, token: str):
    from hermes_cli import projects_db as pdb

    token = (token or "").strip()
    if not token:
        return None
    projects = pdb.list_projects(conn, include_archived=True)
    # Exact id / slug / name first, then case-insensitive slug / name.
    for proj in projects:
        if token in (proj.id, proj.slug) or proj.name == token:
            return proj
    low = token.lower()
    for proj in projects:
        if proj.slug.lower() == low or proj.name.lower() == low:
            return proj
    return None


def project_list(task_id: Optional[str] = None) -> str:
    from hermes_cli import projects_db as pdb
    from agent.runtime_cwd import resolve_agent_cwd

    with pdb.connect_closing() as conn:
        active = pdb.get_active_id(conn)
        projects = pdb.list_projects(conn)

    current_path = os.path.abspath(os.path.expanduser(str(resolve_agent_cwd())))

    return json.dumps({
        "active_id": active,
        "current_path": current_path,
        "note": (
            "active_id is the logical project; current_path is the actual session "
            "checkout/worktree. Multiple worktrees may belong to the same project."
        ),
        "projects": [
            {
                "id": p.id,
                "slug": p.slug,
                "name": p.name,
                "primary_path": _primary_path(p),
                "folders": [
                    {
                        "path": folder.path,
                        "is_primary": bool(folder.is_primary),
                        "added_at": getattr(folder, "added_at", None),
                        "is_current": (
                            os.path.abspath(os.path.expanduser(folder.path))
                            == current_path
                        ),
                    }
                    for folder in p.folders
                ],
                "active": p.id == active,
            }
            for p in projects
        ],
    })


def project_create(name: str, path: Optional[str] = None, task_id: Optional[str] = None) -> str:
    name = (name or "").strip()
    if not name:
        return json.dumps({"success": False, "error": "name is required"})

    from hermes_cli import projects_db as pdb

    folder = (path or "").strip()
    if folder:
        folder = os.path.abspath(os.path.expanduser(folder))

    try:
        with pdb.connect_closing() as conn:
            pid = pdb.create_project(conn, name=name, folders=[folder] if folder else [], primary_path=folder or None)
            proj = pdb.get_project(conn, pid)
    except ValueError as exc:
        return json.dumps({"success": False, "error": str(exc)})

    if proj is None:
        return json.dumps({"success": False, "error": "project vanished after create"})

    primary = _primary_path(proj)
    workspace_switched = _apply_workspace(task_id, primary, proj.name)
    receipt = {
        "success": True,
        "id": proj.id,
        "slug": proj.slug,
        "name": proj.name,
        "primary_path": primary,
        "workspace_switched": bool(primary and workspace_switched),
    }
    if primary and not workspace_switched:
        receipt.update({
            "partial": True,
            "warning": "project created, but the session workdir could not be switched",
            "error": "failed to switch session workdir",
        })
    return json.dumps(receipt)


def project_switch(project: str, task_id: Optional[str] = None) -> str:
    from hermes_cli import projects_db as pdb

    selector = (project or "").strip()
    selected_path = None
    with pdb.connect_closing() as conn:
        proj = _resolve(conn, selector)
        if proj is None and os.path.isabs(os.path.expanduser(selector)):
            requested = os.path.abspath(os.path.expanduser(selector))
            for candidate in pdb.list_projects(conn):
                for folder in candidate.folders:
                    canonical = os.path.abspath(os.path.expanduser(folder.path))
                    if canonical == requested:
                        proj = candidate
                        selected_path = canonical
                        break
                if proj is not None:
                    break
        if proj is None:
            return json.dumps({"success": False, "error": f"no project matching '{project}'"})
    target_path = selected_path or _primary_path(proj)
    workspace_switched = _apply_workspace(task_id, target_path, proj.name)
    if not workspace_switched:
        return json.dumps({
            "success": False,
            "error": "failed to switch session workdir",
            "id": proj.id,
            "slug": proj.slug,
            "name": proj.name,
            "primary_path": _primary_path(proj),
        })

    return json.dumps({
        "success": True,
        "id": proj.id,
        "slug": proj.slug,
        "name": proj.name,
        "primary_path": _primary_path(proj),
        "selected_path": target_path,
        "workspace_switched": bool(target_path),
        "workspace_switch": {"project_id": proj.id, "project_name": proj.name, "path": target_path},
    })


registry.register(
    name="project_list",
    toolset="project",
    schema={
        "name": "project_list",
        "description": (
            "List logical desktop Projects and their registered folders/worktrees. "
            "Use current_path or folders[].is_current to determine the actual execution "
            "directory; active_id and primary_path do not distinguish worktrees."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    handler=lambda args, **kw: project_list(task_id=kw.get("task_id")),
)

registry.register(
    name="project_create",
    toolset="project",
    schema={
        "name": "project_create",
        "description": (
            "Create a desktop Project (a named workspace). Pass `path` to also request "
            "switching this chat into that folder. The receipt reports the created Project "
            "truthfully: success remains true if Project creation succeeds but workspace "
            "switching is partial; inspect workspace_switched, partial, warning, and error. "
            "This is the intentional way to move the session, not `cd`."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Human name, e.g. 'Aurora Demo'"},
                "path": {"type": "string", "description": "Primary repo/folder to anchor the project to"},
            },
            "required": ["name"],
        },
    },
    handler=lambda args, **kw: project_create(
        name=args.get("name", ""), path=args.get("path"), task_id=kw.get("task_id")
    ),
)

registry.register(
    name="project_switch",
    toolset="project",
    schema={
        "name": "project_switch",
        "description": (
            "Switch this chat into an existing desktop Project by name, slug, id, or an "
            "exact registered folder/worktree path. A project selector uses its primary "
            "folder; a registered folder path switches to that exact folder."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "project": {
                    "type": "string",
                    "description": "Project name, slug, id, or exact registered folder path",
                },
            },
            "required": ["project"],
        },
    },
    handler=lambda args, **kw: project_switch(project=args.get("project", ""), task_id=kw.get("task_id")),
)
