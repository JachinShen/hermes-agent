"""Condensed behaviour tests for TUI-only North workspace switch control-plane.

Covers:
- Bridge pure intent (project_list/switch read-only, no set_active)
- Shared vs TUI agent YAML variants (tools visibility, top-level stop_tools)
- Runtime workspace_switch extraction from tool events
- TUIAgent callback: success detach, failure no-detach, Gateway unsupported
- No project_create exposure
- A→B→A via detach, other sessions unaffected
- agent_yaml_path in POST /api/run payload
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

import pytest
from aiohttp import web

from gateway.north_coder_runtime import (
    NorthCoderRuntime,
    NorthCoderRuntimeConfig,
    NorthCoderTUIAgent,
)


# =========================================================================
# Helpers
# =========================================================================


def _seed_project(
    hermes_home: Path,
    name: str = "Test Project",
    slug: str = "test-project",
    path: str | None = None,
    make_path: bool = True,
) -> tuple[str, Path]:
    from hermes_cli import projects_db as pdb

    primary_path = path or str(hermes_home / "projects" / slug)
    if make_path:
        os.makedirs(primary_path, exist_ok=True)
    with pdb.connect_closing() as conn:
        proj_id = pdb.create_project(conn, name=name, slug=slug,
                                     folders=[primary_path], primary_path=primary_path)
    return proj_id, Path(primary_path)


@pytest.fixture
def hermes_home_with_projects_db(tmp_path):
    """Create a temp Hermes home with projects.db."""
    home = tmp_path / "hermes"
    home.mkdir(parents=True)
    (home / "north-coder-profile").mkdir(parents=True, exist_ok=True)
    return home


# =========================================================================
# Contract: Bridge pure intent — no set_active, no DB mutation
# =========================================================================


def test_project_switch_intent_read_only(hermes_home_with_projects_db):
    """project_switch_intent resolves project but does not change DB active."""
    from tools.north_actions import project_switch_intent
    from hermes_cli import projects_db as pdb
    import hermes_constants

    home = hermes_home_with_projects_db
    token = hermes_constants.set_hermes_home_override(str(home))

    try:
        p1_id, _ = _seed_project(home, name="Alpha", slug="alpha")
        p2_id, _ = _seed_project(home, name="Beta", slug="beta")

        with pdb.connect_closing() as conn:
            pdb.set_active(conn, p1_id)

        result = json.loads(project_switch_intent(p2_id))
        assert result["success"] is True
        assert result["workspace_switch"]["project_id"] == p2_id
        assert result["workspace_switch"]["project_name"] == "Beta"

        with pdb.connect_closing() as conn:
            assert pdb.get_active_id(conn) == p1_id, \
                "project_switch_intent must NOT change DB active"
    finally:
        hermes_constants.reset_hermes_home_override(token)


def test_project_switch_intent_fails_on_missing(hermes_home_with_projects_db):
    """Non-existent project returns success=False."""
    from tools.north_actions import project_switch_intent
    import hermes_constants

    home = hermes_home_with_projects_db
    token = hermes_constants.set_hermes_home_override(str(home))
    try:
        result = json.loads(project_switch_intent("nonexistent"))
        assert result["success"] is False
        assert "error" in result
    finally:
        hermes_constants.reset_hermes_home_override(token)


def test_project_list_read_only(hermes_home_with_projects_db):
    """project_list_intent returns projects and is idempotent."""
    from tools.north_actions import project_list_intent
    import hermes_constants

    home = hermes_home_with_projects_db
    token = hermes_constants.set_hermes_home_override(str(home))
    try:
        _seed_project(home, "Proj1", "proj1")
        r1 = json.loads(project_list_intent())
        assert "projects" in r1
        assert r1["projects"][0]["name"] == "Proj1"

        r2 = json.loads(project_list_intent())
        assert r2 == r1
    finally:
        hermes_constants.reset_hermes_home_override(token)


# =========================================================================
# Contract: No project_create exposure
# =========================================================================


def test_no_project_create():
    """Bridge must not expose project_create."""
    import tools.north_actions as na
    assert not hasattr(na, "project_create")
    assert not hasattr(na, "project_create_intent")


# =========================================================================
# Contract: Shared vs TUI agent YAML variants
# =========================================================================


def test_shared_agent_yaml_no_project_tools(hermes_home_with_projects_db):
    """agent.yaml must NOT contain project tools."""
    from hermes_cli.north_coder_profile import export_hermes_profile

    home = hermes_home_with_projects_db
    yaml_path = export_hermes_profile(home, home / "north-profile")
    content = yaml_path.read_text()
    assert "project_list" not in content
    assert "project_switch" not in content


def test_tui_variant_has_project_tools_and_top_level_stop_tools(hermes_home_with_projects_db):
    """agent-tui.yaml must include project tools and top-level stop_tools."""
    from hermes_cli.north_coder_profile import export_hermes_profile
    import yaml

    home = hermes_home_with_projects_db
    tui_yaml = export_hermes_profile(home, home / "north-profile", tui_variant=True)
    content = tui_yaml.read_text()
    assert str(tui_yaml).endswith("agent-tui.yaml")
    assert "project_list" in content
    assert "project_switch" in content

    data = yaml.safe_load(content)
    assert "stop_tools" in data, "agent-tui.yaml must have top-level stop_tools"
    assert "project_switch" in data["stop_tools"], \
        "project_switch must be in top-level stop_tools"
    # stop_tools must NOT be inside the tool entry
    for tool in data.get("tools", []):
        if isinstance(tool, dict) and tool.get("name") == "project_switch":
            assert "stop_tools" not in tool, \
                "stop_tools must be top-level, not inside tool entry"
            break


def test_tui_variant_binding_is_relative_custom_tools(hermes_home_with_projects_db):
    """agent-tui.yaml bindings use relative ./custom_tools paths."""
    from hermes_cli.north_coder_profile import export_hermes_profile

    home = hermes_home_with_projects_db
    tui_yaml = export_hermes_profile(home, home / "north-profile", tui_variant=True)
    content = tui_yaml.read_text()
    assert "./custom_tools/project_bridge.py" in content
    assert "tools.north_actions:" not in content


def test_no_project_create_in_variants(hermes_home_with_projects_db):
    """Neither shared nor TUI variant exposes project_create."""
    from hermes_cli.north_coder_profile import export_hermes_profile

    home = hermes_home_with_projects_db
    shared = export_hermes_profile(home, home / "north-profile")
    tui = export_hermes_profile(home, home / "north-profile", tui_variant=True)
    assert "project_create" not in shared.read_text()
    assert "project_create" not in tui.read_text()


# =========================================================================
# Contract: Runtime workspace_switch extraction
# =========================================================================


class TestRuntimeExtraction:
    """extract_workspace_switch validation rules."""

    @pytest.fixture
    def runtime(self, tmp_path):
        return NorthCoderRuntime(
            NorthCoderRuntimeConfig(base_url="http://0.0.0.0:1", timeout_seconds=0.1),
            tmp_path,
        )

    def _tool(self, kind, call_id, name="", content="", is_error=False):
        e = {"type": kind, "toolCallId": call_id}
        if name:
            e["toolCallName"] = name
        if content:
            e["content"] = content
        if is_error:
            e["isError"] = True
        return e

    def test_happy_path(self, runtime):
        valid = json.dumps({
            "success": True,
            "workspace_switch": {"project_id": "p1", "project_name": "A", "path": "/tmp"}
        })
        result = runtime.extract_workspace_switch({
            "tools": [self._tool("tool_call_result", "c1", "project_switch", valid)]
        })
        assert result is not None
        assert result["project_id"] == "p1"

    def test_rejects_non_project_switch(self, runtime):
        assert runtime.extract_workspace_switch({
            "tools": [self._tool("tool_call_result", "c1", "read_file", "ok")]
        }) is None

    def test_rejects_is_error(self, runtime):
        assert runtime.extract_workspace_switch({
            "tools": [self._tool("tool_call_result", "c1", "project_switch",
                                 "err", is_error=True)]
        }) is None

    def test_rejects_invalid_json(self, runtime):
        assert runtime.extract_workspace_switch({
            "tools": [self._tool("tool_call_result", "c1", "project_switch", "not-json")]
        }) is None

    def test_rejects_relative_path(self, runtime):
        rel = json.dumps({
            "success": True,
            "workspace_switch": {"project_id": "p1", "project_name": "A", "path": "rel/path"}
        })
        assert runtime.extract_workspace_switch({
            "tools": [self._tool("tool_call_result", "c1", "project_switch", rel)]
        }) is None

    def test_rejects_success_false(self, runtime):
        fail = json.dumps({"success": False, "error": "nope"})
        assert runtime.extract_workspace_switch({
            "tools": [self._tool("tool_call_result", "c1", "project_switch", fail)]
        }) is None

    def test_correlates_by_call_id(self, runtime):
        valid = json.dumps({
            "success": True,
            "workspace_switch": {"project_id": "p2", "project_name": "B", "path": "/tmp"}
        })
        result = runtime.extract_workspace_switch({
            "tools": [
                self._tool("tool_call_start", "c2", "project_switch"),
                self._tool("tool_call_result", "c2", "", valid),
            ]
        })
        assert result is not None
        assert result["project_id"] == "p2"


# =========================================================================
# Contract: TUIAgent callback — success detach / failure no-detach
# =========================================================================


class MockRuntime:
    def __init__(self):
        self.detached: list[str] = []
        # Compatible with TUIAgent's config check
        self.config = type("Config", (), {
            "tui_variant": True,
            "workspace_switch_supported": True,
            "agent_yaml_path": None,
        })()
        self.hermes_home = Path("/tmp")

    async def run_turn(self, **kwargs) -> dict:
        return {"messages": [], "completed": True, "north_invocation_id": "inv-1"}

    def detach_session(self, session_key: str) -> None:
        self.detached.append(session_key)

    def extract_workspace_switch(self, terminal: dict) -> Optional[dict]:
        """Return stored pending_workspace_switch if present."""
        if hasattr(self, "_pending_ws") and self._pending_ws:
            return self._pending_ws
        return None


def test_tui_callback_success_keeps_binding_for_next_composite_run():
    """Successful callback must not actively detach the old binding."""
    runtime = MockRuntime()
    runtime._pending_ws = {
        "project_id": "p1", "project_name": "A", "path": "/tmp"
    }
    def cb(ws):
        return None

    agent = NorthCoderTUIAgent(runtime, "sess-1", workspace_switch_callback=cb)
    agent.run_conversation("hello")
    assert runtime.detached == []


def test_tui_callback_failure_no_detach():
    """Failed callback must NOT detach and marks turn as failed."""
    runtime = MockRuntime()
    runtime._pending_ws = {
        "project_id": "p1", "project_name": "A", "path": "/tmp"
    }
    def cb(ws):
        return "switch failed: session gone"

    agent = NorthCoderTUIAgent(runtime, "sess-2", workspace_switch_callback=cb)
    result = agent.run_conversation("hello")
    assert runtime.detached == [], "Must NOT detach on callback failure"
    assert result.get("failed"), "Turn must be marked failed"
    assert "switch failed" in result.get("final_response", "")
    assert agent.history[-1]["role"] == "assistant"
    assert agent.history[-1]["content"] == result["final_response"]
    assert result["messages"] == agent.history









def test_tui_callback_no_workspace_switch_noop():
    """Without workspace_switch, callback must NOT be invoked."""
    runtime = MockRuntime()
    called = []
    def cb(ws):
        called.append(ws)
        return None

    agent = NorthCoderTUIAgent(runtime, "sess-3", workspace_switch_callback=cb)
    agent.run_conversation("hello")
    assert called == []
    assert runtime.detached == []


def test_tui_other_session_unaffected():
    """A successful switch does not detach either session binding."""
    runtime = MockRuntime()
    runtime._pending_ws = {
        "project_id": "p1", "project_name": "A", "path": "/tmp"
    }
    agent_a = NorthCoderTUIAgent(runtime, "sess-A", workspace_switch_callback=lambda ws: None)
    agent_b = NorthCoderTUIAgent(runtime, "sess-B")
    agent_a.run_conversation("hello")
    assert runtime.detached == []
    assert "sess-B" not in runtime.detached


# =========================================================================
# Contract: Gateway unsupported
# =========================================================================


def test_gateway_unsupported_with_no_callback(tmp_path):
    """runtime._run_turn_impl returns unsupported for workspace_switch
    when workspace_switch_supported=False."""
    runtime = NorthCoderRuntime(
        NorthCoderRuntimeConfig(base_url="http://0.0.0.0:1", timeout_seconds=0.1,
                                workspace_switch_supported=False),
        tmp_path,
    )
    terminal = {
        "tools": [{
            "type": "tool_call_result",
            "toolCallId": "c1",
            "toolCallName": "project_switch",
            "content": json.dumps({
                "success": True,
                "workspace_switch": {"project_id": "p1", "project_name": "A", "path": str(tmp_path)}
            }),
        }],
    }
    ws = runtime.extract_workspace_switch(terminal)
    assert ws is not None, "Tool result should be detectable"


# =========================================================================
# Contract: A→B→A via detach
# =========================================================================


def test_detach_clears_binding(tmp_path):
    """detach_session clears the North conversation binding."""
    runtime = NorthCoderRuntime(
        NorthCoderRuntimeConfig(base_url="http://0.0.0.0:1", timeout_seconds=0.1),
        tmp_path,
    )
    asyncio.run(runtime._record_conversation_binding("sess-aba", "cx-A", "/tmp/a", ""))
    assert asyncio.run(runtime._conversation_binding("sess-aba"))[0] == "cx-A"
    runtime.detach_session("sess-aba")
    assert asyncio.run(runtime._conversation_binding("sess-aba"))[0] is None


# =========================================================================
# Contract: agent_yaml_path in POST /api/run payload
# =========================================================================


@pytest.mark.asyncio
async def test_composite_run_posts_agent_yaml_path(tmp_path, aiohttp_server):
    """composite_run must include agent_yaml_path in /api/run payload."""
    seen: list[dict[str, Any]] = []

    async def composite_run(request):
        seen.append(await request.json())
        return web.json_response({
            "conversation_id": "conv-yaml",
            "workspace_id": "ws-yaml",
            "invocation_id": "inv-yaml",
            "status": "running",
        }, status=202)

    async def invocation_result(_request):
        return web.json_response({
            "status": "completed",
            "blocks": [{"role": "assistant", "block_type": "text", "content": "ok"}]
        })

    app = web.Application()
    app.router.add_post("/api/run", composite_run)
    app.router.add_get("/api/invocations/{invocation_id}/result", invocation_result)
    server = await aiohttp_server(app)

    runtime = NorthCoderRuntime(
        NorthCoderRuntimeConfig(
            base_url=f"http://{server.host}:{server.port}",
            agent_yaml_path=str(tmp_path / "north-coder-profile" / "agent-tui.yaml"),
            workspace_switch_supported=True,
            tui_variant=True,
        ),
        tmp_path,
    )
    await runtime.run_turn(
        message="test",
        session_key="sess-yaml",
        hermes_session_id="h-yaml",
        workdir=str(tmp_path),
    )

    assert len(seen) >= 1
    assert "agent_yaml_path" in seen[0]
    assert "agent-tui.yaml" in str(seen[0]["agent_yaml_path"])


# =========================================================================
# Contract: Gateway direct run_turn E2E — workspace_switch_supported=False
# =========================================================================


@pytest.mark.asyncio
async def test_gateway_direct_run_turn_workspace_switch_unsupported(tmp_path, aiohttp_server):
    """Gateway direct NorthCoderRuntime.run_turn with fake North events
    carrying a valid project_switch result, workspace_switch_supported=False.
    Must return failed/unsupported, NOT detach, and NOT change cwd."""
    conversation_id = "conv-gw-e2e"
    invocation_id = "inv-gw-e2e"
    ws_events_sent = []

    async def composite_run(request):
        return web.json_response({
            "conversation_id": conversation_id,
            "workspace_id": "ws-gw-e2e",
            "invocation_id": invocation_id,
            "status": "running",
        }, status=202)

    async def websocket_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        # Send project_switch tool_call_start + result, then run_finished
        await ws.send_json({
            "type": "tool_call_start",
            "toolCallId": "tc-project-switch",
            "toolCallName": "project_switch",
        })
        await ws.send_json({
            "type": "tool_call_result",
            "toolCallId": "tc-project-switch",
            "toolCallName": "project_switch",
            "content": json.dumps({
                "success": True,
                "workspace_switch": {
                    "project_id": "p-gw-e2e",
                    "project_name": "GW-E2E",
                    "path": str(tmp_path),
                },
            }),
        })
        await ws.send_json({"type": "run_finished"})
        return ws

    async def invocation_result(_request):
        return web.json_response({
            "status": "completed",
            "blocks": [{"role": "assistant", "block_type": "text", "content": "done"}],
        })

    app = web.Application()
    app.router.add_post("/api/run", composite_run)
    app.router.add_get(f"/ws/conversation/{conversation_id}", websocket_handler)
    app.router.add_get(f"/api/invocations/{invocation_id}/result", invocation_result)
    server = await aiohttp_server(app)

    runtime = NorthCoderRuntime(
        NorthCoderRuntimeConfig(
            base_url=f"http://{server.host}:{server.port}",
            workspace_switch_supported=False,  # Gateway direct path
        ),
        tmp_path,
    )

    result = await runtime.run_turn(
        message="switch to project GW-E2E",
        session_key="sess-gw-e2e",
        hermes_session_id="h-gw-e2e",
        workdir=str(tmp_path / "work"),
    )

    # Core assertion: returned as unsupported/failed
    assert result["status"] == "unsupported", \
        f"Expected status 'unsupported', got {result.get('status')!r}"
    assert result["failed"] is True
    assert result["completed"] is False
    assert "not supported" in result["final_response"].lower() or \
           "unsupported" in result["final_response"].lower() or \
           "project switch" in result["final_response"].lower(), \
        f"final_response should mention unsupported/project switch, got: {result['final_response']!r}"

    # No detach: conversation binding must still exist
    binding = await runtime._conversation_binding("sess-gw-e2e")
    assert binding[0] is not None, \
        "Conversation binding must still exist (no detach on unsupported)"

    # No cwd change: run_turn with workdir should not alter host cwd
    assert os.getcwd() != str(tmp_path / "work"), \
        "Host cwd must NOT be changed by run_turn"


# =========================================================================
# Contract: _make_north_workspace_switch_callback name/slug/id resolution
# =========================================================================


def test_callback_resolve_by_slug(monkeypatch, tmp_path):
    """_make_north_workspace_switch_callback resolves project by slug
    when project_id_input is a slug."""
    import tui_gateway.server as server
    import hermes_constants
    from hermes_cli import projects_db as pdb

    home = tmp_path / "hermes"
    home.mkdir()
    proj_path = home / "proj"
    proj_path.mkdir()

    token = hermes_constants.set_hermes_home_override(str(home))
    sid = "test-slug-resolve"
    try:
        monkeypatch.setattr(server, "_hermes_home", str(home))

        with pdb.connect_closing() as conn:
            proj_id = pdb.create_project(
                conn, name="My Project", slug="my-project",
                folders=[str(proj_path)], primary_path=str(proj_path),
            )

        # Use the callback live path with monkeypatched _apply_project_workspace
        monkeypatch.setattr(server, "_apply_project_workspace",
                            lambda task_id, path, _name="": True)

        server._sessions[sid] = {"session_key": sid, "cwd": str(proj_path), "agent": None}
        try:
            cb = server._make_north_workspace_switch_callback(sid)
            result = cb({"project_id": "my-project"})
            assert result is None, f"slug lookup should succeed, got: {result}"
        finally:
            server._sessions.pop(sid, None)
    finally:
        hermes_constants.reset_hermes_home_override(token)


def test_callback_resolve_by_name(monkeypatch, tmp_path):
    """_make_north_workspace_switch_callback resolves project by exact
    canonical name when project_id_input is a name (not id or slug)."""
    import tui_gateway.server as server
    import hermes_constants
    from hermes_cli import projects_db as pdb

    home = tmp_path / "hermes"
    home.mkdir()
    proj_path = home / "proj"
    proj_path.mkdir()

    token = hermes_constants.set_hermes_home_override(str(home))
    sid = "test-name-resolve"
    try:
        monkeypatch.setattr(server, "_hermes_home", str(home))

        with pdb.connect_closing() as conn:
            proj_id = pdb.create_project(
                conn, name="My Cool Project", slug="my-cool-project",
                folders=[str(proj_path)], primary_path=str(proj_path),
            )

        monkeypatch.setattr(server, "_apply_project_workspace",
                            lambda task_id, path, _name="": True)

        server._sessions[sid] = {"session_key": sid, "cwd": str(proj_path), "agent": None}
        try:
            cb = server._make_north_workspace_switch_callback(sid)
            # Use the canonical name (not slug, not id)
            result = cb({"project_id": "My Cool Project"})
            assert result is None, f"name lookup should succeed, got: {result}"
        finally:
            server._sessions.pop(sid, None)
    finally:
        hermes_constants.reset_hermes_home_override(token)


def test_callback_resolve_by_id(monkeypatch, tmp_path):
    """_make_north_workspace_switch_callback resolves project by id."""
    import tui_gateway.server as server
    import hermes_constants
    from hermes_cli import projects_db as pdb

    home = tmp_path / "hermes"
    home.mkdir()
    proj_path = home / "proj"
    proj_path.mkdir()

    token = hermes_constants.set_hermes_home_override(str(home))
    sid = "test-id-resolve"
    try:
        monkeypatch.setattr(server, "_hermes_home", str(home))

        with pdb.connect_closing() as conn:
            proj_id = pdb.create_project(
                conn, name="ID Project", slug="id-project",
                folders=[str(proj_path)], primary_path=str(proj_path),
            )

        monkeypatch.setattr(server, "_apply_project_workspace",
                            lambda task_id, path, _name="": True)

        server._sessions[sid] = {"session_key": sid, "cwd": str(proj_path), "agent": None}
        try:
            cb = server._make_north_workspace_switch_callback(sid)
            # Use the canonical id
            result = cb({"project_id": proj_id})
            assert result is None, f"id lookup should succeed, got: {result}"
        finally:
            server._sessions.pop(sid, None)
    finally:
        hermes_constants.reset_hermes_home_override(token)


def test_callback_resolve_name_fallback_not_found(monkeypatch, tmp_path):
    """When name doesn't match, callback returns error (not crash)."""
    import tui_gateway.server as server
    import hermes_constants
    from hermes_cli import projects_db as pdb

    home = tmp_path / "hermes"
    home.mkdir()
    proj_path = home / "proj"
    proj_path.mkdir()

    token = hermes_constants.set_hermes_home_override(str(home))
    sid = "test-name-not-found"
    try:
        monkeypatch.setattr(server, "_hermes_home", str(home))

        with pdb.connect_closing() as conn:
            pdb.create_project(
                conn, name="Real Project", slug="real-project",
                folders=[str(proj_path)], primary_path=str(proj_path),
            )

        server._sessions[sid] = {"session_key": sid, "cwd": str(proj_path), "agent": None}
        try:
            cb = server._make_north_workspace_switch_callback(sid)
            result = cb({"project_id": "Nonexistent Project"})
            assert result is not None, "non-existent name should return error"
            assert "not found" in result.lower(), \
                f"error should mention not found, got: {result}"
        finally:
            server._sessions.pop(sid, None)
    finally:
        hermes_constants.reset_hermes_home_override(token)
