"""North TUI/Gateway regression tests.

旧版test_north_surface_parity.py（1068行，11用例）经独立审计无效：TUI/Gateway
同一run_turn自比较、history seed core pass、expect_tool_sequence死pass、RED
只是源码字符串、无silent WS+REST running/failed覆盖、非实际TUI dispatch。已替换。

保留有价值的回归用例：
  - cancel POST idempotent guard
  - WS正常完成 + REST reconcile（partial保留）
  - WS静默（有partial）+ REST完成 — REST优先不丢partial
  - workspace_switch TUI callback vs Gateway unsupported

新增silent WS并发REST轮询修复的seam测试参见test_north_ws_silent_fix.py。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import pytest
from aiohttp import web

from gateway.north_coder_runtime import (
    NorthCoderRuntime,
    NorthCoderRuntimeConfig,
)


def _make_app(*, routes: list[web.RouteDef]) -> web.Application:
    app = web.Application()
    for route in routes:
        app.router.add_route(route[0], route[1], route[2])
    return app


def _make_runtime(
    host: str,
    port: int,
    hermes_home: Path,
    *,
    workspace_switch_supported: bool = False,
) -> NorthCoderRuntime:
    return NorthCoderRuntime(
        NorthCoderRuntimeConfig(
            base_url=f"http://{host}:{port}",
            workspace_switch_supported=workspace_switch_supported,
            timeout_seconds=5.0,
            seed_token_budget=2000,
            agent_yaml_path=str(hermes_home / "north-coder-profile" / "agent.yaml"),
        ),
        hermes_home,
    )


# ===================================================================
# 回归1: Cancel — at most one POST, interrupted状态正确
# ===================================================================


@pytest.mark.asyncio
async def test_cancel_idempotent_post_once(tmp_path, aiohttp_server):
    """cancel_session_async POST至多一次，两次调用仍为1次."""
    accepted = asyncio.Event()
    release_response = asyncio.Event()
    cancel_all: list[str] = []

    async def composite_run(request):
        accepted.set()
        await release_response.wait()
        return web.json_response({
            "conversation_id": "conv-cancel",
            "workspace_id": "home-default",
            "invocation_id": "inv-cancel",
            "status": "running",
        }, status=202)

    async def cancel_invocation(request):
        inv_id = request.match_info["invocation_id"]
        cancel_all.append(inv_id)
        return web.json_response({"status": "cancelled"})

    async def invocation_result(request):
        return web.json_response({"status": "cancelled", "blocks": []})

    app = _make_app(routes=[
        ("POST", "/api/run", composite_run),
        ("POST", "/api/invocations/{invocation_id}/cancel", cancel_invocation),
        ("GET", "/api/invocations/{invocation_id}/result", invocation_result),
    ])
    server = await aiohttp_server(app)

    gw_runtime = _make_runtime(server.host, server.port, tmp_path)

    cancel_all.clear()
    accepted.clear()
    release_response.clear()

    gw_turn = asyncio.create_task(
        gw_runtime.run_turn(
            message="cancel me",
            session_key="regr-cancel-gw",
            hermes_session_id="h-cancel-gw",
            workspace_id="home-default",
        )
    )
    await accepted.wait()

    # Two cancel calls — must produce at most one POST
    await gw_runtime.cancel_session_async("regr-cancel-gw")
    await asyncio.sleep(0.05)
    await gw_runtime.cancel_session_async("regr-cancel-gw")
    await asyncio.sleep(0.05)

    release_response.set()
    gw_result = await gw_turn
    assert gw_result.get("interrupted") is True, "cancel应产生interrupted=True"
    assert len(cancel_all) == 1, (
        f"cancel必须POST且仅一次，got {len(cancel_all)}: {cancel_all}"
    )


# ===================================================================
# 回归2: WS正常完成 + REST reconcile — partial保留
# ===================================================================


@pytest.mark.asyncio
async def test_ws_normal_complete_rest_reconcile(tmp_path, aiohttp_server):
    """WS正常发run_finished后关闭，REST返回completed → reconcile保留partial deltas."""
    async def composite_run(request):
        return web.json_response({
            "conversation_id": "conv-normal-ws",
            "workspace_id": "home-default",
            "invocation_id": "inv-normal-ws",
            "status": "running",
        }, status=202)

    async def invocation_result(request):
        return web.json_response({
            "status": "completed",
            "blocks": [{"role": "assistant", "block_type": "text", "content": "FULL REST TEXT"}],
        })

    async def websocket(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "run_started", "messageId": "m1"})
        await ws.send_json({"type": "text_message_content", "messageId": "m1", "delta": "partial "})
        await ws.send_json({"type": "run_finished", "messageId": "m1"})
        await ws.close()
        return ws

    app = _make_app(routes=[
        ("POST", "/api/run", composite_run),
        ("GET", "/api/invocations/{invocation_id}/result", invocation_result),
        ("GET", "/ws/conversation/{conversation_id}", websocket),
    ])
    server = await aiohttp_server(app)

    deltas: list[str] = []
    runtime = _make_runtime(server.host, server.port, tmp_path)

    result = await runtime.run_turn(
        message="ws partial test",
        session_key="regr-wserr-gw",
        hermes_session_id="h-wserr-gw",
        workspace_id="home-default",
        on_delta=lambda d: deltas.append(d),
    )

    # WS正常完成（无transport_error），partial deltas应保留
    assert result.get("completed") is True
    assert "".join(deltas) == "partial "
    assert result.get("final_response") is not None


# ===================================================================
# 回归3: WS静默（有partial）+ REST完成 → REST优先且WS partial不丢
# WS发partial后静默，REST返回completed（循环running中延迟），
# REST先terminal → 取消WS → partial保留
# ===================================================================


@pytest.mark.asyncio
async def test_ws_hangs_rest_eventually_completes(tmp_path, aiohttp_server):
    """WS发partial后hang，REST先返回completed → REST优先，partial不丢."""

    async def composite_run(request):
        return web.json_response({
            "conversation_id": "conv-hang-rest",
            "workspace_id": "home-default",
            "invocation_id": "inv-hang-rest",
            "status": "running",
        }, status=202)

    rest_call_count = 0

    async def invocation_result(request):
        nonlocal rest_call_count
        rest_call_count += 1
        # Delay completed response enough so WS gets to send its partial first
        if rest_call_count < 8:
            return web.json_response({"status": "running", "blocks": []})
        return web.json_response({
            "status": "completed",
            "blocks": [{"role": "assistant", "block_type": "text", "content": "FULL REST TEXT"}],
        })

    async def websocket(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "run_started", "messageId": "m1"})
        await ws.send_json({"type": "text_message_content", "messageId": "m1", "delta": "partial "})
        # Hang forever
        await asyncio.Event().wait()
        return ws

    app = _make_app(routes=[
        ("POST", "/api/run", composite_run),
        ("GET", "/api/invocations/{invocation_id}/result", invocation_result),
        ("GET", "/ws/conversation/{conversation_id}", websocket),
    ])
    server = await aiohttp_server(app)

    deltas: list[str] = []
    runtime = _make_runtime(server.host, server.port, tmp_path)

    result = await runtime.run_turn(
        message="ws hang test",
        session_key="regr-hang-gw",
        hermes_session_id="h-hang-gw",
        workspace_id="home-default",
        on_delta=lambda d: deltas.append(d),
    )

    # REST completed before WS → authoritative from REST, WS cancelled
    assert result.get("completed") is True, (
        f"REST completed应使最终状态为completed，got {result}"
    )
    # on_delta fires for the partial received before WS was cancelled
    assert "".join(deltas) == "partial ", (
        f"partial delta应在on_delta中被收到：{''.join(deltas)!r}"
    )
    # REST completes first, but WS partial is retained (no transport_error clear)
    assert rest_call_count >= 8, (
        f"REST应被轮询至少8次才得到completed，got {rest_call_count}"
    )
# ===================================================================
# 回归4: Workspace switch — Gateway unsupported vs TUI callback
# ===================================================================


@pytest.mark.asyncio
async def test_workspace_switch_gateway_unsupported(tmp_path, aiohttp_server):
    """Gateway默认(workspace_switch_supported=False)拒绝workspace_switch."""
    async def composite_run(request):
        return web.json_response({
            "conversation_id": "conv-ws-switch",
            "workspace_id": "home-default",
            "invocation_id": "inv-ws-switch",
            "status": "running",
        }, status=202)

    async def invocation_result(request):
        return web.json_response({
            "status": "completed",
            "blocks": [{"role": "assistant", "block_type": "text", "content": "switch done"}],
        })

    async def websocket(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "run_started", "messageId": "m1"})
        valid_ws = json.dumps({
            "success": True,
            "workspace_switch": {"project_id": "p99", "project_name": "Target", "path": "/tmp"},
        })
        await ws.send_json({"type": "tool_call_start", "toolCallId": "ws-1", "toolCallName": "project_switch"})
        await ws.send_json({"type": "tool_call_result", "toolCallId": "ws-1", "content": valid_ws})
        await ws.send_json({"type": "text_message_content", "messageId": "m1", "delta": "switching"})
        await ws.send_json({"type": "run_finished", "messageId": "m1"})
        await ws.close()
        return ws

    app = _make_app(routes=[
        ("POST", "/api/run", composite_run),
        ("GET", "/api/invocations/{invocation_id}/result", invocation_result),
        ("GET", "/ws/conversation/{conversation_id}", websocket),
    ])
    server = await aiohttp_server(app)

    runtime = _make_runtime(server.host, server.port, tmp_path, workspace_switch_supported=False)

    result = await runtime.run_turn(
        message="switch project",
        session_key="regr-ws-switch-gw",
        hermes_session_id="h-ws-switch-gw",
        workspace_id="home-default",
    )

    assert result.get("failed") is True, "Gateway必须标记为failed"
    assert result.get("status") == "unsupported", (
        f"Gateway status应为unsupported，got {result.get('status')}"
    )
    assert result.get("completed") is False

    # Gateway不得detach session binding
    binding = await runtime._conversation_binding("regr-ws-switch-gw")
    assert binding is not None and binding[0] is not None, "Gateway不得detach session"
