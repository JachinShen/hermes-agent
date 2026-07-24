"""RED seam tests: 并发REST轮询修复，防止silent WS死锁.

网关 contract:
  1) WS负责stream/progress，REST并发作为权威终态
  2) REST先terminal时结束WS消费
  3) WS先terminal时仍REST reconcile
  4) 双方都永久running时timeout_seconds为硬deadline

每个测试使用真实aiohttp TestServer模拟North 0.4行为。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from aiohttp import web

from gateway.north_coder_runtime import (
    NorthCoderRuntime,
    NorthCoderRuntimeConfig,
)

# ── 共享帮助函数 ────────────────────────────────────────────────────────


def _make_runtime(
    host: str,
    port: int,
    hermes_home: Path,
    *,
    timeout_seconds: float = 5.0,
) -> NorthCoderRuntime:
    return NorthCoderRuntime(
        NorthCoderRuntimeConfig(
            base_url=f"http://{host}:{port}",
            timeout_seconds=timeout_seconds,
            seed_token_budget=2000,
            agent_yaml_path=str(hermes_home / "north-coder-profile" / "agent.yaml"),
        ),
        hermes_home,
    )


# ===================================================================
# RED — 测试A: WS静默 + REST running→failed
# WS保持open不发事件，REST先running后failed
# 断言：快速返回failed，不等总timeout
# ===================================================================


@pytest.mark.asyncio
async def test_silent_ws_rest_becomes_failed(aiohttp_server, tmp_path):
    """WS保持静默，REST延迟返回failed → 应快速返回failed."""
    invoked = False
    poll_count = 0

    async def composite_run(request):
        nonlocal invoked
        invoked = True
        return web.json_response({
            "conversation_id": "conv-silent-fail",
            "workspace_id": "home-default",
            "invocation_id": "inv-silent-fail",
            "status": "running",
        }, status=202)

    async def invocation_result(request):
        nonlocal poll_count
        poll_count += 1
        if poll_count < 3:
            return web.json_response({"status": "running", "blocks": []})
        return web.json_response({
            "status": "failed",
            "blocks": [],
            "error": "provider transport failed",
        })

    async def websocket(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        # 永不发送任何事件——模拟provider无声失败
        await asyncio.Event().wait()
        return ws

    app = web.Application()
    app.router.add_post("/api/run", composite_run)
    app.router.add_get("/api/invocations/{inv_id}/result", invocation_result)
    app.router.add_get("/ws/conversation/{conv_id}", websocket)
    server = await aiohttp_server(app)

    runtime = _make_runtime(server.host, server.port, tmp_path, timeout_seconds=10.0)

    # ── 期望：REST先failed，快速返回（不等10s timeout）──
    result = await runtime.run_turn(
        message="hello",
        session_key="test-silent-fail",
        hermes_session_id="h-silent-fail",
        workspace_id="home-default",
    )

    assert result.get("failed") is True, (
        f"应在REST failed后快速返回failed，got status={result.get('status')}"
    )
    assert invoked is True
    assert poll_count >= 2, f"应至少轮询2次才得到failed，got {poll_count}"


# ===================================================================
# RED — 测试B: WS静默 + REST永久running → timeout
# WS和REST都不terminal，配置短timeout_seconds
# 断言：有界退出（TimeoutError），非无限卡住
# ===================================================================


@pytest.mark.asyncio
async def test_silent_ws_rest_running_timeout(aiohttp_server, tmp_path):
    """WS静默 + REST永远running → timeout_seconds硬deadline触发超时."""
    invoked = False

    async def composite_run(request):
        nonlocal invoked
        invoked = True
        return web.json_response({
            "conversation_id": "conv-timeout",
            "workspace_id": "home-default",
            "invocation_id": "inv-timeout",
            "status": "running",
        }, status=202)

    async def invocation_result(request):
        return web.json_response({"status": "running", "blocks": []})

    async def websocket(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await asyncio.Event().wait()
        return ws

    app = web.Application()
    app.router.add_post("/api/run", composite_run)
    app.router.add_get("/api/invocations/{inv_id}/result", invocation_result)
    app.router.add_get("/ws/conversation/{conv_id}", websocket)
    server = await aiohttp_server(app)

    runtime = _make_runtime(server.host, server.port, tmp_path, timeout_seconds=0.5)

    result = await runtime.run_turn(
        message="timeout test",
        session_key="test-timeout",
        hermes_session_id="h-timeout",
        workspace_id="home-default",
    )

    # WS先timeout(transport_error)，REST poll也timeout → 返回structured failed而非无限卡住
    assert result.get("status") == "failed", (
        f"双方timeout应返回failed状态，got {result.get('status')}"
    )
    assert invoked is True


# ===================================================================
# RED — 测试C: 正常WS stream + REST completed
# WS正常发delta + run_finished，REST返回completed
# 断言：不丢delta，正常返回
# ===================================================================


@pytest.mark.asyncio
async def test_normal_ws_stream_completed(aiohttp_server, tmp_path):
    """WS正常流式 + REST completed → 正常完成，不丢delta."""
    invoked = False

    async def composite_run(request):
        nonlocal invoked
        invoked = True
        return web.json_response({
            "conversation_id": "conv-normal",
            "workspace_id": "home-default",
            "invocation_id": "inv-normal",
            "status": "running",
        }, status=202)

    async def invocation_result(request):
        return web.json_response({
            "status": "completed",
            "blocks": [{"role": "assistant", "block_type": "text", "content": "full text"}],
        })

    async def websocket(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "run_started", "messageId": "m1"})
        await ws.send_json({"type": "text_message_content", "messageId": "m1", "delta": "Hello, "})
        await ws.send_json({"type": "text_message_content", "messageId": "m1", "delta": "North!"})
        await ws.send_json({"type": "run_finished", "messageId": "m1"})
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_post("/api/run", composite_run)
    app.router.add_get("/api/invocations/{inv_id}/result", invocation_result)
    app.router.add_get("/ws/conversation/{conv_id}", websocket)
    server = await aiohttp_server(app)

    deltas: list[str] = []
    runtime = _make_runtime(server.host, server.port, tmp_path, timeout_seconds=5.0)

    result = await runtime.run_turn(
        message="hello",
        session_key="test-normal-stream",
        hermes_session_id="h-normal-stream",
        workspace_id="home-default",
        on_delta=lambda d: deltas.append(d),
    )

    assert result.get("completed") is True, (
        f"正常流式应完成，got status={result.get('status')}"
    )
    assert "".join(deltas) == "Hello, North!", (
        f"delta不匹配：{''.join(deltas)!r}"
    )
    assert result.get("final_response") in ("Hello, North!", "full text"), (
        f"final_response应为delta或REST内容：{result.get('final_response')!r}"
    )
    assert invoked is True


# ===================================================================
# RED — 测试D: WS先完成 + REST延迟 → reconcile保留
# ===================================================================


@pytest.mark.asyncio
async def test_ws_finishes_first_rest_reconciles(aiohttp_server, tmp_path):
    """WS先发run_finished，REST延迟返回completed → reconcile保留."""
    invoked = False
    rest_called = False

    async def composite_run(request):
        nonlocal invoked
        invoked = True
        return web.json_response({
            "conversation_id": "conv-recon",
            "workspace_id": "home-default",
            "invocation_id": "inv-recon",
            "status": "running",
        }, status=202)

    async def invocation_result(request):
        nonlocal rest_called
        rest_called = True
        return web.json_response({
            "status": "completed",
            "blocks": [{"role": "assistant", "block_type": "text", "content": "rest full text"}],
        })

    async def websocket(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "run_started", "messageId": "m1"})
        await ws.send_json({"type": "text_message_content", "messageId": "m1", "delta": "partial "})
        await ws.send_json({"type": "run_finished", "messageId": "m1"})
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_post("/api/run", composite_run)
    app.router.add_get("/api/invocations/{inv_id}/result", invocation_result)
    app.router.add_get("/ws/conversation/{conv_id}", websocket)
    server = await aiohttp_server(app)

    runtime = _make_runtime(server.host, server.port, tmp_path, timeout_seconds=5.0)
    deltas: list[str] = []

    result = await runtime.run_turn(
        message="hello",
        session_key="test-recon-gw",
        hermes_session_id="h-recon-gw",
        workspace_id="home-default",
        on_delta=lambda d: deltas.append(d),
    )

    assert result.get("completed") is True, f"reconcile后应completed，got {result.get('status')}"
    assert "".join(deltas) == "partial ", f"delta应保留：{''.join(deltas)!r}"
    assert rest_called is True, "REST应被轮询"
    assert result.get("final_response") == "partial ", "final_response应保留WS delta"


# ===================================================================
# RED — 测试E: _consume_events有receive_timeout安全网
# ===================================================================


def test_consume_events_ws_receive_timeout(tmp_path):
    """_consume_events在ws.receive()长时间无消息时应设置transport_error并退出."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    runtime = NorthCoderRuntime(
        NorthCoderRuntimeConfig(base_url="http://0.0.0.0:1", timeout_seconds=0.5),
        tmp_path,
    )

    mock_ws = AsyncMock()

    async def receive_never_returns():
        await asyncio.Event().wait()  # 永远不返回

    mock_ws.receive = receive_never_returns

    full_response: list[str] = ["partial "]

    # 不应无限阻塞——应在timeout后退出
    result = asyncio.run(runtime._consume_events(
        mock_ws, full_response, on_delta=None, on_event=None,
    ))

    # 此时_consume_events应通过transport_error或异常退出
    assert result.get("transport_error") is not None or result.get("status") in ("error", "failed"), (
        f"应退出而非无限阻塞，got {result}"
    )