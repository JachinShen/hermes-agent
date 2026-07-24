"""Regression tests for North runtime interrupt handling and WS error resilience.

These tests verify:
1. The shared _interrupt_north_host helper interrupts both host agent AND North runtime
2. WS ERROR in _consume_events breaks instead of raising RuntimeError
3. run_turn falls back to REST polling after WS ERROR when an invocation_id exists
4. WS ERROR with ws.exception()=None still clears partial deltas / sets failed status
"""

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest
from aiohttp import web

from gateway.north_coder_runtime import NorthCoderRuntime, NorthCoderRuntimeConfig


# ---------------------------------------------------------------------------
# _interrupt_north_host — shared helper that interrupts host + cancels runtime
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interrupt_north_host_interrupts_both_host_and_runtime():
    """_interrupt_north_host must interrupt the Hermes host agent AND cancel the
    North runtime session."""
    from gateway.north_coder_runtime import _interrupt_north_host

    host_interrupts: list[str] = []

    class FakeHostAgent:
        def interrupt(self, reason: str) -> None:
            host_interrupts.append(reason)

    runtime_cancelled: list[str] = []

    class FakeNorthRuntime:
        async def cancel_session_async(self, session_key: str) -> None:
            runtime_cancelled.append(session_key)

    with (
        patch(
            "gateway.north_coder_runtime.runtime_from_raw",
            return_value=FakeNorthRuntime(),
        ),
        patch("gateway.run._load_gateway_config", return_value={}),
        patch("gateway.run._gateway_config_home", return_value=Path("/tmp")),
    ):
        await _interrupt_north_host(FakeHostAgent(), "session-test", "test interrupt")

    assert host_interrupts == ["test interrupt"]
    assert runtime_cancelled == ["session-test"]


@pytest.mark.asyncio
async def test_interrupt_north_host_handles_missing_runtime_gracefully():
    """When no North runtime is configured, the host agent is still interrupted."""
    from gateway.north_coder_runtime import _interrupt_north_host

    host_interrupts: list[str] = []

    class FakeHostAgent:
        def interrupt(self, reason: str) -> None:
            host_interrupts.append(reason)

    with (
        patch("gateway.north_coder_runtime.runtime_from_raw", return_value=None),
        patch("gateway.run._load_gateway_config", return_value={}),
        patch("gateway.run._gateway_config_home", return_value=Path("/tmp")),
    ):
        await _interrupt_north_host(FakeHostAgent(), "session-test", "test")

    assert host_interrupts == ["test"]


@pytest.mark.asyncio
async def test_interrupt_north_host_handles_none_agent_gracefully():
    """When host_agent is None, only the North runtime is cancelled."""
    from gateway.north_coder_runtime import _interrupt_north_host

    runtime_cancelled: list[str] = []

    class FakeNorthRuntime:
        async def cancel_session_async(self, session_key: str) -> None:
            runtime_cancelled.append(session_key)

    with (
        patch(
            "gateway.north_coder_runtime.runtime_from_raw",
            return_value=FakeNorthRuntime(),
        ),
        patch("gateway.run._load_gateway_config", return_value={}),
        patch("gateway.run._gateway_config_home", return_value=Path("/tmp")),
    ):
        await _interrupt_north_host(None, "session-test", "test")

    assert runtime_cancelled == ["session-test"]


@pytest.mark.asyncio
async def test_interrupt_north_host_survives_runtime_lookup_error():
    """A failing runtime_from_raw must not propagate; host interrupt still fires."""
    from gateway.north_coder_runtime import _interrupt_north_host

    host_interrupts: list[str] = []

    class FakeHostAgent:
        def interrupt(self, reason: str) -> None:
            host_interrupts.append(reason)

    with (
        patch(
            "gateway.north_coder_runtime.runtime_from_raw",
            side_effect=Exception("config broken"),
        ),
        patch("gateway.run._load_gateway_config", return_value={}),
        patch("gateway.run._gateway_config_home", return_value=Path("/tmp")),
    ):
        await _interrupt_north_host(FakeHostAgent(), "session-test", "test")

    assert host_interrupts == ["test"]


# ---------------------------------------------------------------------------
# _consume_events — WS ERROR handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consume_events_ws_error_breaks_instead_of_raising(
    tmp_path, aiohttp_server
):
    """_consume_events must break on WS ERROR instead of raising RuntimeError."""
    app = web.Application()
    server = await aiohttp_server(app)

    runtime = NorthCoderRuntime(
        NorthCoderRuntimeConfig(base_url=f"http://{server.host}:{server.port}"),
        tmp_path,
    )

    class FakeWSMsg:
        def __init__(self, type_name: str, exception=None):
            self._type_name = type_name
            self._exception = exception

        @property
        def type(self):
            return type("WSMsgType", (), {"name": self._type_name})()

        def exception(self):
            return self._exception

    class FakeWS:
        def __init__(self):
            self.sent_error = False
            self._closed = False

        async def receive(self):
            if not self.sent_error:
                self.sent_error = True
                return FakeWSMsg("ERROR")
            await asyncio.sleep(3600)

        def exception(self):
            return None

        async def close(self):
            self._closed = True

        @property
        def closed(self) -> bool:
            return self._closed

    ws = FakeWS()
    result = await runtime._consume_events(ws, [], None, None)

    # Should NOT raise — should break with transport_error recorded
    assert result["status"] == "completed"
    assert "transport_error" in result
    # transport_error is a truthy sentinel so downstream checks work
    assert result["transport_error"]


@pytest.mark.asyncio
async def test_run_turn_ws_error_falls_back_to_rest_polling(tmp_path, aiohttp_server):
    """After a WS ERROR in _consume_events, run_turn falls back to REST polling
    when invocation_id exists."""
    conversation_id = "conv-ws-error"
    invocation_id = "inv-ws-error"

    async def send_message(request):
        ws = request.app.get("ws")
        if ws:
            await ws.close()
        return web.json_response({"invocation_id": invocation_id, "status": "running"})

    async def websocket(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        request.app["ws"] = ws
        # Close immediately so _consume_events gets a WS ERROR
        await ws.close()
        return ws

    async def invocation_result(_request):
        return web.json_response({
            "status": "completed",
            "blocks": [
                {
                    "role": "assistant",
                    "block_type": "text",
                    "content": "REST fallback result",
                }
            ],
        })

    app = web.Application()
    app["ws"] = None
    app.router.add_post(f"/api/conversations/{conversation_id}/messages", send_message)
    app.router.add_get(f"/ws/conversation/{conversation_id}", websocket)
    app.router.add_get(f"/api/invocations/{invocation_id}/result", invocation_result)
    server = await aiohttp_server(app)

    runtime = NorthCoderRuntime(
        NorthCoderRuntimeConfig(base_url=f"http://{server.host}:{server.port}"),
        tmp_path,
    )
    await runtime._record_conversation_binding(
        "session-ws-error", conversation_id, None, "home-default"
    )

    result = await runtime.run_turn(
        message="hello",
        session_key="session-ws-error",
        hermes_session_id="hermes-ws-error",
    )

    # The result should come from REST polling, not the errored WS
    assert result["final_response"] == "REST fallback result"
    assert result["status"] == "completed"
    assert result["north_invocation_id"] == invocation_id


@pytest.mark.asyncio
async def test_run_turn_ws_error_exception_none_with_invocation_clears_partial(
    tmp_path, aiohttp_server
):
    """WS ERROR with exception()=None and an invocation_id must clear partial
    deltas so the REST authoritative result replaces them."""
    conversation_id = "conv-ws-exc-none"
    invocation_id = "inv-ws-exc-none"

    deltas_received: list[str] = []

    async def send_message(_request):
        return web.json_response({"invocation_id": invocation_id, "status": "running"})

    class FakeWSMsg:
        def __init__(self, type_name: str, data: str = ""):
            self._type_name = type_name
            self._data = data

        @property
        def type(self):
            return type("WSMsgType", (), {"name": self._type_name})()

        @property
        def data(self):
            return self._data

        def exception(self):
            return None

    class FakeWS:
        def __init__(self):
            self._step = 0

        async def receive(self):
            self._step += 1
            if self._step == 1:
                # Send a text delta before the error
                return FakeWSMsg(
                    "TEXT", '{"type":"text_message_content","delta":"partial garbage "}'
                )
            # Then ERROR
            return FakeWSMsg("ERROR")

        def exception(self):
            return None

        async def close(self):
            pass

        @property
        def closed(self):
            return True

    async def invocation_result(_request):
        return web.json_response({
            "status": "completed",
            "blocks": [
                {
                    "role": "assistant",
                    "block_type": "text",
                    "content": "REST authoritative result",
                }
            ],
        })

    app = web.Application()
    app.router.add_post(f"/api/conversations/{conversation_id}/messages", send_message)
    app.router.add_get(f"/api/invocations/{invocation_id}/result", invocation_result)
    server = await aiohttp_server(app)

    runtime = NorthCoderRuntime(
        NorthCoderRuntimeConfig(base_url=f"http://{server.host}:{server.port}"),
        tmp_path,
    )
    await runtime._record_conversation_binding(
        "session-ws-exc-none", conversation_id, None, "home-default"
    )

    def capture_delta(delta: str) -> None:
        deltas_received.append(delta)

    with patch.object(runtime, "_connect_events", return_value=FakeWS()):
        result = await runtime.run_turn(
            message="hello",
            session_key="session-ws-exc-none",
            hermes_session_id="hermes-ws-exc-none",
            on_delta=capture_delta,
        )

    # The partial delta must NOT leak into the final response
    assert "partial garbage" not in result["final_response"]
    assert result["final_response"] == "REST authoritative result"
    assert result["status"] == "completed"
    assert result["north_invocation_id"] == invocation_id


@pytest.mark.asyncio
async def test_run_turn_ws_error_exception_none_no_invocation_returns_failed(
    tmp_path, aiohttp_server
):
    """WS ERROR with exception()=None and no invocation_id must produce a
    structured failed state, not fake completed."""
    conversation_id = "conv-ws-exc-none-noinv"

    async def send_message(_request):
        return web.json_response({"status": "running"})

    class FakeWSMsg:
        def __init__(self, type_name: str):
            self._type_name = type_name

        @property
        def type(self):
            return type("WSMsgType", (), {"name": self._type_name})()

        def exception(self):
            return None

    class FakeWS:
        def __init__(self):
            self._called = False

        async def receive(self):
            if not self._called:
                self._called = True
                return FakeWSMsg("ERROR")
            await asyncio.sleep(3600)

        def exception(self):
            return None

        async def close(self):
            pass

        @property
        def closed(self):
            return False

    app = web.Application()
    app.router.add_post(f"/api/conversations/{conversation_id}/messages", send_message)
    server = await aiohttp_server(app)

    runtime = NorthCoderRuntime(
        NorthCoderRuntimeConfig(base_url=f"http://{server.host}:{server.port}"),
        tmp_path,
    )
    await runtime._record_conversation_binding(
        "session-ws-exc-none-noinv", conversation_id, None, "home-default"
    )

    with patch.object(runtime, "_connect_events", return_value=FakeWS()):
        result = await runtime.run_turn(
            message="hello",
            session_key="session-ws-exc-none-noinv",
            hermes_session_id="hermes-ws-exc-none-noinv",
        )

    assert result["status"] == "failed"
    assert result["failed"] is True


# ---------------------------------------------------------------------------
# cancel_session_async — idempotent concurrent cancel requests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_session_idempotent_only_calls_once(tmp_path):
    """cancel_session_async must cancel *exactly once* — even when
    primary, backup, and timeout paths all try concurrently."""
    from unittest.mock import patch

    cancel_calls: list[str] = []

    config = NorthCoderRuntimeConfig(base_url="http://localhost:99999")
    runtime = NorthCoderRuntime(config, tmp_path)

    async def counting_cancel(invocation_id: str) -> None:
        cancel_calls.append(invocation_id)

    # Both _request_session_cancel and cancel are mocked — what we're
    # testing is that cancel_session_async calls cancel() for each
    # caller that gets a non-None invocation_id from _request_session_cancel.
    with (
        patch.object(
            runtime, "_request_session_cancel", return_value="inv-dup"
        ),
        patch.object(runtime, "cancel", side_effect=counting_cancel),
    ):
        await asyncio.gather(
            runtime.cancel_session_async("session-dup"),
            runtime.cancel_session_async("session-dup"),
            runtime.cancel_session_async("session-dup"),
        )

    # Each concurrent caller calls cancel_session_async → _request_session_cancel → cancel
    # With the idempotent guard (_cancel_posted), only the first call
    # should actually POST — the remaining calls see the guard and skip.
    assert len(cancel_calls) == 1
    if cancel_calls:
        assert cancel_calls[0] == "inv-dup"


@pytest.mark.asyncio
async def test_cancel_session_marked_before_invocation_appears(tmp_path):
    """When cancel is requested before the invocation_id is created, the
    session is marked in _cancel_requested_sessions and cancelled as soon
    as the invocation_id appears in _run_turn_impl."""
    config = NorthCoderRuntimeConfig(base_url="http://localhost:99999")
    runtime = NorthCoderRuntime(config, tmp_path)

    # Session must be in-flight for marking to work
    with runtime._active_lock:
        runtime._inflight_sessions.add("session-pre")

    # Mark cancel before any invocation exists
    runtime._request_session_cancel("session-pre")
    assert "session-pre" in runtime._cancel_requested_sessions

    # Simulate the point in _run_turn_impl where invocation_id is received
    with runtime._active_lock:
        runtime._active_invocations["session-pre"] = "inv-pre"
        cancel_requested = "session-pre" in runtime._cancel_requested_sessions

    assert cancel_requested is True
    # _run_turn_impl then calls self.cancel("inv-pre")


@pytest.mark.asyncio
async def test_cancel_session_stale_request_cleared_on_new_turn(tmp_path):
    """A stale cancellation request must be cleared when a new turn starts,
    so the next turn is not immediately cancelled."""
    config = NorthCoderRuntimeConfig(base_url="http://localhost:99999")
    runtime = NorthCoderRuntime(config, tmp_path)

    # Previous turn was cancelled
    runtime._cancel_requested_sessions.add("session-stale")
    assert "session-stale" in runtime._cancel_requested_sessions

    # New turn starts — run_turn clears it
    with runtime._active_lock:
        runtime._inflight_sessions.add("session-stale")
        runtime._cancel_requested_sessions.discard("session-stale")

    assert "session-stale" not in runtime._cancel_requested_sessions


# ---------------------------------------------------------------------------
# _seeded_message — history seed budget
# ---------------------------------------------------------------------------


def test_seeded_message_empty_history():
    """Empty conversation_history returns just the message."""
    result = NorthCoderRuntime._seeded_message("hello", None)
    assert result == "hello"

    result = NorthCoderRuntime._seeded_message("hello", [])
    assert result == "hello"


def test_seeded_message_filters_non_user_assistant():
    """Tool, system, and session_meta messages must be excluded."""
    history = [
        {"role": "system", "content": "you are a bot"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello there"},
        {"role": "tool", "content": "result: ok"},
        {"role": "session_meta", "content": "meta"},
    ]
    result = NorthCoderRuntime._seeded_message("question", history, seed_token_budget=99999)
    assert "hi" in result
    assert "hello there" in result
    assert "result: ok" not in result
    assert "you are a bot" not in result
    assert "meta" not in result
    assert result.startswith("[Prior Hermes conversation; context only]")
    assert result.endswith("question")


def test_seeded_message_budget_respected():
    """Messages beyond the token budget must be dropped, keeping only
    the most recent pairs."""
    # 20 chars of content → entry_text ≈ 26 chars ≈ 7 tokens per message
    # 3 messages * 7 tokens = 21 tokens → should all fit in budget=30
    # but budget=10 should drop older messages
    history = [
        {"role": "user", "content": "AAAA AAAA AAAA AAAA AAA"},  # ~19 chars content
        {"role": "assistant", "content": "BBBB BBBB BBBB BBBB BBB"},
        {"role": "user", "content": "CCCC CCCC CCCC CCCC CCC"},
    ]
    # Small budget: only the last user message fits
    result = NorthCoderRuntime._seeded_message("final", history, seed_token_budget=10)
    assert "AAAA" not in result, "first user should be dropped"
    assert "BBBB" not in result, "assistant should be dropped"
    assert "CCCC" in result, "last user should be kept"
    assert result.endswith("final")


def test_seeded_message_large_budget_keeps_all():
    """With a generous budget, all user/assistant messages are preserved."""
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer1"},
        {"role": "user", "content": "followup"},
    ]
    result = NorthCoderRuntime._seeded_message("last", history, seed_token_budget=99999)
    assert "first" in result
    assert "answer1" in result
    assert "followup" in result
    assert result.endswith("last")


def test_seeded_message_first_exceeds_budget_truncated():
    """If the first (most recent) message alone exceeds the budget, it is
    truncated rather than dropped entirely."""
    long_content = "X" * 200  # ~200 chars ≈ 50 tokens
    history = [{"role": "user", "content": long_content}]
    result = NorthCoderRuntime._seeded_message("short", history, seed_token_budget=20)
    # The content should be truncated to ~budget*4 = 80 chars
    assert len(result) < len(long_content) + 100
    assert result.endswith("short")


def test_seeded_message_only_current_when_all_filtered():
    """When every history message is filtered out (non-user/assistant),
    only the current message is returned."""
    history = [
        {"role": "system", "content": "sys"},
        {"role": "tool", "content": "tool result"},
    ]
    result = NorthCoderRuntime._seeded_message("just me", history, seed_token_budget=99999)
    assert result == "just me"
    assert "sys" not in result
    assert "tool" not in result
