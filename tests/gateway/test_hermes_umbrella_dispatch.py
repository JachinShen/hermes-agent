"""Tests for Gateway `/hermes` umbrella routing.

The generic gateway dispatcher (`gateway/run.py`) receives MessageEvent
with ``text="/hermes runtime ncoder"`` on any platform (not just Slack).
Before the fix, ``event.get_command()`` returned ``"hermes"`` which is NOT
in ``GATEWAY_KNOWN_COMMANDS``, producing an "Unknown command `/hermes`"
response.

The fix: at the dispatch entry point, detect ``command == "hermes"`` and
expand it via ``hermes_cli.commands.slack_subcommand_map()`` into a canonical
slash command (e.g. ``/hermes runtime ncoder`` → ``/runtime ncoder``) before
normal dispatch continues.

Behaviour contract over snapshots — these tests assert the *shape* of the
outcome (expanded vs not-expanded, help fallback, recursion guard), not
exact implementation details.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )
    runner.session_store = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    return runner


# ------------------------------------------------------------------
# /hermes umbrella expansion — behaviour contracts
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hermes_runtime_expands_to_runtime(monkeypatch):
    """``/hermes runtime`` must be expanded to ``/runtime`` so it reaches
    the /runtime handler instead of returning "Unknown command"."""
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._handle_runtime_command = AsyncMock(return_value="runtime:ok")

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/hermes runtime"))

    assert result is not None
    assert "Unknown command" not in (result or "")
    # Must route to the /runtime handler, not the unknown-command guard.
    runner._handle_runtime_command.assert_awaited_once()


@pytest.mark.asyncio
async def test_hermes_runtime_native_expands(monkeypatch):
    """``/hermes runtime native`` must reach the /runtime handler with
    ``native`` as args."""
    import gateway.run as gateway_run

    runner = _make_runner()
    _seen_args = {}

    async def _capture_runtime(*args, **kwargs):
        _seen_args["value"] = kwargs
        return "runtime:native:ok"

    runner._handle_runtime_command = _capture_runtime

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/hermes runtime native"))

    assert result == "runtime:native:ok"
    assert "Unknown command" not in (result or "")


@pytest.mark.asyncio
async def test_hermes_runtime_ncoder_expands(monkeypatch):
    """``/hermes runtime ncoder`` — the original Slack-thread use case —
    must reach the /runtime handler with ``ncoder`` as args."""
    import gateway.run as gateway_run

    runner = _make_runner()
    _seen_args = {}

    async def _capture_runtime(*args, **kwargs):
        _seen_args["value"] = kwargs
        return "runtime:ncoder:ok"

    runner._handle_runtime_command = _capture_runtime

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/hermes runtime ncoder"))

    assert result == "runtime:ncoder:ok"
    assert "Unknown command" not in (result or "")


@pytest.mark.asyncio
async def test_hermes_alone_falls_back_to_help(monkeypatch):
    """``/hermes`` with no subcommand must fall back to ``/help``
    (mirroring Slack adapter behaviour)."""
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._handle_help_command = AsyncMock(return_value="help:ok")

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/hermes"))

    assert result is not None
    assert "Unknown command" not in (result or "")
    runner._handle_help_command.assert_awaited_once()


@pytest.mark.asyncio
async def test_hermes_free_form_passes_through_as_text(monkeypatch):
    """``/hermes totally-made-up-command`` with no matching subcommand
    in the map must strip the ``/hermes`` prefix and pass through as
    regular text to the agent — NOT be rejected as 'Unknown command'.
    The message_type must be updated to ``MessageType.TEXT``."""
    import gateway.run as gateway_run
    from gateway.platforms.base import MessageType

    runner = _make_runner()
    _seen_text = None
    _seen_type = None

    original = gateway_run.GatewayRunner._handle_message_with_agent

    async def _capture(self, event, source, _quick_key, run_generation):
        nonlocal _seen_text, _seen_type
        _seen_text = event.text
        _seen_type = event.message_type
        return await original(self, event, source, _quick_key, run_generation)

    monkeypatch.setattr(
        gateway_run.GatewayRunner,
        "_handle_message_with_agent",
        _capture,
    )
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(
        _make_event("/hermes totally-made-up-command")
    )

    # Must NOT return "Unknown command"
    assert result is None or "Unknown command" not in result
    # The text must be the free-form part without /hermes
    assert _seen_text == "totally-made-up-command"
    assert _seen_type == MessageType.TEXT


@pytest.mark.asyncio
async def test_hermes_free_form_chinese_treated_as_text(monkeypatch):
    """``/hermes 你有多少个 skill`` — Chinese free-form question must
    strip the ``/hermes`` prefix and reach the agent as plain text with
    ``message_type=TEXT``."""
    import gateway.run as gateway_run
    from gateway.platforms.base import MessageType

    runner = _make_runner()
    _seen_text = None
    _seen_type = None

    original = gateway_run.GatewayRunner._handle_message_with_agent

    async def _capture(self, event, source, _quick_key, run_generation):
        nonlocal _seen_text, _seen_type
        _seen_text = event.text
        _seen_type = event.message_type
        return await original(self, event, source, _quick_key, run_generation)

    monkeypatch.setattr(
        gateway_run.GatewayRunner,
        "_handle_message_with_agent",
        _capture,
    )
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(
        _make_event("/hermes 你有多少个 skill")
    )

    assert result is None or "Unknown command" not in (result or "")
    assert _seen_text == "你有多少个 skill"
    assert _seen_type == MessageType.TEXT


@pytest.mark.asyncio
async def test_hermes_compact_expands_to_compress(monkeypatch):
    """``/hermes compact`` must be expanded to ``/compress`` so it
    reaches the /compress handler."""
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._handle_compress_command = AsyncMock(return_value="compress:ok")

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/hermes compact"))

    assert result is not None
    assert "Unknown command" not in (result or "")
    runner._handle_compress_command.assert_awaited_once()


@pytest.mark.asyncio
async def test_hermes_does_not_cause_recursion(monkeypatch):
    """The umbrella expansion must NOT cause recursion: if for any reason
    the subcommand map returned a value starting with ``/hermes``, the
    dispatcher must not loop. Only expand once."""
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("recursive hermes expansion leaked to agent")
    )

    # Stub slack_subcommand_map to return a hermes-cycling value
    from hermes_cli import commands as _cmds_mod

    monkeypatch.setattr(
        _cmds_mod,
        "slack_subcommand_map",
        lambda: {"loop": "/hermes loop"},
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/hermes loop"))

    # Must NOT loop — should see "Unknown command" because after expansion
    # it becomes "/hermes loop" again, which is NOT a known command.
    assert result is not None
    assert "Unknown command" in result


@pytest.mark.asyncio
async def test_known_slash_command_not_affected_by_hermes_expansion(monkeypatch):
    """A normal ``/status`` command must NOT be affected by the hermes
    umbrella expansion in any way."""
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._running_agents = {}

    # _handle_status_command is a real Level-2 handler
    runner._handle_status_command = AsyncMock(return_value="status:ok")

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/status"))

    assert result == "status:ok"
    runner._handle_status_command.assert_awaited_once()


@pytest.mark.asyncio
async def test_hermes_whitespace_only_treated_as_empty(monkeypatch):
    """``/hermes   `` (with trailing whitespace) must behave the same
    as ``/hermes`` — fall back to ``/help``."""
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._handle_help_command = AsyncMock(return_value="help:ok")

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/hermes   "))

    assert result is not None
    assert "Unknown command" not in (result or "")
    runner._handle_help_command.assert_awaited_once()
