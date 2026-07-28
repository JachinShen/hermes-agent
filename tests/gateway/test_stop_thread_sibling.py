"""Regression tests: /stop can interrupt a sibling participant's run in a
per-user thread.

When ``thread_sessions_per_user=True``, each participant in a thread gets an
isolated session key (``...:{thread_id}:{user_id}``).  A run another user
started lives under a different key, so the caller's own ``/stop`` used to find
nothing and reply "no active task to stop".  Authorized users should be able to
stop any run in the same thread.
"""

import pytest
from unittest.mock import patch

from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL, _INTERRUPT_REASON_STOP
from gateway.session import SessionSource, build_session_key
from gateway.platforms.base import Platform, MessageEvent, MessageType


class _FakeAgent:
    def __init__(self):
        self.interrupts = []

    def interrupt(self, reason):
        self.interrupts.append(reason)


def _thread_source(uid, thread_id="thr1", chat_id="chan1"):
    return SessionSource(
        platform=Platform.DISCORD,
        chat_type="forum",
        chat_id=chat_id,
        thread_id=thread_id,
        user_id=uid,
    )


def _per_user_key(uid, thread_id="thr1", chat_id="chan1"):
    return build_session_key(
        _thread_source(uid, thread_id, chat_id),
        thread_sessions_per_user=True,
    )


# ---------------------------------------------------------------------------
# _sibling_thread_run_keys
# ---------------------------------------------------------------------------


def test_sibling_finds_other_users_run_in_same_thread():
    runner = object.__new__(GatewayRunner)
    key_a = _per_user_key("userA")
    key_b = _per_user_key("userB")
    runner._running_agents = {key_b: _FakeAgent()}
    assert runner._sibling_thread_run_keys(_thread_source("userA"), key_a) == [key_b]


def test_sibling_excludes_callers_own_key():
    runner = object.__new__(GatewayRunner)
    key_a = _per_user_key("userA")
    key_b = _per_user_key("userB")
    runner._running_agents = {key_a: _FakeAgent(), key_b: _FakeAgent()}
    assert runner._sibling_thread_run_keys(_thread_source("userA"), key_a) == [key_b]


def test_sibling_skips_pending_sentinel():
    runner = object.__new__(GatewayRunner)
    key_a = _per_user_key("userA")
    key_b = _per_user_key("userB")
    runner._running_agents = {key_b: _AGENT_PENDING_SENTINEL}
    assert runner._sibling_thread_run_keys(_thread_source("userA"), key_a) == []


def test_sibling_does_not_match_different_thread_same_chat():
    # thr1 caller must not match a run in thr11 (prefix-collision guard).
    runner = object.__new__(GatewayRunner)
    key_a = _per_user_key("userA", thread_id="thr1")
    key_b_other = _per_user_key("userB", thread_id="thr11")
    runner._running_agents = {key_b_other: _FakeAgent()}
    assert runner._sibling_thread_run_keys(_thread_source("userA"), key_a) == []


def test_sibling_returns_empty_for_non_thread_source():
    # Non-thread group/channel must NOT trigger the cross-user fallback.
    runner = object.__new__(GatewayRunner)
    nonthread = SessionSource(
        platform=Platform.DISCORD, chat_type="group", chat_id="chan1", user_id="userA"
    )
    grp_b = build_session_key(
        SessionSource(
            platform=Platform.DISCORD, chat_type="group", chat_id="chan1", user_id="userB"
        )
    )
    runner._running_agents = {grp_b: _FakeAgent()}
    assert runner._sibling_thread_run_keys(nonthread, "agent:main:discord:group:chan1:userA") == []


# ---------------------------------------------------------------------------
# _handle_stop_command fallback path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interrupt_and_clear_session_cancels_matching_north_invocation():
    runner = object.__new__(GatewayRunner)
    source = _thread_source("userA")
    key = _per_user_key("userA")
    agent = _FakeAgent()
    runner._running_agents = {key: agent}
    runner._pending_messages = {}
    runner._adapter_for_source = lambda _source: None
    runner._invalidate_session_run_generation = lambda *_args, **_kwargs: None
    runner._release_running_agent_state = lambda _key: None
    runner._evict_cached_agent = lambda _key: None

    cancelled = []

    class _NorthRuntime:
        async def cancel_session_async(self, session_key):
            cancelled.append(session_key)

    with patch(
        "gateway.north_coder_runtime.runtime_from_raw", return_value=_NorthRuntime()
    ), patch("gateway.run._load_gateway_config", return_value={}), patch(
        "gateway.run._gateway_config_home"
    ):
        await runner._interrupt_and_clear_session(
            key,
            source,
            interrupt_reason=_INTERRUPT_REASON_STOP,
            invalidation_reason="test_stop_north",
        )

    assert agent.interrupts == [_INTERRUPT_REASON_STOP]
    assert cancelled == [key]


class _StoreEntry:
    def __init__(self, session_key):
        self.session_key = session_key


class _FakeStore:
    def __init__(self, session_key):
        self._key = session_key

    def get_or_create_session(self, source):
        return _StoreEntry(self._key)


@pytest.mark.asyncio
async def test_stop_interrupts_sibling_thread_run_when_authorized(monkeypatch):
    runner = object.__new__(GatewayRunner)
    key_a = _per_user_key("userA")
    key_b = _per_user_key("userB")
    runner._running_agents = {key_b: _FakeAgent()}
    runner.session_store = _FakeStore(key_a)

    interrupted = []

    async def _fake_interrupt(session_key, source, *, interrupt_reason, invalidation_reason):
        interrupted.append((session_key, interrupt_reason, invalidation_reason))

    runner._interrupt_and_clear_session = _fake_interrupt
    runner._is_user_authorized = lambda source: True

    event = MessageEvent(
        text="/stop", message_type=MessageType.TEXT, source=_thread_source("userA")
    )
    result = await runner._handle_stop_command(event)

    assert interrupted == [(key_b, _INTERRUPT_REASON_STOP, "stop_command_thread_sibling")]
    # EphemeralReply or str — both carry the "stopped" message, not "no_active".
    assert "no active" not in str(getattr(result, "text", result)).lower()


@pytest.mark.asyncio
async def test_stop_does_not_interrupt_sibling_when_unauthorized(monkeypatch):
    runner = object.__new__(GatewayRunner)
    key_a = _per_user_key("userA")
    key_b = _per_user_key("userB")
    runner._running_agents = {key_b: _FakeAgent()}
    runner.session_store = _FakeStore(key_a)

    interrupted = []

    async def _fake_interrupt(session_key, source, *, interrupt_reason, invalidation_reason):
        interrupted.append(session_key)

    runner._interrupt_and_clear_session = _fake_interrupt
    runner._is_user_authorized = lambda source: False

    event = MessageEvent(
        text="/stop", message_type=MessageType.TEXT, source=_thread_source("userA")
    )
    result = await runner._handle_stop_command(event)

    assert interrupted == []
    assert "no active" in str(getattr(result, "text", result)).lower()


# ---------------------------------------------------------------------------
# /stop with no active agent still clears a stuck platform status (#32295)
# ---------------------------------------------------------------------------


class _FakeStatusAdapter:
    def __init__(self):
        self.cleared = []

    async def _stop_typing_with_metadata(self, chat_id, metadata=None):
        self.cleared.append((chat_id, metadata))


@pytest.mark.asyncio
async def test_stop_no_active_agent_clears_stuck_status():
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    key = _per_user_key("userA")
    runner.session_store = _FakeStore(key)
    runner._is_user_authorized = lambda source: True

    adapter = _FakeStatusAdapter()
    runner.adapters = {Platform.DISCORD: adapter}
    runner._thread_metadata_for_source = (
        lambda source, reply_to_message_id=None: {"thread_id": source.thread_id}
    )
    runner._reply_anchor_for_event = lambda event: None

    event = MessageEvent(
        text="/stop", message_type=MessageType.TEXT, source=_thread_source("userA")
    )
    result = await runner._handle_stop_command(event)

    assert "no active" in str(getattr(result, "text", result)).lower()
    assert adapter.cleared == [("chan1", {"thread_id": "thr1"})]


@pytest.mark.asyncio
async def test_stop_no_active_agent_survives_status_clear_failure():
    """A failing adapter clear must not break the /stop reply."""
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    key = _per_user_key("userA")
    runner.session_store = _FakeStore(key)
    runner._is_user_authorized = lambda source: True

    class _BoomAdapter:
        async def _stop_typing_with_metadata(self, chat_id, metadata=None):
            raise RuntimeError("boom")

    runner.adapters = {Platform.DISCORD: _BoomAdapter()}
    runner._thread_metadata_for_source = (
        lambda source, reply_to_message_id=None: None
    )
    runner._reply_anchor_for_event = lambda event: None

    event = MessageEvent(
        text="/stop", message_type=MessageType.TEXT, source=_thread_source("userA")
    )
    result = await runner._handle_stop_command(event)

    assert "no active" in str(getattr(result, "text", result)).lower()
