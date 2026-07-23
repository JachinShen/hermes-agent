"""Contract tests for North foreground background review scheduling.

These tests validate that ``schedule_north_background_review`` in
``gateway/north_coder_runtime`` correctly determines when to trigger a
background memory/skill review after a successful North turn, using the
new architecture: ``host._spawn_background_review`` directly, with
``review_host`` (Gateway) or ``tui_host_builder`` (TUI lazy factory).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest



# ── helpers ──────────────────────────────────────────────────────────────────


def _north_result(
    *,
    completed: bool = True,
    interrupted: bool = False,
    failed: bool = False,
    requires_action: bool = False,
    tools: Optional[list[dict]] = None,
) -> dict:
    """Minimal North run_turn result dict."""
    status = "completed"
    if requires_action:
        status = "requires_action"
    elif interrupted:
        status = "cancelled"
    elif failed:
        status = "failed"
    return {
        "completed": completed,
        "interrupted": interrupted,
        "failed": failed,
        "requires_action": requires_action,
        "status": status,
        "tools": tools or [],
        "final_response": "test response",
        "north_conversation_id": "north-cx-1",
        "north_invocation_id": "inv-1",
    }


def _canonical_history(users: int = 1, tools: int = 0) -> list[dict]:
    """Hermes canonical transcript with *users* user turns and *tools* call chains."""
    msgs: list[dict] = []
    for i in range(users):
        msgs.append({"role": "user", "content": f"user message {i}"})
        if i < tools:
            msgs.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": f"call_{i}", "function": {"name": "read_file", "arguments": "{}"}}
                ],
            })
            msgs.append({
                "role": "tool",
                "tool_call_id": f"call_{i}",
                "content": json.dumps({"success": True}),
            })
            msgs.append({
                "role": "assistant",
                "content": f"assistant response {i}",
            })
        else:
            msgs.append({"role": "assistant", "content": f"assistant response {i}"})
    return msgs


def _make_host_stub(**defaults: Any) -> tuple[Any, dict, Any]:
    """Return (stub, captured_dict, event) for a Hermes AIAgent stub."""
    import threading
    called: dict[str, Any] = dict(defaults)
    event = threading.Event()

    class _Stub:
        background_review_callback: Any = None
        memory_notifications: str = "on"
        _memory_enabled: bool = True
        _user_profile_enabled: bool = False
        valid_tool_names: set = {"skill_manage"}

        def _spawn_background_review(
            self,
            messages_snapshot: list[dict],
            *,
            review_memory: bool = False,
            review_skills: bool = False,
        ):
            called["called"] = True
            called["review_memory"] = review_memory
            called["review_skills"] = review_skills
            called["messages_snapshot"] = list(messages_snapshot)
            event.set()

        def close(self):
            pass

    return _Stub(), called, event


# ── Guard conditions ─────────────────────────────────────────────────────────


class TestGuardConditions:
    """schedule_north_background_review must skip non-complete states."""

    @staticmethod
    def _call(monkeypatch, *, result, host_factory=None):
        import gateway.north_coder_runtime as ncr
        ncr._NORTH_REVIEW_STATE.clear()
        stub, captured, _ = _make_host_stub()
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"memory": {"nudge_interval": 1}, "skills": {"creation_nudge_interval": 1}},
        )
        ncr.schedule_north_background_review(
            canonical_history=[{"role": "user", "content": "hi"}],
            north_result=result,
            session_id="guard-test",
            review_host=stub,
        )
        return captured

    def test_triggers_on_successful_completion(self, monkeypatch):
        captured = self._call(monkeypatch, result=_north_result())
        assert captured.get("called") is True, "completed turn should trigger review"

    def test_skips_when_not_completed(self, monkeypatch):
        captured = self._call(monkeypatch, result=_north_result(completed=False, failed=True))
        assert captured.get("called") is not True

    def test_skips_on_interrupted(self, monkeypatch):
        captured = self._call(monkeypatch, result=_north_result(interrupted=True))
        assert captured.get("called") is not True

    def test_skips_on_requires_action(self, monkeypatch):
        captured = self._call(monkeypatch, result=_north_result(requires_action=True))
        assert captured.get("called") is not True


class TestNudgeThresholds:
    """Review fires only when accumulated count >= respective threshold."""

    @staticmethod
    def _call(monkeypatch, *, history, result, mem_interval=1, skill_interval=1, **kw):
        import gateway.north_coder_runtime as ncr
        ncr._NORTH_REVIEW_STATE.clear()
        stub, captured, _ = _make_host_stub()
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"memory": {"nudge_interval": mem_interval}, "skills": {"creation_nudge_interval": skill_interval}},
        )
        ncr.schedule_north_background_review(
            canonical_history=history,
            north_result=result,
            session_id="thresh-test",
            review_host=stub,
            **kw,
        )
        return captured

    def test_memory_nudge_counts_user_turns(self, monkeypatch):
        captured = self._call(monkeypatch, history=_canonical_history(users=3, tools=0), result=_north_result(), mem_interval=3)
        assert captured.get("review_memory") is True

    def test_skills_nudge_counts_north_tool_events(self, monkeypatch):
        history = _canonical_history(users=1, tools=0)
        north_tools = [
            {"type": "tool_call_start", "toolCallName": "read_file"},
            {"type": "tool_call_result", "content": "ok"},
        ]
        result = _north_result(tools=north_tools)
        captured = self._call(monkeypatch, history=history, result=result, skill_interval=1)
        assert captured.get("review_skills") is True

    def test_review_input_is_canonical_transcript(self, monkeypatch):
        history = _canonical_history(users=2, tools=1)
        captured = self._call(monkeypatch, history=history, result=_north_result(), mem_interval=1)
        assert captured.get("called") is True
        snapshot = captured.get("messages_snapshot", [])
        assert all(m.get("role") in {"user", "assistant", "tool", "system"} for m in snapshot)
        user_msgs = [m for m in snapshot if m.get("role") == "user"]
        assert len(user_msgs) == 2

    def test_skill_review_snapshot_includes_north_tool_activity(self, monkeypatch):
        history = _canonical_history(users=1)
        tools = [
            {
                "type": "tool_call_start",
                "toolCallId": "north-1",
                "toolCallName": "read_file",
                "input": {"path": "/tmp/example"},
            },
            {
                "type": "tool_call_result",
                "toolCallId": "north-1",
                "content": "file contents",
            },
        ]
        captured = self._call(
            monkeypatch,
            history=history,
            result=_north_result(tools=tools),
            mem_interval=0,
            skill_interval=1,
        )
        snapshot = captured["messages_snapshot"]
        assert snapshot[-3]["tool_calls"][0]["function"]["name"] == "read_file"
        assert snapshot[-2] == {
            "role": "tool",
            "tool_call_id": "north-1",
            "content": "file contents",
        }
        assert snapshot[-1]["role"] == "assistant"


class TestNoReviewBelowThreshold:
    """Review does not fire when accumulated count < threshold."""

    @staticmethod
    def _call(monkeypatch, *, history, result, mem_interval=3, skill_interval=5):
        import gateway.north_coder_runtime as ncr
        ncr._NORTH_REVIEW_STATE.clear()
        stub, captured, _ = _make_host_stub()
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"memory": {"nudge_interval": mem_interval}, "skills": {"creation_nudge_interval": skill_interval}},
        )
        ncr.schedule_north_background_review(
            canonical_history=history,
            north_result=result,
            session_id="below-thresh",
            review_host=stub,
        )
        return captured

    def test_below_threshold_does_not_review(self, monkeypatch):
        history = _canonical_history(users=1, tools=0)
        result = _north_result(tools=[])
        captured = self._call(monkeypatch, history=history, result=result, mem_interval=3, skill_interval=5)
        assert captured.get("called") is not True

    def test_memory_review_from_canonical_history(self, monkeypatch):
        history = _canonical_history(users=6, tools=2)
        result = _north_result(tools=[])
        captured = self._call(monkeypatch, history=history, result=result, mem_interval=3, skill_interval=3)
        assert captured.get("review_memory") is True
        assert captured.get("review_skills") is not True

    def test_skills_from_north_tool_events(self, monkeypatch):
        history = _canonical_history(users=1, tools=0)
        north_tools = [
            {"type": "tool_call_start", "toolCallName": "read_file"},
            {"type": "tool_call_start", "toolCallName": "write_file"},
        ]
        result = _north_result(tools=north_tools)
        captured = self._call(monkeypatch, history=history, result=result, mem_interval=5, skill_interval=2)
        assert captured.get("review_skills") is True


class TestDiffBasedAccumulation:
    """Nudge counters accumulate via diff (delta from last_seen count)."""

    def test_state_accumulates_across_turns(self, monkeypatch):
        import gateway.north_coder_runtime as ncr
        ncr._NORTH_REVIEW_STATE.clear()
        stub, captured, event = _make_host_stub()

        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"memory": {"nudge_interval": 10}, "skills": {"creation_nudge_interval": 10}},
        )

        # Turn 1: 1 user turn -> accumulator == 1, below threshold
        ncr.schedule_north_background_review(
            canonical_history=_canonical_history(users=1, tools=0),
            north_result=_north_result(tools=[]),
            session_id="diff-accum",
            review_host=stub,
        )
        assert captured.get("called") is not True

        # Turn 2: 3 user turns -> delta == 2, accumulator == 3
        ncr.schedule_north_background_review(
            canonical_history=_canonical_history(users=3, tools=0),
            north_result=_north_result(tools=[]),
            session_id="diff-accum",
            review_host=stub,
        )
        assert captured.get("called") is not True

        # Turn 3: 12 user turns -> delta == 9, accumulator == 12 >= 10
        ncr.schedule_north_background_review(
            canonical_history=_canonical_history(users=12, tools=0),
            north_result=_north_result(tools=[]),
            session_id="diff-accum",
            review_host=stub,
        )
        assert captured.get("called") is True

        # After review, accum counters are reset
        state = ncr._NORTH_REVIEW_STATE.get("diff-accum", {})
        assert state.get("accum_user_turns", 999) == 0
        assert state.get("accum_tool_iters", 999) == 0

    def test_state_is_per_session(self, monkeypatch):
        import gateway.north_coder_runtime as ncr
        ncr._NORTH_REVIEW_STATE.clear()
        stub, _, _ = _make_host_stub()
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})

        # Call twice so state entries exist
        ncr.schedule_north_background_review(
            canonical_history=[{"role": "user", "content": "a"}],
            north_result=_north_result(),
            session_id="sess-a",
            review_host=stub,
        )
        ncr.schedule_north_background_review(
            canonical_history=[{"role": "user", "content": "b"}],
            north_result=_north_result(),
            session_id="sess-b",
            review_host=stub,
        )
        assert "sess-a" in ncr._NORTH_REVIEW_STATE
        assert "sess-b" in ncr._NORTH_REVIEW_STATE

    def test_memory_review_does_not_reset_skill_progress(self, monkeypatch):
        import gateway.north_coder_runtime as ncr

        ncr._NORTH_REVIEW_STATE.clear()
        stub, captured, _ = _make_host_stub()
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {
                "memory": {"nudge_interval": 1},
                "skills": {"creation_nudge_interval": 3},
            },
        )
        two_tools = [
            {"type": "tool_call_start", "toolCallId": "a"},
            {"type": "tool_call_start", "toolCallId": "b"},
        ]
        ncr.schedule_north_background_review(
            canonical_history=_canonical_history(users=1),
            north_result=_north_result(tools=two_tools),
            session_id="independent-counters",
            review_host=stub,
        )
        state = ncr._NORTH_REVIEW_STATE["independent-counters"]
        assert captured["review_memory"] is True
        assert captured["review_skills"] is False
        assert state["accum_tool_iters"] == 2

        ncr.schedule_north_background_review(
            canonical_history=_canonical_history(users=2),
            north_result=_north_result(
                tools=[{"type": "tool_call_start", "toolCallId": "c"}],
            ),
            session_id="independent-counters",
            review_host=stub,
        )
        assert captured["review_skills"] is True


class TestReviewHostPathway:
    """review_host (Gateway) pathway - host._spawn_background_review called directly."""

    def test_review_host_called_directly(self, monkeypatch):
        import gateway.north_coder_runtime as ncr
        ncr._NORTH_REVIEW_STATE.clear()
        stub, captured, _ = _make_host_stub()
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"memory": {"nudge_interval": 1}, "skills": {"creation_nudge_interval": 1}},
        )
        ncr.schedule_north_background_review(
            canonical_history=[{"role": "user", "content": "hi"}],
            north_result=_north_result(),
            session_id="host-test",
            review_host=stub,
        )
        assert captured.get("called") is True

    def test_review_host_callback_wired(self, monkeypatch):
        import gateway.north_coder_runtime as ncr
        ncr._NORTH_REVIEW_STATE.clear()
        stub, _, _ = _make_host_stub()
        callback = lambda msg: None
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"memory": {"nudge_interval": 1}, "skills": {"creation_nudge_interval": 1}},
        )
        ncr.schedule_north_background_review(
            canonical_history=[{"role": "user", "content": "hi"}],
            north_result=_north_result(),
            session_id="cb-test",
            review_host=stub,
            background_review_callback=callback,
        )
        assert stub.background_review_callback is callback


class TestTUILazyFactoryPathway:
    """tui_host_builder (TUI) pathway - host built on demand when thresholds met."""

    def test_lazy_builder_invoked_on_threshold_met(self, monkeypatch):
        import gateway.north_coder_runtime as ncr
        ncr._NORTH_REVIEW_STATE.clear()
        stub, captured, _ = _make_host_stub()
        built = [False]

        def builder():
            built[0] = True
            return stub

        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"memory": {"nudge_interval": 1}, "skills": {"creation_nudge_interval": 1}},
        )
        ncr.schedule_north_background_review(
            canonical_history=[{"role": "user", "content": "hi"}],
            north_result=_north_result(),
            session_id="tui-lazy-test",
            tui_host_builder=builder,
        )
        assert built[0] is True
        assert captured.get("called") is True

    def test_lazy_builder_not_invoked_below_threshold(self, monkeypatch):
        import gateway.north_coder_runtime as ncr
        ncr._NORTH_REVIEW_STATE.clear()
        built = [False]

        def builder():
            built[0] = True
            return None

        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"memory": {"nudge_interval": 999}, "skills": {"creation_nudge_interval": 999}},
        )
        ncr.schedule_north_background_review(
            canonical_history=[{"role": "user", "content": "hi"}],
            north_result=_north_result(),
            session_id="tui-no-build",
            tui_host_builder=builder,
        )
        assert built[0] is False, "builder must NOT be called below threshold"


class TestNoAvailableHost:
    """When no review_host and no tui_host_builder, review is skipped silently."""

    def test_logs_warning_when_no_host(self, monkeypatch, caplog):
        import gateway.north_coder_runtime as ncr
        ncr._NORTH_REVIEW_STATE.clear()
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"memory": {"nudge_interval": 1}, "skills": {"creation_nudge_interval": 1}},
        )
        import logging
        caplog.set_level(logging.WARNING)
        ncr.schedule_north_background_review(
            canonical_history=[{"role": "user", "content": "hi"}],
            north_result=_north_result(),
            session_id="no-host",
            review_host=None,
        )
        assert len(caplog.records) >= 1
        assert "no review host available" in caplog.text


# ── NorthCoderTUIAgent ──────────────────────────────────────────────────────


class TestTUIAgentConstructor:
    """NorthCoderTUIAgent.__init__ with new signature."""

    def test_accepts_review_host_factory(self):
        import gateway.north_coder_runtime as ncr
        from pathlib import Path
        config = ncr.NorthCoderRuntimeConfig()
        runtime = ncr.NorthCoderRuntime(config, Path("/tmp"))
        factory = lambda: None
        agent = ncr.NorthCoderTUIAgent(runtime, "sess", _review_host_factory=factory)
        assert agent._review_host_factory is factory

    def test_defaults_to_none_factory(self):
        import gateway.north_coder_runtime as ncr
        from pathlib import Path
        config = ncr.NorthCoderRuntimeConfig()
        runtime = ncr.NorthCoderRuntime(config, Path("/tmp"))
        agent = ncr.NorthCoderTUIAgent(runtime, "sess")
        assert agent._review_host_factory is None


class TestTUIAgentScheduleBackgroundReview:
    """delegates to the global schedule_north_background_review."""

    def test_delegates_to_global_scheduler(self, monkeypatch):
        import gateway.north_coder_runtime as ncr
        from pathlib import Path
        captured: dict[str, Any] = {}

        def fake_sched(*, canonical_history, north_result, session_id, **kw):
            captured["called"] = True
            captured["session_id"] = session_id
            captured["tui_host_builder"] = kw.get("tui_host_builder")

        monkeypatch.setattr(ncr, "schedule_north_background_review", fake_sched)
        config = ncr.NorthCoderRuntimeConfig()
        runtime = ncr.NorthCoderRuntime(config, Path("/tmp"))
        factory = lambda: None
        agent = ncr.NorthCoderTUIAgent(runtime, "tui-session", _review_host_factory=factory)
        history = [{"role": "user", "content": "hello"}]
        result = _north_result()
        agent.schedule_background_review(history, result)
        assert captured.get("called") is True
        assert captured["session_id"] == "tui-session"
        assert callable(captured["tui_host_builder"])


class TestTUIAgentRunConversationIntegration:
    """run_conversation must call schedule_background_review after success."""

    def _make_runtime_and_mocks(self, monkeypatch, *, completed=True, tools=None):
        import gateway.north_coder_runtime as ncr
        from pathlib import Path

        scheduled: dict[str, Any] = {}

        def fake_sched(*, canonical_history, north_result, session_id, **kw):
            scheduled["called"] = True
            scheduled["history"] = list(canonical_history)
            scheduled["north_result"] = north_result
            scheduled["session_id"] = session_id

        monkeypatch.setattr(ncr, "schedule_north_background_review", fake_sched)

        config = ncr.NorthCoderRuntimeConfig(base_url="http://localhost:0", timeout_seconds=1)
        runtime = ncr.NorthCoderRuntime(config, Path("/tmp"))
        agent = ncr.NorthCoderTUIAgent(runtime, "test-tui-session")

        async def fake_run_turn(*, message, session_key, hermes_session_id,
                                conversation_history=None, **kwargs):
            return {
                "completed": completed,
                "final_response": "response",
                "messages": [{"role": "user", "content": message},
                             {"role": "assistant", "content": "response"}],
                "tools": tools or [],
                "status": "completed" if completed else "failed",
                "north_conversation_id": "north-cx",
                "north_invocation_id": "inv-1",
            }

        monkeypatch.setattr(runtime, "run_turn", fake_run_turn)
        return agent, scheduled

    def test_schedules_review_after_successful_turn(self, monkeypatch):
        agent, scheduled = self._make_runtime_and_mocks(monkeypatch)
        agent.run_conversation(
            "hello",
            conversation_history=[{"role": "user", "content": "previous"}],
        )
        assert scheduled.get("called") is True
        assert scheduled["session_id"] == "test-tui-session"

    def test_does_not_schedule_when_not_completed(self, monkeypatch):
        agent, scheduled = self._make_runtime_and_mocks(monkeypatch, completed=False)
        agent.run_conversation("hello")
        assert scheduled.get("called") is not True

    def test_canonical_history_includes_exchange(self, monkeypatch):
        agent, scheduled = self._make_runtime_and_mocks(monkeypatch)
        prior = [{"role": "user", "content": "earlier question"},
                 {"role": "assistant", "content": "earlier answer"}]
        agent.run_conversation("new question", conversation_history=list(prior))
        assert scheduled.get("called") is True
        history = scheduled.get("history", [])
        contents = [m.get("content") for m in history if m.get("role") == "user"]
        assert "earlier question" in contents
        assert "new question" in contents

    def test_tool_iterations_in_north_result(self, monkeypatch):
        tools = [
            {"type": "tool_call_start", "toolCallName": "read_file"},
            {"type": "tool_call_result", "content": "ok"},
        ]
        agent, scheduled = self._make_runtime_and_mocks(monkeypatch, tools=tools)
        agent.run_conversation("hi")
        assert scheduled.get("called") is True
        result_tools = scheduled["north_result"].get("tools", [])
        assert len(result_tools) == 2


# ── Smoke / importability ───────────────────────────────────────────────────


class TestSmoke:
    """Lightweight smoke tests."""

    def test_function_is_importable(self):
        from gateway.north_coder_runtime import schedule_north_background_review
        assert callable(schedule_north_background_review)

    def test_no_crash_on_all_status_states(self):
        from gateway.north_coder_runtime import schedule_north_background_review
        import gateway.north_coder_runtime as ncr
        ncr._NORTH_REVIEW_STATE.clear()
        for completed, interrupted, failed, requires_action in [
            (True, False, False, False),
            (False, False, True, False),
            (False, True, False, False),
            (False, False, False, True),
        ]:
            schedule_north_background_review(
                canonical_history=[{"role": "user", "content": "test"}],
                north_result=_north_result(
                    completed=completed, interrupted=interrupted,
                    failed=failed, requires_action=requires_action,
                ),
                session_id="smoke-test",
            )

    def test_tui_agent_constructible(self):
        from gateway.north_coder_runtime import (
            NorthCoderTUIAgent,
            NorthCoderRuntimeConfig,
            NorthCoderRuntime,
        )
        from pathlib import Path
        config = NorthCoderRuntimeConfig()
        runtime = NorthCoderRuntime(config, Path("/tmp"))
        agent = NorthCoderTUIAgent(runtime, "smoke-tui")
        assert agent.session_key == "smoke-tui"
        assert agent._review_host_factory is None


# ── C) Config 0 disables ────────────────────────────────────────────────────


class TestConfigZeroDisables:
    """memory_nudge <= 0 or skill_nudge <= 0 must disable that axis."""

    @staticmethod
    def _call(monkeypatch, *, history, result, mem_interval=1, skill_interval=1, **kw):
        import gateway.north_coder_runtime as ncr
        ncr._NORTH_REVIEW_STATE.clear()
        stub, captured, _ = _make_host_stub()
        stub._memory_enabled = True
        stub.valid_tool_names = {"skill_manage"}
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"memory": {"nudge_interval": mem_interval}, "skills": {"creation_nudge_interval": skill_interval}},
        )
        ncr.schedule_north_background_review(
            canonical_history=history,
            north_result=result,
            session_id="zero-test",
            review_host=stub,
            **kw,
        )
        return captured

    def test_memory_nudge_zero_disables(self, monkeypatch):
        """memory_nudge=0 must prevent memory review regardless of user turns."""
        history = _canonical_history(users=50, tools=0)
        north_tools = [{"type": "tool_call_start", "toolCallName": "read_file"} for _ in range(5)]
        result = _north_result(tools=north_tools)
        captured = self._call(monkeypatch, history=history, result=result, mem_interval=0, skill_interval=1)
        assert captured.get("called") is True
        assert captured.get("review_memory") is not True
        assert captured.get("review_skills") is True

    def test_skill_nudge_zero_disables(self, monkeypatch):
        """skill_nudge=0 must prevent skill review regardless of tool events."""
        history = _canonical_history(users=1, tools=0)
        north_tools = [{"type": "tool_call_start", "toolCallName": "read_file"} for _ in range(50)]
        result = _north_result(tools=north_tools)
        captured = self._call(monkeypatch, history=history, result=result, mem_interval=1, skill_interval=0)
        assert captured.get("called") is True
        assert captured.get("review_memory") is True
        assert captured.get("review_skills") is not True

    def test_both_zero_disables_both(self, monkeypatch):
        """Both at 0 means no review at all."""
        history = _canonical_history(users=50, tools=0)
        north_tools = [{"type": "tool_call_start", "toolCallName": "read_file"} for _ in range(50)]
        result = _north_result(tools=north_tools)
        captured = self._call(monkeypatch, history=history, result=result, mem_interval=0, skill_interval=0)
        assert captured.get("called") is not True

    def test_both_zero_with_host_still_skips(self, monkeypatch, caplog):
        """Both at 0 logs no warning — we return before host check."""
        import gateway.north_coder_runtime as ncr
        import logging
        ncr._NORTH_REVIEW_STATE.clear()
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"memory": {"nudge_interval": 0}, "skills": {"creation_nudge_interval": 0}},
        )
        caplog.set_level(logging.WARNING)
        ncr.schedule_north_background_review(
            canonical_history=[{"role": "user", "content": "hi"}],
            north_result=_north_result(),
            session_id="both-zero",
            review_host=None,
        )
        assert any("no review host" not in r.getMessage() for r in caplog.records) or len(caplog.records) == 0


# ── D) Modulo baseline ──────────────────────────────────────────────────────


class TestModuloBaseline:
    """First state entry uses modulo to avoid immediate trigger on resume."""

    @staticmethod
    def _call(monkeypatch, *, history, result, mem_interval=20, skill_interval=999):
        import gateway.north_coder_runtime as ncr
        ncr._NORTH_REVIEW_STATE.clear()
        stub, captured, _ = _make_host_stub()
        stub._memory_enabled = True
        stub.valid_tool_names = {"skill_manage"}
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"memory": {"nudge_interval": mem_interval}, "skills": {"creation_nudge_interval": skill_interval}},
        )
        ncr.schedule_north_background_review(
            canonical_history=history,
            north_result=result,
            session_id="mod-test",
            review_host=stub,
        )
        return captured, ncr._NORTH_REVIEW_STATE

    def test_total_21_interval_20_not_triggers(self, monkeypatch):
        """Fresh session with 21 user turns, interval=20 → NOT triggered."""
        captured, state = self._call(
            monkeypatch,
            history=_canonical_history(users=21, tools=0),
            result=_north_result(tools=[]),
            mem_interval=20,
        )
        assert captured.get("called") is not True
        # accumulator should be 1 (21 % 20)
        assert state.get("mod-test", {}).get("accum_user_turns", -1) == 1

    def test_total_20_interval_20_triggers(self, monkeypatch):
        """Fresh session with exactly 20 user turns, interval=20 → triggers."""
        captured, state = self._call(
            monkeypatch,
            history=_canonical_history(users=20, tools=0),
            result=_north_result(tools=[]),
            mem_interval=20,
        )
        assert captured.get("called") is True
        assert captured.get("review_memory") is True
        # after trigger, accum resets to 0
        assert state.get("mod-test", {}).get("accum_user_turns", -1) == 0

    def test_delta_after_modulo_triggers_at_next_boundary(self, monkeypatch):
        """After modulo-trigger, subsequent delta accumulates to next boundary."""
        import gateway.north_coder_runtime as ncr
        ncr._NORTH_REVIEW_STATE.clear()
        stub, captured, _ = _make_host_stub()
        stub._memory_enabled = True
        stub.valid_tool_names = {"skill_manage"}
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"memory": {"nudge_interval": 20}, "skills": {"creation_nudge_interval": 999}},
        )

        # First call: 20 users → triggers (modulo = 0)
        ncr.schedule_north_background_review(
            canonical_history=_canonical_history(users=20, tools=0),
            north_result=_north_result(tools=[]),
            session_id="mod-delta",
            review_host=stub,
        )
        assert captured.get("called") is True
        assert captured.get("review_memory") is True
        captured.clear()

        # Second call: now at 40 users → delta = 40 - 20 = 20, triggers
        ncr.schedule_north_background_review(
            canonical_history=_canonical_history(users=40, tools=0),
            north_result=_north_result(tools=[]),
            session_id="mod-delta",
            review_host=stub,
        )
        assert captured.get("called") is True
        assert captured.get("review_memory") is True
        s = ncr._NORTH_REVIEW_STATE.get("mod-delta", {})
        assert s.get("accum_user_turns", -1) == 0


# ── D) Native-style gating ──────────────────────────────────────────────────


class TestNativeGating:
    """Review must respect host-level memory/skill gates."""

    @staticmethod
    def _call(monkeypatch, *, host_kwargs=None, mem_interval=1, skill_interval=1):
        import gateway.north_coder_runtime as ncr
        ncr._NORTH_REVIEW_STATE.clear()
        stub, captured, _ = _make_host_stub(**(host_kwargs or {}))
        for k, v in (host_kwargs or {}).items():
            setattr(stub, k, v)
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"memory": {"nudge_interval": mem_interval}, "skills": {"creation_nudge_interval": skill_interval}},
        )
        ncr.schedule_north_background_review(
            canonical_history=[{"role": "user", "content": "hi"}],
            north_result=_north_result(tools=[{"type": "tool_call_start", "toolCallName": "x"}]),
            session_id="gate-test",
            review_host=stub,
        )
        return captured

    def test_memory_gated_when_both_disabled(self, monkeypatch):
        captured = self._call(monkeypatch, host_kwargs={"_memory_enabled": False, "_user_profile_enabled": False})
        assert captured.get("called") is True, "skills may still trigger"
        assert captured.get("review_memory") is not True

    def test_memory_allowed_when_memory_enabled(self, monkeypatch):
        captured = self._call(monkeypatch, host_kwargs={"_memory_enabled": True})
        assert captured.get("review_memory") is True

    def test_memory_allowed_when_user_profile_enabled(self, monkeypatch):
        captured = self._call(monkeypatch, host_kwargs={"_memory_enabled": False, "_user_profile_enabled": True})
        assert captured.get("review_memory") is True

    def test_skills_gated_when_skill_manage_missing(self, monkeypatch):
        captured = self._call(monkeypatch, host_kwargs={"_memory_enabled": True, "valid_tool_names": {"other_tool"}})
        assert captured.get("called") is True
        assert captured.get("review_skills") is not True

    def test_skills_allowed_when_skill_manage_present(self, monkeypatch):
        captured = self._call(monkeypatch, host_kwargs={"_memory_enabled": True, "valid_tool_names": {"skill_manage", "other"}})
        assert captured.get("review_skills") is True

    def test_both_gated_no_spawn(self, monkeypatch):
        captured = self._call(monkeypatch, host_kwargs={
            "_memory_enabled": False, "_user_profile_enabled": False,
            "valid_tool_names": {"other_tool"},
        })
        assert captured.get("called") is not True


# ── B) Host lifecycle: no close in scheduler, caching in TUIAgent ───────────


class TestHostLifecycle:
    """Scheduler must NOT close host; TUIAgent caches and owns close."""

    def test_scheduler_does_not_close_review_host(self, monkeypatch):
        """review_host passed directly must NOT be closed by scheduler."""
        import gateway.north_coder_runtime as ncr
        ncr._NORTH_REVIEW_STATE.clear()
        closed = [False]

        class _NonClosingStub:
            background_review_callback = None
            memory_notifications = "on"
            _memory_enabled = True
            valid_tool_names = {"skill_manage", "memory"}

            def _spawn_background_review(self, **kw):
                pass

            def close(self):
                closed[0] = True

        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"memory": {"nudge_interval": 1}, "skills": {"creation_nudge_interval": 1}},
        )
        ncr.schedule_north_background_review(
            canonical_history=[{"role": "user", "content": "hi"}],
            north_result=_north_result(),
            session_id="no-close-test",
            review_host=_NonClosingStub(),
        )
        assert closed[0] is False, "scheduler must NOT close review_host"

    def test_scheduler_does_not_close_tui_built_host(self, monkeypatch):
        """tui_host_builder-created host must NOT be closed by scheduler."""
        import gateway.north_coder_runtime as ncr
        ncr._NORTH_REVIEW_STATE.clear()
        closed = [False]

        class _TuiStub:
            background_review_callback = None
            memory_notifications = "on"
            _memory_enabled = True
            valid_tool_names = {"skill_manage", "memory"}

            def _spawn_background_review(self, **kw):
                pass

            def close(self):
                closed[0] = True

        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"memory": {"nudge_interval": 1}, "skills": {"creation_nudge_interval": 1}},
        )
        ncr.schedule_north_background_review(
            canonical_history=[{"role": "user", "content": "hi"}],
            north_result=_north_result(),
            session_id="no-close-tui",
            tui_host_builder=lambda: _TuiStub(),
        )
        assert closed[0] is False, "scheduler must NOT close tui-built host"

    def test_tui_agent_caches_host_and_closes_on_close(self, monkeypatch):
        """TUIAgent caches review host; close() cleans it up."""
        import gateway.north_coder_runtime as ncr
        from pathlib import Path
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"memory": {"nudge_interval": 1}, "skills": {"creation_nudge_interval": 1}},
        )
        config = ncr.NorthCoderRuntimeConfig()
        runtime = ncr.NorthCoderRuntime(config, Path("/tmp"))
        closed = [False]

        class _CachedStub:
            background_review_callback = None
            memory_notifications = "on"
            _memory_enabled = True
            valid_tool_names = {"skill_manage", "memory"}

            def _spawn_background_review(self, **kw):
                pass

            def close(self):
                closed[0] = True

        factory = lambda: _CachedStub()
        agent = ncr.NorthCoderTUIAgent(runtime, "cache-test", _review_host_factory=factory)

        # First call: factory invoked, host cached
        agent.schedule_background_review(
            [{"role": "user", "content": "hi"}],
            _north_result(),
        )
        assert agent._review_host is not None
        cached_host = agent._review_host

        # Second call: same cached host reused
        agent.schedule_background_review(
            [{"role": "user", "content": "bye"}],
            _north_result(),
        )
        assert agent._review_host is cached_host

        # close() cleans up cached host
        agent.close()
        assert closed[0] is True
        assert agent._review_host is None


# ── A) TUI _make_agent factory test ─────────────────────────────────────────


class TestMakeAgentReviewHostFactory:
    """_make_agent (TUI server) must wire a valid _review_host_factory
    on the North agent, and the factory must produce hosts with expected
    lifecycle attributes."""

    def test_make_agent_wires_lazy_native_review_host(self, monkeypatch):
        """The live TUI construction seam passes a lazy native-host factory."""
        import gateway.north_coder_runtime as ncr
        import tui_gateway.server as server

        captured: dict[str, Any] = {}

        class _NorthAgent:
            runtime_override = None

        def fake_tui_agent_from_raw(raw, hermes_home, session_key, _review_host_factory=None):
            captured["factory"] = _review_host_factory
            captured["session_key"] = session_key
            return _NorthAgent()

        monkeypatch.setattr(server, "_load_cfg", lambda: {"agent_runtime": {"type": "ncoder"}})
        monkeypatch.setattr(ncr, "tui_agent_from_raw", fake_tui_agent_from_raw)

        original_make_agent = server._make_agent
        agent = original_make_agent("sid", "key", session_id="canonical-session")
        assert agent.runtime_override == "ncoder"
        assert captured["session_key"] == "canonical-session"
        assert callable(captured["factory"])

        class _NativeHost:
            pass

        native_host = _NativeHost()
        native_call: dict[str, Any] = {}

        def fake_native_make_agent(*args, **kwargs):
            native_call.update(kwargs)
            return native_host

        monkeypatch.setattr(server, "_make_agent", fake_native_make_agent)
        produced = captured["factory"]()
        assert produced is native_host
        assert native_call["runtime_override"] == "native"
        assert produced._persist_disabled is True
        assert produced._end_session_on_close is False


def test_tui_agent_returns_full_canonical_history(monkeypatch):
    """A completed North turn must not replace prior TUI history with one pair."""
    import gateway.north_coder_runtime as ncr

    runtime = ncr.NorthCoderRuntime(
        ncr.NorthCoderRuntimeConfig(base_url="http://localhost:0"),
        Path("/tmp"),
    )
    agent = ncr.NorthCoderTUIAgent(runtime, "canonical-history")

    async def fake_run_turn(**kwargs):
        return {
            "completed": True,
            "interrupted": False,
            "requires_action": False,
            "messages": [
                {"role": "user", "content": "current question"},
                {"role": "assistant", "content": "current answer"},
            ],
            "tools": [],
            "final_response": "current answer",
        }

    monkeypatch.setattr(runtime, "run_turn", fake_run_turn)
    monkeypatch.setattr(agent, "schedule_background_review", lambda *args, **kwargs: None)
    prior = [
        {"role": "user", "content": "prior question"},
        {"role": "assistant", "content": "prior answer"},
    ]

    result = agent.run_conversation("current question", conversation_history=prior)

    assert result["messages"] == prior + [
        {"role": "user", "content": "current question"},
        {"role": "assistant", "content": "current answer"},
    ]


def test_gateway_hook_composes_canonical_history_and_wires_real_host(monkeypatch):
    import gateway.north_coder_runtime as ncr
    from gateway.run import _schedule_north_review_after_turn

    captured = {}
    host = object()
    callback = object()

    def fake_schedule(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(ncr, "schedule_north_background_review", fake_schedule)
    prior = [
        {"role": "user", "content": "prior"},
        {"role": "assistant", "content": "prior answer"},
    ]
    current = [
        {"role": "user", "content": "current"},
        {"role": "assistant", "content": "current answer"},
    ]
    result = _north_result()
    result["messages"] = current

    _schedule_north_review_after_turn(
        agent_history=prior,
        result=result,
        session_id="gateway-session",
        session_key="gateway-key",
        review_host=host,
        background_review_callback=callback,
        memory_notifications="verbose",
    )

    assert captured["canonical_history"] == prior + current
    assert [m["content"] for m in captured["canonical_history"]].count("current") == 1
    assert captured["review_host"] is host
    assert captured["background_review_callback"] is callback
    assert captured["memory_notifications"] == "verbose"
    assert captured["session_id"] == "gateway-session"


def test_scheduler_reaches_real_aia_agent_spawn_seam(monkeypatch):
    import agent.background_review as background_review
    import gateway.north_coder_runtime as ncr
    import run_agent
    from run_agent import AIAgent

    ncr._NORTH_REVIEW_STATE.clear()
    captured = {}

    def fake_spawn(parent_agent, messages_snapshot, **kwargs):
        captured["parent_agent"] = parent_agent
        captured["messages_snapshot"] = messages_snapshot
        captured.update(kwargs)
        return (lambda: None), "review prompt"

    class FakeThread:
        def __init__(self, **kwargs):
            captured["thread_kwargs"] = kwargs

        def start(self):
            captured["thread_started"] = True

    class RealSeamHost:
        _spawn_background_review = AIAgent._spawn_background_review
        valid_tool_names = {"memory", "skill_manage"}
        _memory_enabled = True
        _user_profile_enabled = True
        background_review_callback = None
        memory_notifications = "on"

    host = RealSeamHost()
    monkeypatch.setattr(background_review, "spawn_background_review_thread", fake_spawn)
    monkeypatch.setattr(run_agent.threading, "Thread", FakeThread)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "memory": {"nudge_interval": 1},
            "skills": {"creation_nudge_interval": 0},
        },
    )

    ncr.schedule_north_background_review(
        canonical_history=_canonical_history(users=1),
        north_result=_north_result(),
        session_id="real-spawn-seam",
        review_host=host,
    )

    assert captured["parent_agent"] is host
    assert captured["review_memory"] is True
    assert captured["review_skills"] is False
    assert captured["thread_started"] is True
    assert captured["thread_kwargs"]["daemon"] is True
    assert captured["thread_kwargs"]["name"] == "bg-review"


def test_same_session_counter_is_safe_under_concurrent_duplicate_scheduling(monkeypatch):
    import threading

    import gateway.north_coder_runtime as ncr

    ncr._NORTH_REVIEW_STATE.clear()
    stub, _, _ = _make_host_stub()
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "memory": {"nudge_interval": 100},
            "skills": {"creation_nudge_interval": 100},
        },
    )
    barrier = threading.Barrier(8)

    def schedule_once():
        barrier.wait()
        ncr.schedule_north_background_review(
            canonical_history=_canonical_history(users=1),
            north_result=_north_result(tools=[]),
            session_id="concurrent-session",
            review_host=stub,
        )

    threads = [threading.Thread(target=schedule_once) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    state = ncr._NORTH_REVIEW_STATE["concurrent-session"]
    assert state["last_seen_user_count"] == 1
    assert state["accum_user_turns"] == 1
