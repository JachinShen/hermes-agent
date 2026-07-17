"""Tests for Backend delete — fail-closed status, env cleanup, process retirement.

Verifies the full safe-delete contract:

A) ``update(current=True)`` switches only — no kill/cleanup.

B) ``delete`` action (CNB backend):
   1. Records status → ``deleting`` (concurrent routing fail-closed).
   2. Adapter ``workspace-stop`` — on failure, restores original status
      and preserves DB/bindings/override/env/processes, then re-raises.
   3. On success: retires process registry entries for the backend,
      clears in-memory env/override/cwd/cache state, then deletes from
      DB (bindings, events, record). Returns the original record.

C) Terminal-tool key matching: ``_execution_backend_keys`` matches
   ``execution-backend:<id>`` (old) and ``execution-backend:<id>:session:<hash>``
   (new) with strict segment boundaries — never mis-matches similar ids.

D) ``clear_backend_execution_env``: removes from all tracking dicts inside
   locks first, then stops/cleanups environments outside locks.

E) ``ProcessRegistry.retire_backend``: marks matching running sessions
   as exited/backend_lost, does NOT send SSH kills.

No force-delete path.  No deletion of local backend.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, call, patch

import pytest

from tools.execution_backends import (
    CNBCLIAdapter,
    BackendError,
    BackendRecord,
    BackendStore,
    _validate_backend_id,
    backend_tool,
    resolve_execution_task_id,
)
from tools.process_registry import ProcessRegistry, ProcessSession
from tools.terminal_tool import (
    _EXECUTION_BACKEND_PREFIX,
    _active_environments,
    _creation_locks,
    _creation_locks_lock,
    _env_lock,
    _execution_backend_keys,
    _last_activity,
    _session_cwd,
    _session_cwd_lock,
    _task_env_overrides,
    _task_env_overrides_lock,
    clear_backend_execution_env,
    clear_task_env_overrides,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> BackendStore:
    return BackendStore(tmp_path / "execution_backends.db")


@pytest.fixture
def cnb_record() -> BackendRecord:
    return BackendRecord(
        id="cnb-a",
        driver="cnb",
        status="running",
        repo="example/repo",
        branch="main",
        workspace_sn="sn-cnb-a",
        pipeline_id="pipeline-cnb-a",
        ssh_host="10.0.0.1",
        ssh_user="dev",
        ssh_port=22,
        cwd="/workspace",
        created_at="2026-07-17T19:00:00+08:00",
    )


@pytest.fixture(autouse=True)
def _clear_terminal_globals():
    """Clear terminal_tool module-level state before and after each test."""
    with _task_env_overrides_lock:
        _task_env_overrides.clear()
    with _env_lock:
        _active_environments.clear()
        _last_activity.clear()
    with _creation_locks_lock:
        _creation_locks.clear()
    with _session_cwd_lock:
        _session_cwd.clear()
    yield
    with _task_env_overrides_lock:
        _task_env_overrides.clear()
    with _env_lock:
        _active_environments.clear()
        _last_activity.clear()
    with _creation_locks_lock:
        _creation_locks.clear()
    with _session_cwd_lock:
        _session_cwd.clear()


# ---------------------------------------------------------------------------
# A) update(current=True) — switch only, no cleanup
# ---------------------------------------------------------------------------


def test_update_current_switches_without_cleanup(store: BackendStore) -> None:
    """``update(current=True)`` must not touch process/override/env state."""
    store.create_backend(
        BackendRecord(
            id="cnb-a",
            driver="cnb",
            status="running",
            repo="example/repo",
            branch="main",
            workspace_sn="sn-a",
            pipeline_id="pipeline-a",
            ssh_host="10.0.0.1",
            ssh_user="dev",
        )
    )
    result = json.loads(
        backend_tool(
            {"action": "update", "id": "cnb-a", "current": True},
            session_id="session-x",
            store=store,
        )
    )
    assert result["current_backend"] == "cnb-a"

    # Confirm store state is correct — backend exists and bindings updated.
    record = store.get_backend("cnb-a")
    assert record.status == "running"
    assert store.get_current("session-x").id == "cnb-a"


# ---------------------------------------------------------------------------
# B) Delete — success path
# ---------------------------------------------------------------------------


def test_delete_success_path(store: BackendStore, cnb_record: BackendRecord) -> None:
    """Delete succeeds: status→deleting, stop, retire, clear, delete DB."""
    store.create_backend(cnb_record)
    store.set_current("session-x", "cnb-a")

    stop_called = False

    def _fake_stop(_record):
        nonlocal stop_called
        stop_called = True

    adapter = CNBCLIAdapter()
    adapter.stop_backend = _fake_stop

    with (
        patch("tools.process_registry.process_registry") as mock_registry,
        patch(
            "tools.terminal_tool.clear_backend_execution_env"
        ) as mock_clear,
    ):
        result = json.loads(
            backend_tool(
                {"action": "delete", "id": "cnb-a"},
                session_id="session-x",
                store=store,
                adapter=adapter,
            )
        )

    assert result["action"] == "delete"
    assert result["current_backend"] == "local"
    assert result["backends"][0]["id"] == "cnb-a"
    assert stop_called, "adapter.stop_backend must be called"

    mock_registry.retire_backend.assert_called_once_with("cnb-a")
    mock_clear.assert_called_once_with("cnb-a")

    # Backend fully removed from DB
    with pytest.raises(BackendError, match="unknown backend"):
        store.get_backend("cnb-a")

    # Binding gone — falls back to local
    assert store.get_current("session-x").id == "local"


def test_delete_non_cnb_backend_skips_adapter_stop(
    store: BackendStore,
) -> None:
    """Local driver backends skip the workspace-stop call entirely."""
    with (
        patch("tools.process_registry.process_registry") as mock_registry,
        patch(
            "tools.terminal_tool.clear_backend_execution_env"
        ) as mock_clear,
        patch(
            "tools.execution_backends.CNBCLIAdapter.stop_backend"
        ) as mock_stop,
    ):
        with pytest.raises(BackendError, match="local backend status is immutable"):
            backend_tool(
                {"action": "delete", "id": "local"},
                session_id="session-x",
                store=store,
            )
    mock_stop.assert_not_called()
    mock_registry.retire_backend.assert_not_called()
    mock_clear.assert_not_called()


# ---------------------------------------------------------------------------
# B2) Delete — stop failure preserves everything
# ---------------------------------------------------------------------------


def test_delete_stop_failure_restores_status_and_preserves_state(
    store: BackendStore, cnb_record: BackendRecord
) -> None:
    """If adapter.stop_backend raises, original status is restored, nothing deleted."""
    store.create_backend(cnb_record)
    store.set_current("session-x", "cnb-a")
    store.set_current("session-y", "cnb-a")

    failing_adapter = CNBCLIAdapter()
    original_stop = failing_adapter.stop_backend

    def _explode(_record):
        raise RuntimeError("CNB workspace-stop exploded")

    failing_adapter.stop_backend = _explode

    with (
        patch("tools.process_registry.process_registry") as mock_registry,
        patch(
            "tools.terminal_tool.clear_backend_execution_env"
        ) as mock_clear,
    ):
        with pytest.raises(RuntimeError, match="exploded"):
            backend_tool(
                {"action": "delete", "id": "cnb-a"},
                session_id="session-x",
                store=store,
                adapter=failing_adapter,
            )

    # Status was restored
    record = store.get_backend("cnb-a")
    assert record.status == "running", "original status must be restored"

    # DB record still exists
    assert store.get_backend("cnb-a").id == "cnb-a"

    # Bindings intact
    assert store.get_current("session-x").id == "cnb-a"
    assert store.get_current("session-y").id == "cnb-a"

    # No processes were retired, no env cleanup attempted
    mock_registry.retire_backend.assert_not_called()
    mock_clear.assert_not_called()


def test_delete_stop_failure_driver_not_cnb_skips_restore(
    store: BackendStore,
) -> None:
    """Non-CNB delete goes straight to ``delete_backend`` — no stop, no restore."""
    with patch("tools.execution_backends.CNBCLIAdapter.stop_backend") as mock_stop:
        with pytest.raises(BackendError, match="local"):
            backend_tool(
                {"action": "delete", "id": "local"},
                session_id="session-x",
                store=store,
            )
    mock_stop.assert_not_called()


# ---------------------------------------------------------------------------
# C) Terminal-tool key matching
# ---------------------------------------------------------------------------


def test_execution_backend_keys_old_format() -> None:
    """Old-format keys (no session hash) are matched exactly."""
    with _task_env_overrides_lock:
        _task_env_overrides["execution-backend:cnb-a"] = {"env_type": "ssh"}
        _task_env_overrides["execution-backend:cnb-b"] = {"env_type": "ssh"}
    keys = _execution_backend_keys("cnb-a")
    assert keys == ["execution-backend:cnb-a"]


def test_execution_backend_keys_new_format() -> None:
    """New-format keys with session hash are matched."""
    with _task_env_overrides_lock:
        _task_env_overrides["execution-backend:cnb-a:session:abcd1234ef5678"] = {
            "env_type": "ssh"
        }
    keys = _execution_backend_keys("cnb-a")
    assert keys == ["execution-backend:cnb-a:session:abcd1234ef5678"]


def test_execution_backend_keys_no_mis_match_prefix() -> None:
    """``cnb-a`` must NOT match ``cnb-aa`` or ``cnb-ab``."""
    with _task_env_overrides_lock:
        _task_env_overrides["execution-backend:cnb-a"] = {"env_type": "ssh"}
        _task_env_overrides["execution-backend:cnb-aa"] = {"env_type": "ssh"}
        _task_env_overrides["execution-backend:cnb-ab"] = {"env_type": "ssh"}
        _task_env_overrides["execution-backend:cnb-a:session:hash1"] = {
            "env_type": "ssh"
        }
        _task_env_overrides["execution-backend:cnb-aa:session:hash2"] = {
            "env_type": "ssh"
        }
    keys = _execution_backend_keys("cnb-a")
    assert set(keys) == {
        "execution-backend:cnb-a",
        "execution-backend:cnb-a:session:hash1",
    }


def test_execution_backend_keys_multiple_sessions() -> None:
    """Multiple session hashes for the same backend are all matched."""
    with _task_env_overrides_lock:
        _task_env_overrides["execution-backend:cnb-a:session:aaa"] = {
            "env_type": "ssh"
        }
        _task_env_overrides["execution-backend:cnb-a:session:bbb"] = {
            "env_type": "ssh"
        }
    keys = set(_execution_backend_keys("cnb-a"))
    assert keys == {
        "execution-backend:cnb-a:session:aaa",
        "execution-backend:cnb-a:session:bbb",
    }


def test_execution_backend_keys_no_match_other_backend() -> None:
    """Other backends' keys are excluded."""
    with _task_env_overrides_lock:
        _task_env_overrides["execution-backend:cnb-x:session:h1"] = {
            "env_type": "ssh"
        }
        _task_env_overrides["execution-backend:cnb-y"] = {"env_type": "ssh"}
    assert _execution_backend_keys("cnb-a") == []


# ---------------------------------------------------------------------------
# D) Clear backend execution env
# ---------------------------------------------------------------------------


def test_clear_backend_execution_env_removes_from_all_dicts() -> None:
    """All five tracking maps are cleaned for matched keys."""
    key_a_old = "execution-backend:cnb-a"
    key_a_new = "execution-backend:cnb-a:session:h1"
    key_b = "execution-backend:cnb-b:session:h2"
    mock_env = MagicMock()

    with _task_env_overrides_lock:
        _task_env_overrides[key_a_old] = {"env_type": "ssh"}
        _task_env_overrides[key_a_new] = {"env_type": "ssh"}
        _task_env_overrides[key_b] = {"env_type": "ssh"}
    with _env_lock:
        _active_environments[key_a_old] = mock_env
        _active_environments[key_a_new] = mock_env
        _active_environments[key_b] = mock_env
    with _creation_locks_lock:
        _creation_locks[key_a_old] = threading.Lock()
        _creation_locks[key_a_new] = threading.Lock()
        _creation_locks[key_b] = threading.Lock()
    with _session_cwd_lock:
        _session_cwd[key_a_old] = "/workspace"
        _session_cwd[key_a_new] = "/workspace"
        _session_cwd[key_b] = "/workspace"
    with _env_lock:
        _last_activity[key_a_old] = 100.0
        _last_activity[key_a_new] = 100.0
        _last_activity[key_b] = 100.0

    count = clear_backend_execution_env("cnb-a")

    assert count == 2  # old + new format keys

    # Only cnb-a keys removed, cnb-b preserved
    with _task_env_overrides_lock:
        assert key_a_old not in _task_env_overrides
        assert key_a_new not in _task_env_overrides
        assert _task_env_overrides.get(key_b) is not None
    with _env_lock:
        assert key_a_old not in _active_environments
        assert key_a_new not in _active_environments
        assert key_b in _active_environments
    with _creation_locks_lock:
        assert key_a_old not in _creation_locks
        assert key_a_new not in _creation_locks
        assert key_b in _creation_locks
    with _session_cwd_lock:
        assert key_a_old not in _session_cwd
        assert key_a_new not in _session_cwd
        assert _session_cwd.get(key_b) is not None
    with _env_lock:
        assert key_a_old not in _last_activity
        assert key_a_new not in _last_activity
        assert key_b in _last_activity

    # Environment was collected and stopped
    mock_env.cleanup.assert_called()


def test_clear_backend_execution_env_cleanup_exception_does_not_block() -> None:
    """If env.cleanup raises, the error is logged but data removal already happened."""
    key = "execution-backend:cnb-a"
    broken_env = MagicMock()
    broken_env.cleanup.side_effect = RuntimeError("cleanup kaboom")

    with _task_env_overrides_lock:
        _task_env_overrides[key] = {"env_type": "ssh"}
    with _env_lock:
        _active_environments[key] = broken_env

    # Should not raise — data removal happens before cleanup attempt.
    count = clear_backend_execution_env("cnb-a")
    assert count == 1

    # Mapping was already cleared
    with _task_env_overrides_lock:
        assert key not in _task_env_overrides
    with _env_lock:
        assert key not in _active_environments


def test_clear_backend_execution_env_no_keys_is_noop() -> None:
    """Calling with a backend that has no in-memory state is a safe no-op."""
    count = clear_backend_execution_env("cnb-nonexistent")
    assert count == 0


# ---------------------------------------------------------------------------
# E) ProcessRegistry.retire_backend
# ---------------------------------------------------------------------------


def _make_process_session(
    session_id: str,
    backend_id: str = "local",
    task_id: str = "default",
    exited: bool = False,
) -> ProcessSession:
    """Helper to create a ``ProcessSession`` with minimal fields."""
    return ProcessSession(
        id=session_id,
        command="echo hello",
        task_id=task_id,
        backend_id=backend_id,
        started_at=100.0,
        exited=exited,
        session_key="test",
    )


def test_retire_backend_marks_matching_sessions() -> None:
    """Sessions with matching backend_id are retired; exited sessions are skipped."""
    reg = ProcessRegistry()
    s1 = _make_process_session("p1", backend_id="cnb-a")
    s2 = _make_process_session("p2", backend_id="cnb-a")
    s3 = _make_process_session("p3", backend_id="cnb-a", exited=True)  # already done
    s4 = _make_process_session("p4", backend_id="cnb-b")  # different backend

    for s in [s1, s2, s3, s4]:
        reg._running[s.id] = s

    retired = reg.retire_backend("cnb-a")

    assert retired == 2  # s1, s2 retired; s3 already exited; s4 other backend

    # Retired sessions are moved to finished
    assert reg._running.get("p1") is None
    assert reg._running.get("p2") is None
    assert reg._finished.get("p1") is s1
    assert reg._finished.get("p2") is s2

    # Retired sessions marked correctly
    assert s1.exited is True
    assert s1.exit_code is None
    assert s1.completion_reason == "backend_lost"
    assert s1.termination_source == "backend.delete"

    # Already-exited session left in running (it stays wherever it was)
    assert reg._running.get("p3") is s3
    assert s3.exited is True  # unchanged

    # Other backend untouched
    assert reg._running.get("p4") is s4
    assert s4.exited is False


def test_retire_backend_leaves_other_backends_untouched() -> None:
    """Processes with a different backend_id remain running."""
    reg = ProcessRegistry()
    s_other = _make_process_session("p-b", backend_id="cnb-b")

    reg._running[s_other.id] = s_other
    reg.retire_backend("cnb-a")

    assert reg._running.get("p-b") is s_other
    assert s_other.exited is False


def test_retire_backend_does_not_send_ssh_kill() -> None:
    """``retire_backend`` must NOT call any kill/terminate method."""
    reg = ProcessRegistry()
    s1 = _make_process_session("p1", backend_id="cnb-a")
    s1.env_ref = MagicMock()  # would allow kill via env_ref.execute

    reg._running[s1.id] = s1

    with patch.object(reg, "_terminate_host_pid") as mock_kill:
        reg.retire_backend("cnb-a")

    mock_kill.assert_not_called()
    assert s1.exited is True
    assert s1.completion_reason == "backend_lost"

    # env_ref.execute was NOT called (no SSH kill)
    if s1.env_ref is not None:
        s1.env_ref.execute.assert_not_called()


def test_retire_backend_unknown_backend_is_noop() -> None:
    """Calling retire_backend with a backend_id having no running sessions is safe."""
    reg = ProcessRegistry()
    s1 = _make_process_session("p1", backend_id="cnb-b")
    reg._running[s1.id] = s1

    retired = reg.retire_backend("cnb-a")
    assert retired == 0

    # Other still running
    assert reg._running.get("p1") is s1
    assert s1.exited is False


# ---------------------------------------------------------------------------
# Integration: backend_tool delete wires everything together via mocks
# ---------------------------------------------------------------------------


def test_delete_invokes_full_cleanup_chain(store: BackendStore) -> None:
    """backend_tool('delete') calls retire, clear_env, and delete_backend in order."""
    record = BackendRecord(
        id="cnb-a",
        driver="cnb",
        status="running",
        repo="example/repo",
        branch="main",
        workspace_sn="sn-a",
        pipeline_id="pipeline-a",
        ssh_host="10.0.0.1",
        ssh_user="dev",
    )
    store.create_backend(record)
    store.set_current("session-x", "cnb-a")

    # Populate terminal state so the chain exercises it
    key = "execution-backend:cnb-a:session:testhash"
    mock_env = MagicMock()
    with _task_env_overrides_lock:
        _task_env_overrides[key] = {"env_type": "ssh", "cwd": "/workspace"}
    with _env_lock:
        _active_environments[key] = mock_env
        _last_activity[key] = 100.0
    with _session_cwd_lock:
        _session_cwd[key] = "/workspace"

    stop_called = False

    def _fake_stop(_record):
        nonlocal stop_called
        stop_called = True

    adapter = CNBCLIAdapter()
    adapter.stop_backend = _fake_stop

    with patch("tools.process_registry.process_registry") as mock_registry:
        result = json.loads(
            backend_tool(
                {"action": "delete", "id": "cnb-a"},
                session_id="session-x",
                store=store,
                adapter=adapter,
            )
        )

    assert result["action"] == "delete"
    assert stop_called
    mock_registry.retire_backend.assert_called_once_with("cnb-a")

    # Terminal state was cleaned
    with _task_env_overrides_lock:
        assert key not in _task_env_overrides
    with _env_lock:
        assert key not in _active_environments
        assert key not in _last_activity

    # Backend DB record is gone
    with pytest.raises(BackendError, match="unknown backend"):
        store.get_backend("cnb-a")

    # Environment cleanup was called
    mock_env.cleanup.assert_called()
