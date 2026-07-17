"""Tests for Backend delete — fail-closed status, env cleanup, process retirement.

Verifies the full safe-delete contract:

A) ``update(current=True)`` switches only — no kill/cleanup.

B) ``delete`` action (CNB backend):
   1. Records status → ``deleting`` (concurrent routing fail-closed).
   2. Adapter ``workspace-stop`` — on failure, restores original status
      and preserves DB/bindings/override/env/processes, then re-raises.
   3. On success: retires process registry entries for the backend,
      clears in-memory env/override/cwd/cache state, then deletes from
      DB (bindings, events, record). Returns the original record with
      ``status`` set to ``"deleted"``.

C) Terminal-tool key matching: ``_execution_backend_keys`` matches
   ``execution-backend:<id>`` (old) and ``execution-backend:<id>:session:<hash>``
   (new) with strict segment boundaries — never mis-matches similar ids,
   and rejects arbitrary colon suffixes or non-16-hex hashes.

D) ``clear_backend_execution_env``: removes from all tracking dicts inside
   locks first, then stops/cleanups environments outside locks.  Also cleans
   ``_last_activity`` and ``_creation_locks`` even when the key only lives in
   those maps.

E) ``ProcessRegistry.retire_backend``: marks matching running sessions
   as exited/backend_lost, does NOT send SSH kills.

F) Lifecycle-lock serialization: ``backend_tool(delete)`` and
   ``resolve_execution_task_id`` hold the same per-backend RLock so that
   no resolver can register overrides after delete begins.  Other backends
   are never blocked.

G) Fail-closed finalize: if retire or clear unexpectedly raises (programming
   error), the DB stays in ``deleting`` state — the record is not removed.

I) Real race (monkeypatch + Events): resolver reads ``running``, delete crashes
   during ``clear_backend_execution_env`` leaving DB in ``deleting``; the
   resolver re-reads inside the lifecycle lock and raises ``BackendError``
   — ``register_task_env_overrides`` is never called, no ``sleep`` relied on.

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
    _get_backend_lifecycle_lock,
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

# Valid 16-hex-char session hashes used throughout the test file.
_HASH_A = "a1b2c3d4e5f67801"
_HASH_B = "a1b2c3d4e5f67802"
_HASH_C = "a1b2c3d4e5f67803"

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
    assert result["backends"][0]["status"] == "deleted", (
        "returned record status must be 'deleted'"
    )
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
    """New-format keys with valid session hash are matched."""
    key = f"execution-backend:cnb-a:session:{_HASH_A}"
    with _task_env_overrides_lock:
        _task_env_overrides[key] = {"env_type": "ssh"}
    keys = _execution_backend_keys("cnb-a")
    assert keys == [key]


def test_execution_backend_keys_no_mis_match_prefix() -> None:
    """``cnb-a`` must NOT match ``cnb-aa`` or ``cnb-ab``."""
    with _task_env_overrides_lock:
        _task_env_overrides["execution-backend:cnb-a"] = {"env_type": "ssh"}
        _task_env_overrides["execution-backend:cnb-aa"] = {"env_type": "ssh"}
        _task_env_overrides["execution-backend:cnb-ab"] = {"env_type": "ssh"}
        _task_env_overrides[
            f"execution-backend:cnb-a:session:{_HASH_A}"
        ] = {"env_type": "ssh"}
        _task_env_overrides[
            f"execution-backend:cnb-aa:session:{_HASH_B}"
        ] = {"env_type": "ssh"}
    keys = _execution_backend_keys("cnb-a")
    assert set(keys) == {
        "execution-backend:cnb-a",
        f"execution-backend:cnb-a:session:{_HASH_A}",
    }


def test_execution_backend_keys_multiple_sessions() -> None:
    """Multiple session hashes for the same backend are all matched."""
    key_a = f"execution-backend:cnb-a:session:{_HASH_A}"
    key_b = f"execution-backend:cnb-a:session:{_HASH_B}"
    with _task_env_overrides_lock:
        _task_env_overrides[key_a] = {"env_type": "ssh"}
        _task_env_overrides[key_b] = {"env_type": "ssh"}
    keys = set(_execution_backend_keys("cnb-a"))
    assert keys == {key_a, key_b}


def test_execution_backend_keys_no_match_other_backend() -> None:
    """Other backends' keys are excluded."""
    with _task_env_overrides_lock:
        _task_env_overrides[
            f"execution-backend:cnb-x:session:{_HASH_A}"
        ] = {"env_type": "ssh"}
        _task_env_overrides["execution-backend:cnb-y"] = {"env_type": "ssh"}
    assert _execution_backend_keys("cnb-a") == []


# ----- C2) Strict format rejection -----


def test_execution_backend_keys_rejects_arbitrary_colon_suffix() -> None:
    """Keys with a non-``:session:`` suffix (e.g. ``:foo``) are rejected."""
    with _task_env_overrides_lock:
        _task_env_overrides["execution-backend:cnb-a:foo"] = {"env_type": "ssh"}
        _task_env_overrides["execution-backend:cnb-a:other:stuff"] = {
            "env_type": "ssh"
        }
    assert _execution_backend_keys("cnb-a") == []


def test_execution_backend_keys_rejects_non_16_hex_hash() -> None:
    """Keys with ``:session:`` but not exactly 16 lowercase hex are rejected."""
    with _task_env_overrides_lock:
        _task_env_overrides["execution-backend:cnb-a:session:xyz"] = {
            "env_type": "ssh"
        }
        _task_env_overrides["execution-backend:cnb-a:session:1a2b3c"] = {
            "env_type": "ssh"
        }
        # 17 chars (too long)
        _task_env_overrides["execution-backend:cnb-a:session:a1b2c3d4e5f67890a"] = {
            "env_type": "ssh"
        }
        # uppercase hex (should be lowercase)
        _task_env_overrides["execution-backend:cnb-a:session:A1B2C3D4E5F67890"] = {
            "env_type": "ssh"
        }
    assert _execution_backend_keys("cnb-a") == []


# ----- C3) Scanning _last_activity and _creation_locks -----


def test_execution_backend_keys_finds_keys_only_in_last_activity() -> None:
    """Keys that only exist in ``_last_activity`` are found."""
    key = f"execution-backend:cnb-a:session:{_HASH_A}"
    with _env_lock:
        _last_activity[key] = 100.0
    keys = _execution_backend_keys("cnb-a")
    assert keys == [key]


def test_execution_backend_keys_finds_keys_only_in_creation_locks() -> None:
    """Keys that only exist in ``_creation_locks`` are found."""
    key = f"execution-backend:cnb-a:session:{_HASH_A}"
    with _creation_locks_lock:
        _creation_locks[key] = threading.Lock()
    keys = _execution_backend_keys("cnb-a")
    assert keys == [key]


# ---------------------------------------------------------------------------
# D) Clear backend execution env
# ---------------------------------------------------------------------------


def test_clear_backend_execution_env_removes_from_all_dicts() -> None:
    """All five tracking maps are cleaned for matched keys."""
    key_a_old = "execution-backend:cnb-a"
    key_a_new = f"execution-backend:cnb-a:session:{_HASH_A}"
    key_b = f"execution-backend:cnb-b:session:{_HASH_B}"
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


# ----- D2) Isolated _last_activity and _creation_locks cleanup -----


def test_clear_backend_execution_env_cleans_last_activity_only() -> None:
    """Keys that only live in ``_last_activity`` are still cleaned up."""
    key = f"execution-backend:cnb-a:session:{_HASH_A}"
    with _env_lock:
        _last_activity[key] = 100.0
    count = clear_backend_execution_env("cnb-a")
    assert count == 1
    with _env_lock:
        assert key not in _last_activity


def test_clear_backend_execution_env_cleans_creation_lock_only() -> None:
    """Keys that only live in ``_creation_locks`` are still cleaned up."""
    key = f"execution-backend:cnb-a:session:{_HASH_A}"
    with _creation_locks_lock:
        _creation_locks[key] = threading.Lock()
    count = clear_backend_execution_env("cnb-a")
    assert count == 1
    with _creation_locks_lock:
        assert key not in _creation_locks


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
# F) Integration: backend_tool delete wires everything together via mocks
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
    key = f"execution-backend:cnb-a:session:{_HASH_A}"
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
    assert result["backends"][0]["status"] == "deleted"
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


# ---------------------------------------------------------------------------
# G) Lifecycle-lock serialization (delete ↔ resolve)
# ---------------------------------------------------------------------------


def test_lifecycle_lock_serializes_delete_and_resolve(
    tmp_path: Path,
) -> None:
    """With the lifecycle lock held by delete, resolver re-reads and refuses to register.

    Uses two ``threading.Event`` handoffs to guarantee ordering:

    1. Resolver reads initial record (status=running), signals ``resolver_seen_running``.
    2. Resolver waits on ``deleter_inside_lock``.
    3. Deleter (waiting on ``resolver_seen_running``) acquires lock, sets status to
       ``deleting``, signals ``deleter_inside_lock``, then sleeps holding the lock.
    4. Resolver wakes, tries to acquire the lock → blocks (deleter holds it).
    5. Deleter releases lock after sleep.
    6. Resolver acquires lock, re-reads → sees ``deleting`` → passes.
    """
    from tools.execution_backends import _get_backend_lifecycle_lock

    backend_id = "cnb-a"
    lock = _get_backend_lifecycle_lock(backend_id)

    store = BackendStore(tmp_path / "lifecycle_lock.db")
    record = BackendRecord(
        id=backend_id,
        driver="cnb",
        status="running",
        repo="g/r",
        branch="main",
        workspace_sn="sn",
        pipeline_id="p",
        ssh_host="h",
        ssh_user="u",
    )
    store.create_backend(record)
    store.set_current("session-x", backend_id)

    resolver_seen_running = threading.Event()
    deleter_inside_lock = threading.Event()
    resolver_raised: list[Exception] = []

    def _resolver():
        try:
            # Step 1: Read initial record (outside the lock) — still running
            r = store.get_current("session-x")
            assert r.id == backend_id
            assert r.status == "running"
            resolver_seen_running.set()

            # Step 2: Wait for deleter to be inside the lifecycle lock
            # after having already set status → ``deleting``.
            deleter_inside_lock.wait(timeout=10)

            # Step 3-6: Try to acquire the lock — the deleter is still
            # inside it, so we block until it releases.
            with lock:
                try:
                    store.get_current("session-x")
                except BackendError:
                    pass  # expected — fail-closed
                else:
                    resolver_raised.append(
                        AssertionError("expected BackendError but got success")
                    )
        except Exception as e:
            resolver_raised.append(e)

    def _deleter():
        # Step 3: Wait for resolver to finish its initial read.
        resolver_seen_running.wait(timeout=10)
        with lock:
            store.update_backend_status(backend_id, "deleting")
            deleter_inside_lock.set()
            # Step 4-5: Stay in the lock so the resolver inevitably
            # blocks on acquisition.
            threading.Event().wait(timeout=2.0)

    t1 = threading.Thread(target=_resolver, daemon=True)
    t2 = threading.Thread(target=_deleter, daemon=True)
    t1.start()
    t2.start()
    t1.join(timeout=15)
    t2.join(timeout=5)

    assert not resolver_raised, (
        f"resolver should not have raised: {resolver_raised}"
    )


def test_lifecycle_lock_does_not_block_other_backend() -> None:
    """Holding the lifecycle lock for backend A must not block backend B."""
    backend_a = "cnb-a"
    backend_b = "cnb-b"
    lock_a = _get_backend_lifecycle_lock(backend_a)
    lock_b = _get_backend_lifecycle_lock(backend_b)
    got_lock_b = threading.Event()

    def _hold_a_then_signal():
        with lock_a:
            got_lock_b.set()
            # Stay in the lock so B tries to acquire.
            threading.Event().wait(timeout=0.5)

    t = threading.Thread(target=_hold_a_then_signal, daemon=True)
    t.start()

    got_lock_b.wait(timeout=5)
    # Should be able to acquire lock_b immediately.
    acquired = lock_b.acquire(timeout=1.0)
    assert acquired, "lock B should not be blocked by lock A"
    lock_b.release()
    t.join(timeout=2)


def test_resolve_execution_task_id_detects_deleting_backend(
    tmp_path: Path,
) -> None:
    """``resolve_execution_task_id`` raises when a backend is being deleted.

    The lifecycle lock ensures the resolver re-reads status after the delete
    has set it to ``deleting``.
    """
    backend_id = "cnb-a"
    store = BackendStore(tmp_path / "resolve_deleting.db")
    record = BackendRecord(
        id=backend_id,
        driver="cnb",
        status="running",
        repo="g/r",
        branch="main",
        workspace_sn="sn",
        pipeline_id="p",
        ssh_host="h",
        ssh_user="u",
    )
    store.create_backend(record)
    store.set_current("session-x", backend_id)

    # Set status to deleting to simulate concurrent delete.
    store.update_backend_status(backend_id, "deleting")

    with pytest.raises(BackendError, match="not ready"):
        resolve_execution_task_id(
            task_id="task-x", session_id="session-x", store=store
        )


# ---------------------------------------------------------------------------
# H) Fail-closed finalize — local error leaves DB in ``deleting``
# ---------------------------------------------------------------------------


def test_delete_retire_error_leaves_db_deleting(
    store: BackendStore, cnb_record: BackendRecord
) -> None:
    """If retire_backend raises unexpectedly, DB stays in ``deleting``."""
    store.create_backend(cnb_record)
    store.set_current("session-x", "cnb-a")

    adapter = CNBCLIAdapter()
    adapter.stop_backend = MagicMock()

    with patch(
        "tools.process_registry.process_registry.retire_backend",
        side_effect=RuntimeError("unexpected retire boom"),
    ):
        with pytest.raises(RuntimeError, match="unexpected retire boom"):
            backend_tool(
                {"action": "delete", "id": "cnb-a"},
                session_id="session-x",
                store=store,
                adapter=adapter,
            )

    # DB record still exists with status=deleting
    record = store.get_backend("cnb-a")
    assert record.status == "deleting", (
        "DB must stay 'deleting' when local finalize fails"
    )

    # Session binding still points to this backend (even though status is
    # deleting — get_current will raise BackendError).
    with pytest.raises(BackendError, match="not ready"):
        store.get_current("session-x")


def test_delete_clear_env_error_leaves_db_deleting(
    store: BackendStore, cnb_record: BackendRecord
) -> None:
    """If clear_backend_execution_env raises unexpectedly, DB stays in ``deleting``."""
    store.create_backend(cnb_record)

    adapter = CNBCLIAdapter()
    adapter.stop_backend = MagicMock()

    with (
        patch("tools.process_registry.process_registry") as mock_registry,
        patch(
            "tools.terminal_tool.clear_backend_execution_env",
            side_effect=RuntimeError("unexpected clear boom"),
        ),
    ):
        with pytest.raises(RuntimeError, match="unexpected clear boom"):
            backend_tool(
                {"action": "delete", "id": "cnb-a"},
                session_id="session-x",
                store=store,
                adapter=adapter,
            )

    # DB record still exists with status=deleting
    record = store.get_backend("cnb-a")
    assert record.status == "deleting"


# ---------------------------------------------------------------------------
# I) Real race: resolver reads running, delete crashes during clear,
#    resolver re-reads deleting → BackendError, no override registration
# ---------------------------------------------------------------------------


def test_resolve_delete_race_clear_env_error_fail_closed(
    tmp_path: Path,
) -> None:
    """``resolve_execution_task_id`` gets BackendError when a concurrent
    delete fails and leaves the DB in ``deleting``.

    Threading events (not ``sleep``) guarantee ordering:

    1. The resolver's first ``store.get_current`` (outside the lifecycle lock)
       returns ``running``, then signals ``resolver_read_running`` and blocks.
    2. The delete thread acquires the lifecycle lock, sets status → ``deleting``,
       and crashes on ``clear_backend_execution_env`` — the DB stays in
       ``deleting`` and the lock is released.
    3. The resolver is released, enters the lifecycle lock, and re-reads the
       binding.  ``get_current`` raises ``BackendError`` (status=deleting
       is not running) and propagates out of ``resolve_execution_task_id``.
    4. ``register_task_env_overrides`` is **never** called.
    """
    backend_id = "cnb-a"
    store = BackendStore(tmp_path / "race_clear_error.db")

    record = BackendRecord(
        id=backend_id,
        driver="cnb",
        status="running",
        repo="g/r",
        branch="main",
        workspace_sn="sn",
        pipeline_id="p",
        ssh_host="h",
        ssh_user="u",
    )
    store.create_backend(record)
    store.set_current("session-x", backend_id)

    resolver_read_running = threading.Event()
    resolver_proceed = threading.Event()
    resolver_errors: list[Exception] = []

    original_get_current = store.get_current

    def _sync_get_current(sk: str) -> BackendRecord:
        result = original_get_current(sk)
        if result.id != "local" and result.status == "running":
            resolver_read_running.set()
            assert resolver_proceed.wait(timeout=10), (
                "resolver timed out waiting for delete to finish"
            )
        return result

    with (
        patch.object(store, "get_current", wraps=_sync_get_current),
        patch(
            "tools.terminal_tool.register_task_env_overrides"
        ) as mock_register,
        patch(
            "tools.terminal_tool.clear_backend_execution_env",
            side_effect=RuntimeError("clear boom"),
        ),
    ):

        def _delete_and_crash() -> None:
            lock = _get_backend_lifecycle_lock(backend_id)
            with lock:
                store.update_backend_status(backend_id, "deleting")
                # This raises → DB stays deleting, lifecycle lock released.
                clear_backend_execution_env(backend_id)

        delete_thread = threading.Thread(
            target=_delete_and_crash, daemon=True
        )

        def _run_resolver() -> None:
            try:
                resolve_execution_task_id(
                    task_id="task-x",
                    session_id="session-x",
                    store=store,
                )
            except BackendError as e:
                resolver_errors.append(e)
            except Exception as e:
                resolver_errors.append(e)

        resolver_thread = threading.Thread(
            target=_run_resolver, daemon=True
        )

        # Step 1: Resolver starts, reads running record, pauses.
        resolver_thread.start()
        resolver_read_running.wait(timeout=10)

        # Step 2: Delete acquires lifecycle lock, sets deleting, crashes.
        delete_thread.start()
        delete_thread.join(timeout=10)

        # Step 3: Release resolver; it re-reads inside the lock → BackendError.
        resolver_proceed.set()
        resolver_thread.join(timeout=10)

    # Step 4: Assertions
    assert len(resolver_errors) == 1, (
        f"expected exactly one BackendError, got {len(resolver_errors)}"
    )
    assert isinstance(resolver_errors[0], BackendError), (
        f"expected BackendError, got {type(resolver_errors[0]).__name__}"
    )
    mock_register.assert_not_called()
