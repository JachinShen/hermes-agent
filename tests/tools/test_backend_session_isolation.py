"""Tests for session-scoped CNB execution keys and override concurrent locking.

Requirement:
- ``resolve_execution_task_id`` emits ``execution-backend:<id>:session:<hash>``
- ``_backend_id_from_task_id`` parses both old and new key formats
- ``_task_env_overrides`` operations are protected by an RLock and return copies
- The key does NOT embed the raw session/task ID
- Same backend + same session → stable key
- Same backend + different session → different key
"""

from __future__ import annotations

import copy
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.execution_backends import (
    BackendStore,
    _session_key_hash,
    resolve_execution_task_id,
    maybe_resolve_execution_task_id,
)
from tools.process_registry import _backend_id_from_task_id  # noqa: PLC2701
from tools import terminal_tool


# =========================================================================
# Helpers
# =========================================================================


def _running_cnb_record(backend_id: str, *, host: str) -> "BackendRecord":
    """Minimal BackendRecord for BackendStore.create_backend."""
    from tools.execution_backends import BackendRecord

    return BackendRecord(
        id=backend_id,
        driver="cnb",
        status="running",
        repo="example/repo",
        branch="main",
        workspace_sn=f"sn-{backend_id}",
        pipeline_id=f"pipeline-{backend_id}",
        ssh_host=host,
        ssh_user="dev",
        ssh_port=22,
        cwd="/workspace",
    )


@pytest.fixture
def store(tmp_path: Path) -> BackendStore:
    return BackendStore(tmp_path / "execution_backends.db")


@pytest.fixture(autouse=True)
def _clean_overrides():
    """Isolate _task_env_overrides between tests."""
    before = dict(terminal_tool._task_env_overrides)
    terminal_tool._task_env_overrides.clear()
    yield
    terminal_tool._task_env_overrides.clear()
    terminal_tool._task_env_overrides.update(before)


# =========================================================================
# 1. _session_key_hash helper
# =========================================================================


class TestSessionKeyHash:
    def test_returns_16_hex_chars(self) -> None:
        h = _session_key_hash("session-a")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic_for_same_input(self) -> None:
        assert _session_key_hash("session-a") == _session_key_hash("session-a")

    def test_different_for_different_input(self) -> None:
        assert _session_key_hash("session-a") != _session_key_hash("session-b")

    def test_does_not_leak_raw_session_id(self) -> None:
        raw = "my-raw-session-42"
        h = _session_key_hash(raw)
        assert raw not in h
        # The hash is only 16 hex chars; no part of the raw string
        # (including its substring "42") can appear meaningfully,
        # but a hex nibble coincidence is possible, so only assert
        # the full raw string is absent.
        assert "my-raw-session" not in h


# =========================================================================
# 2. Session-scoped execution key
# =========================================================================


class TestSessionScopedExecutionKey:
    def test_key_includes_backend_and_session_hash(
        self, store: BackendStore
    ) -> None:
        """The key has the form execution-backend:<id>:session:<16hex>."""
        store.create_backend(
            _running_cnb_record("cnb-x", host="x.example")
        )
        store.set_current("session-a", "cnb-x")
        key = resolve_execution_task_id(
            task_id="t1", session_id="session-a", store=store
        )
        assert key.startswith("execution-backend:cnb-x:session:")
        suffix = key[len("execution-backend:cnb-x:session:"):]
        assert len(suffix) == 16
        assert all(c in "0123456789abcdef" for c in suffix)

    def test_same_backend_same_session_stable(
        self, store: BackendStore
    ) -> None:
        """Calling twice with same backend + session yields the same key."""
        store.create_backend(
            _running_cnb_record("cnb-y", host="y.example")
        )
        store.set_current("session-a", "cnb-y")
        k1 = resolve_execution_task_id(
            task_id="t1", session_id="session-a", store=store
        )
        k2 = resolve_execution_task_id(
            task_id="t2", session_id="session-a", store=store
        )
        assert k1 == k2

    def test_same_backend_different_session_differs(
        self, store: BackendStore
    ) -> None:
        """Same backend but different session keys MUST differ."""
        store.create_backend(
            _running_cnb_record("cnb-z", host="z.example")
        )
        store.set_current("session-a", "cnb-z")
        store.set_current("session-b", "cnb-z")

        k_a = resolve_execution_task_id(
            task_id="t", session_id="session-a", store=store
        )
        k_b = resolve_execution_task_id(
            task_id="t", session_id="session-b", store=store
        )
        assert k_a != k_b

    def test_does_not_leak_raw_session_or_task_id(
        self, store: BackendStore
    ) -> None:
        """The key must NOT contain the raw session_id or task_id."""
        store.create_backend(
            _running_cnb_record("cnb-leak", host="leak.example")
        )
        store.set_current("secret-session-007", "cnb-leak")
        key = resolve_execution_task_id(
            task_id="top-secret-task", session_id="secret-session-007",
            store=store,
        )
        assert "secret-session-007" not in key
        assert "top-secret-task" not in key

    def test_key_is_compatible_with_old_parser(
        self, store: BackendStore
    ) -> None:
        """_backend_id_from_task_id correctly extracts the backend id."""
        store.create_backend(
            _running_cnb_record("cnb-old-compat", host="c.example")
        )
        store.set_current("session-a", "cnb-old-compat")
        key = resolve_execution_task_id(
            task_id="t", session_id="session-a", store=store
        )
        assert _backend_id_from_task_id(key) == "cnb-old-compat"


# =========================================================================
# 3. _backend_id_from_task_id — both old and new format
# =========================================================================


class TestBackendIdFromTaskIdNew:
    def test_local_for_default(self) -> None:
        assert _backend_id_from_task_id("default") == "local"

    def test_local_for_plain_task(self) -> None:
        assert _backend_id_from_task_id("child-task") == "local"

    def test_parses_old_format(self) -> None:
        assert (
            _backend_id_from_task_id("execution-backend:cnb-a") == "cnb-a"
        )

    def test_parses_new_format(self) -> None:
        assert (
            _backend_id_from_task_id(
                "execution-backend:cnb-a:session:a1b2c3d4e5f67890"
            )
            == "cnb-a"
        )

    def test_parses_new_format_with_dots(self) -> None:
        assert (
            _backend_id_from_task_id(
                "execution-backend:my.cnb-backend:session:0011223344556677"
            )
            == "my.cnb-backend"
        )

    def test_parses_old_format_with_dots(self) -> None:
        assert (
            _backend_id_from_task_id("execution-backend:my.cnb-backend")
            == "my.cnb-backend"
        )

    def test_prefix_case_sensitive(self) -> None:
        assert _backend_id_from_task_id("Execution-Backend:foo") == "local"


# =========================================================================
# 4. Each session-scoped key registers SSH no-sync override
# =========================================================================


def test_session_key_registers_ssh_no_sync_override(
    store: BackendStore,
) -> None:
    """Each unique session key calls register with SSH env_type + no sync."""
    store.create_backend(
        _running_cnb_record("cnb-reg", host="reg.example")
    )
    store.set_current("session-a", "cnb-reg")

    captured_task_id: list[str] = []
    captured_overrides: list[dict] = []

    def _fake_register(task_id, overrides):
        captured_task_id.append(task_id)
        captured_overrides.append(overrides)
        # Actually register so downstream resolve can find it
        terminal_tool._task_env_overrides[task_id] = overrides

    with patch(
        "tools.terminal_tool.register_task_env_overrides",
        _fake_register,
    ):
        key = resolve_execution_task_id(
            task_id="t", session_id="session-a", store=store
        )

    assert len(captured_task_id) >= 1
    registered_key = captured_task_id[0]
    assert registered_key == key

    # Verify the overrides dict — read from the actual registry
    overrides = terminal_tool.resolve_task_overrides(key)
    assert overrides.get("env_type") == "ssh"
    assert overrides.get("ssh_sync_hermes_home") is False
    assert overrides.get("ssh_host") == "reg.example"


# =========================================================================
# 5. Concurrent register / resolve / clear — no crash, no mutable refs
# =========================================================================

# Shared barrier for thread synchronisation
_BARRIER_COUNT = 8


def _concurrent_worker(
    barrier: threading.Barrier,
    task_id: str,
    errors: list[Exception],
) -> None:
    """Worker that registers, resolves, and clears overrides concurrently."""
    try:
        barrier.wait(timeout=10)

        # register
        terminal_tool.register_task_env_overrides(
            task_id, {"env_type": "ssh", "cwd": "/workspace"}
        )

        # resolve — should not crash
        overrides = terminal_tool.resolve_task_overrides(task_id)
        assert overrides.get("env_type") == "ssh"

        # resolve container task id — should not crash
        cid = terminal_tool._resolve_container_task_id(task_id)  # noqa: SLF001

        # clear
        terminal_tool.clear_task_env_overrides(task_id)

        # after clear, resolve returns empty
        gone = terminal_tool.resolve_task_overrides(task_id)
        assert gone == {}
    except Exception as exc:  # noqa: BLE001
        errors.append(exc)


def test_concurrent_register_resolve_clear_no_crash() -> None:
    """Many threads register/resolve/clear without errors."""
    errors: list[Exception] = []
    barrier = threading.Barrier(_BARRIER_COUNT)

    threads = [
        threading.Thread(
            target=_concurrent_worker,
            args=(barrier, f"task-{i}", errors),
        )
        for i in range(_BARRIER_COUNT)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    assert not errors, f"Concurrent errors: {errors}"


def test_resolve_task_overrides_returns_copy() -> None:
    """Mutating the returned dict must NOT affect the internal registry."""
    terminal_tool.register_task_env_overrides(
        "my-task", {"env_type": "ssh", "cwd": "/tmp"}
    )

    result = terminal_tool.resolve_task_overrides("my-task")
    result["env_type"] = "hacked"

    # Internal state must be unchanged
    internal = terminal_tool._task_env_overrides.get("my-task", {})  # noqa: SLF001
    assert internal.get("env_type") == "ssh"
    assert internal.get("env_type") != "hacked"


# =========================================================================
# 6. Environment tools use session-scoped keys
# =========================================================================


def test_terminal_uses_session_scoped_key(
    store: BackendStore,
) -> None:
    """maybe_resolve_execution_task_id for terminal produces scoped key."""
    store.create_backend(
        _running_cnb_record("cnb-term", host="term.example")
    )
    store.set_current("session-a", "cnb-term")

    with (
        patch(
            "tools.execution_backends.is_execution_backends_enabled",
            return_value=True,
        ),
        patch(
            "tools.execution_backends.get_backend_store",
            return_value=store,
        ),
    ):
        result = maybe_resolve_execution_task_id(
            "terminal", task_id="child", session_id="session-a"
        )

    assert result is not None
    assert result.startswith("execution-backend:cnb-term:session:")
    assert "child" not in result
    assert "session-a" not in result
    # verify _backend_id_from_task_id still works
    assert _backend_id_from_task_id(result) == "cnb-term"


def test_file_tools_use_session_scoped_key(
    store: BackendStore,
) -> None:
    """read_file/write_file etc. also get session-scoped keys."""
    store.create_backend(
        _running_cnb_record("cnb-file", host="file.example")
    )
    store.set_current("session-b", "cnb-file")

    with (
        patch(
            "tools.execution_backends.is_execution_backends_enabled",
            return_value=True,
        ),
        patch(
            "tools.execution_backends.get_backend_store",
            return_value=store,
        ),
    ):
        for name in ("read_file", "write_file", "patch", "search_files"):
            result = maybe_resolve_execution_task_id(
                name, task_id="ftask", session_id="session-b"
            )
            assert result is not None
            assert result.startswith(
                "execution-backend:cnb-file:session:"
            ), f"{name} should get a scoped key, got {result}"
            assert _backend_id_from_task_id(result) == "cnb-file"


def test_execute_code_uses_session_scoped_key(
    store: BackendStore,
) -> None:
    """execute_code also gets session-scoped keys."""
    store.create_backend(
        _running_cnb_record("cnb-code", host="code.example")
    )
    store.set_current("session-c", "cnb-code")

    with (
        patch(
            "tools.execution_backends.is_execution_backends_enabled",
            return_value=True,
        ),
        patch(
            "tools.execution_backends.get_backend_store",
            return_value=store,
        ),
    ):
        result = maybe_resolve_execution_task_id(
            "execute_code", task_id="code-task", session_id="session-c"
        )

    assert result is not None
    assert result.startswith("execution-backend:cnb-code:session:")
    assert _backend_id_from_task_id(result) == "cnb-code"


def test_process_uses_session_scoped_key(
    store: BackendStore,
) -> None:
    """process tool also gets session-scoped keys."""
    store.create_backend(
        _running_cnb_record("cnb-proc", host="proc.example")
    )
    store.set_current("session-d", "cnb-proc")

    with (
        patch(
            "tools.execution_backends.is_execution_backends_enabled",
            return_value=True,
        ),
        patch(
            "tools.execution_backends.get_backend_store",
            return_value=store,
        ),
    ):
        result = maybe_resolve_execution_task_id(
            "process", task_id="p-task", session_id="session-d"
        )

    assert result is not None
    assert result.startswith("execution-backend:cnb-proc:session:")
    assert _backend_id_from_task_id(result) == "cnb-proc"
