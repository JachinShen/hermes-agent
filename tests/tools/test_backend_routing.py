"""Tests for multi-backend routing, security isolation, and process ownership."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.execution_backends import (
    BackendRecord,
    BackendStore,
    BackendError,
    is_execution_backends_enabled,
    maybe_resolve_execution_task_id,
    resolve_execution_task_id,
)
from tools.process_registry import (
    ProcessSession,
    ProcessRegistry,
    _backend_id_from_task_id,  # noqa: PLC2701 — intentionally testing private API
)


# =========================================================================
# 1. Feature disabled behaviour
# =========================================================================


def test_is_execution_backends_disabled_by_default() -> None:
    """execution_backends.enabled defaults to False → no routing change."""
    assert not is_execution_backends_enabled()


def test_disabled_does_not_change_task_id() -> None:
    """When disabled, maybe_resolve_execution_task_id returns task_id unchanged."""
    with patch(
        "tools.execution_backends.is_execution_backends_enabled",
        return_value=False,
    ):
        result = maybe_resolve_execution_task_id(
            "terminal", task_id="child-task", session_id="session-a"
        )
    assert result == "child-task"


def test_disabled_does_not_change_task_id_for_file_tools() -> None:
    """Same for environment tools like read_file."""
    with patch(
        "tools.execution_backends.is_execution_backends_enabled",
        return_value=False,
    ):
        for name in ("read_file", "write_file", "patch", "search_files"):
            result = maybe_resolve_execution_task_id(
                name, task_id="task-x", session_id="session-x"
            )
            assert result == "task-x", f"{name} should return unchanged task_id"


def test_disabled_does_not_change_task_id_for_execute_code() -> None:
    with patch(
        "tools.execution_backends.is_execution_backends_enabled",
        return_value=False,
    ):
        result = maybe_resolve_execution_task_id(
            "execute_code", task_id="code-task", session_id="session-c"
        )
    assert result == "code-task"


def test_disabled_does_not_change_task_id_for_process() -> None:
    with patch(
        "tools.execution_backends.is_execution_backends_enabled",
        return_value=False,
    ):
        result = maybe_resolve_execution_task_id(
            "process", task_id="proc-task", session_id="session-p"
        )
    assert result == "proc-task"


def test_non_environment_tool_returns_task_id_unchanged() -> None:
    """Tools NOT in ENVIRONMENT_TOOL_NAMES always get their original task_id."""
    with patch(
        "tools.execution_backends.is_execution_backends_enabled",
        return_value=True,
    ):
        result = maybe_resolve_execution_task_id(
            "web_search", task_id="web-task", session_id="session-w"
        )
    assert result == "web-task"


def test_non_environment_tool_returns_unchanged_even_when_cnb_selected(
    store: BackendStore,
) -> None:
    """Even with a CNB backend active, non-environment tools keep raw task_id."""
    store.create_backend(_running_cnb("cnb-a", host="a.example"))
    store.set_current("session-a", "cnb-a")
    # Direct resolve_execution_task_id is NOT called for non-environment tools
    # because maybe_resolve_execution_task_id returns early.  Verify the early
    # return behaviour:
    with patch(
        "tools.execution_backends.is_execution_backends_enabled",
        return_value=True,
    ):
        result = maybe_resolve_execution_task_id(
            "vision_analyze", task_id="vis-task", session_id="session-a"
        )
    assert result == "vis-task"


# =========================================================================
# 2. CNB routing — dispatch_task_id
# =========================================================================


def test_cnb_routing_produces_backend_qualified_key(store: BackendStore) -> None:
    """When a CNB backend is active, resolve returns execution-backend:<id>."""
    store.create_backend(_running_cnb("my-backend", host="ssh.example"))
    store.set_current("session-a", "my-backend")
    key = resolve_execution_task_id(
        task_id="task-a", session_id="session-a", store=store
    )
    assert key == "execution-backend:my-backend"


def test_cnb_routing_registers_ssh_overrides(store: BackendStore, monkeypatch) -> None:
    """Verify the SSH overrides are registered correctly."""
    store.create_backend(_running_cnb("cnb-a", host="a.example"))
    store.set_current("session-a", "cnb-a")
    captured: dict = {}

    monkeypatch.setattr(
        "tools.terminal_tool.register_task_env_overrides",
        lambda task_id, overrides: captured.update(
            {"task_id": task_id, "overrides": overrides}
        ),
    )

    key = resolve_execution_task_id(
        task_id="child-task", session_id="session-a", store=store
    )

    assert key == "execution-backend:cnb-a"
    assert captured["task_id"] == key
    assert captured["overrides"] == {
        "env_type": "ssh",
        "cwd": "/workspace",
        "ssh_host": "a.example",
        "ssh_user": "dev",
        "ssh_port": 22,
        "ssh_key": "",
        "ssh_persistent": True,
        "ssh_sync_hermes_home": False,
    }


def test_cnb_routing_fails_closed_for_unready(store: BackendStore) -> None:
    """Non-running backend raises BackendError."""
    store.create_backend(_running_cnb("cnb-a", host="a.example"))
    store.set_current("session-a", "cnb-a")
    store.update_backend_status("cnb-a", "expired")
    with pytest.raises(BackendError, match="expired"):
        resolve_execution_task_id(
            task_id="t", session_id="session-a", store=store
        )


def test_cnb_routing_no_ssh_coordinates(store: BackendStore) -> None:
    """Backend without SSH coordinates raises BackendError."""
    record = _running_cnb("cnb-a", host="a.example")
    record.ssh_host = ""
    record.ssh_user = ""
    store.create_backend(record)
    store.set_current("session-a", "cnb-a")
    with pytest.raises(BackendError, match="no usable SSH"):
        resolve_execution_task_id(
            task_id="t", session_id="session-a", store=store
        )


def test_two_sessions_can_have_different_backends(store: BackendStore) -> None:
    """Verify per-session backend isolation."""
    store.create_backend(_running_cnb("cnb-a", host="a.example"))
    store.create_backend(_running_cnb("cnb-b", host="b.example"))
    store.set_current("session-a", "cnb-a")
    store.set_current("session-b", "cnb-b")

    key_a = resolve_execution_task_id(
        task_id="t", session_id="session-a", store=store
    )
    key_b = resolve_execution_task_id(
        task_id="t", session_id="session-b", store=store
    )
    assert key_a == "execution-backend:cnb-a"
    assert key_b == "execution-backend:cnb-b"


# =========================================================================
# 3. SSH sync_hermes_home=False
# =========================================================================


def test_ssh_sync_hermes_home_is_false_in_overrides(store: BackendStore) -> None:
    """SSH sync_hermes_home is False in execution backend overrides."""
    store.create_backend(_running_cnb("cnb-a", host="a.example"))
    store.set_current("session-a", "cnb-a")
    captured: dict = {}

    with patch(
        "tools.terminal_tool.register_task_env_overrides",
        lambda task_id, overrides: captured.update(overrides),
    ):
        resolve_execution_task_id(
            task_id="t", session_id="session-a", store=store
        )

    assert captured.get("ssh_sync_hermes_home") is False


def test_ssh_env_skips_filesync_when_sync_false(monkeypatch) -> None:
    """SSHEnvironment does NOT construct FileSyncManager when sync=False."""
    from tools.environments.ssh import SSHEnvironment

    ensure_remote_dirs_called = False
    sync_manager_constructed = False

    def _fake_detect_home(self) -> str:
        return "/home/dev"

    def _fake_ensure_remote_dirs(self) -> None:
        nonlocal ensure_remote_dirs_called
        ensure_remote_dirs_called = True

    original_file_sync = None
    try:
        from tools.environments import file_sync as _fs

        original_file_sync = _fs.FileSyncManager
    except ImportError:
        pass

    class FakeFileSyncManager:
        def __init__(self, *args, **kwargs):
            nonlocal sync_manager_constructed
            sync_manager_constructed = True

        def sync(self, force=False):
            pass

        def sync_back(self):
            pass

    # Patch the env's methods to avoid real SSH calls
    monkeypatch.setattr(
        "tools.environments.ssh._ensure_ssh_available", lambda: None
    )
    monkeypatch.setattr(
        "tools.environments.ssh.SSHEnvironment._detect_remote_home",
        _fake_detect_home,
    )
    monkeypatch.setattr(
        "tools.environments.ssh.SSHEnvironment._establish_connection",
        lambda self: None,
    )
    monkeypatch.setattr(
        "tools.environments.ssh.SSHEnvironment._ensure_remote_dirs",
        _fake_ensure_remote_dirs,
    )
    monkeypatch.setattr(
        "tools.environments.ssh.SSHEnvironment.init_session",
        lambda self: None,
    )
    if original_file_sync is not None:
        monkeypatch.setattr(
            "tools.environments.ssh.FileSyncManager", FakeFileSyncManager
        )

    env = SSHEnvironment(
        host="test.example",
        user="dev",
        sync_hermes_home=False,
    )

    assert not ensure_remote_dirs_called
    assert not sync_manager_constructed
    assert env._sync_manager is None

    # Cleanup should not crash when _sync_manager is None
    env.cleanup()


# =========================================================================
# 4. ProcessSession backend_id persistence and restore
# =========================================================================


class TestBackendIdFromTaskId:
    def test_local_backend_for_default_task(self) -> None:
        assert _backend_id_from_task_id("default") == "local"

    def test_local_backend_for_plain_task(self) -> None:
        assert _backend_id_from_task_id("child-task") == "local"

    def test_parses_cnb_from_execution_key(self) -> None:
        assert (
            _backend_id_from_task_id("execution-backend:cnb-a") == "cnb-a"
        )

    def test_parses_cnb_with_dots_and_hyphens(self) -> None:
        assert (
            _backend_id_from_task_id("execution-backend:my.cnb-backend-01")
            == "my.cnb-backend-01"
        )

    def test_prefix_case_sensitive_lowercase_only(self) -> None:
        assert (
            _backend_id_from_task_id("Execution-Backend:foo") == "local"
        )


def test_process_session_backend_id_in_spawn_local() -> None:
    """spawn_local sets backend_id from task_id."""
    reg = ProcessRegistry()
    session = reg.spawn_local(
        command="echo hello",
        task_id="execution-backend:cnb-a",
    )
    assert session.backend_id == "cnb-a"


def test_process_session_backend_id_in_spawn_via_env() -> None:
    """spawn_via_env sets backend_id from task_id."""
    reg = ProcessRegistry()

    class FakeEnv:
        def execute(self, *args, **kwargs):
            return {"output": "12345\n", "exit_code": 0, "pid": 12345}

        def get_temp_dir(self):
            return "/tmp"

    session = reg.spawn_via_env(
        FakeEnv(),
        command="echo hello",
        task_id="execution-backend:cnb-b",
    )
    assert session.backend_id == "cnb-b"


def test_process_session_backend_id_is_local_for_default() -> None:
    reg = ProcessRegistry()
    session = reg.spawn_local(command="true", task_id="default")
    assert session.backend_id == "local"


def test_checkpoint_serializes_backend_id(tmp_path) -> None:
    """Checkpoint file contains backend_id."""
    reg = ProcessRegistry()
    # Override checkpoint path
    from tools.process_registry import CHECKPOINT_PATH as _orig_path

    try:
        checkpoint_file = tmp_path / "processes.json"
        import tools.process_registry as pr

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(pr, "CHECKPOINT_PATH", checkpoint_file)

        session = reg.spawn_local(
            command="sleep 300",
            task_id="execution-backend:cnb-x",
        )
        # Mark as exited so it appears in checkpoint
        session.exited = True
        # The checkpoint only writes running processes, not finished ones
        # But we can write directly
        import json

        entries = [
            {
                "session_id": session.id,
                "command": session.command,
                "pid": session.pid,
                "pid_scope": session.pid_scope,
                "host_start_time": session.host_start_time,
                "cwd": session.cwd,
                "started_at": session.started_at,
                "task_id": session.task_id,
                "backend_id": session.backend_id,
                "session_key": session.session_key,
            }
        ]
        from utils import atomic_json_write

        atomic_json_write(checkpoint_file, entries)

        # Recover
        recovered_count = reg.recover_from_checkpoint()
        assert recovered_count == 1
        # Find recovered session
        recovered = None
        with reg._lock:
            for s in reg._running.values():
                if s.id == session.id:
                    recovered = s
                    break
        assert recovered is not None
        assert recovered.backend_id == "cnb-x"

        monkeypatch.undo()
    finally:
        pass


def test_kill_processes_correct_backend_id_scoping() -> None:
    """Kill_all filters by task_id, not backend_id.

    Fake ProcessSession objects lack real OS process handles so the
    real kill_process path returns error (can't kill a phantom).  We
    patch kill_process for fake-only sessions to verify the filtering
    logic without relaxing the security of the real kill path.
    """
    reg = ProcessRegistry()
    s1 = ProcessSession(
        id="proc_aaa", command="sleep 100", task_id="execution-backend:cnb-a",
        backend_id="cnb-a",
    )
    s2 = ProcessSession(
        id="proc_bbb", command="sleep 200", task_id="execution-backend:cnb-b",
        backend_id="cnb-b",
    )
    with reg._lock:
        reg._running["proc_aaa"] = s1
        reg._running["proc_bbb"] = s2

    # Wrap kill_process to succeed for sessions without real handles
    original_kill = reg.kill_process

    def _tracked_kill(session_id, **kwargs):
        session = reg.get(session_id)
        if session and not session.process and not session._pty and not session.env_ref:
            session.exited = True
            return {"status": "killed", "session_id": session_id}
        return original_kill(session_id, **kwargs)

    with patch.object(reg, "kill_process", _tracked_kill):
        killed = reg.kill_all(task_id="execution-backend:cnb-a")
    assert killed == 1
    assert s1.exited
    assert not s2.exited


# =========================================================================
# 5. Middleware/hooks retain original task/session (integration-style)
# =========================================================================


def test_handle_function_call_passes_original_ids_to_middleware() -> None:
    """Verify that middleware receives original (not execution) task_id.

    This test patches the middleware to capture the IDs it receives,
    and patches registry.dispatch to return immediately.
    """
    from model_tools import handle_function_call

    captured: dict = {}
    original_task = "original-task"
    original_session = "original-session"

    def _fake_middleware(
        name, args, dispatch, *, task_id, session_id, **kwargs
    ):
        captured["middleware_task_id"] = task_id
        captured["middleware_session_id"] = session_id
        return dispatch(args)

    def _fake_dispatch(name, args, **kwargs):
        captured["dispatch_task_id"] = kwargs.get("task_id")
        captured["dispatch_session_id"] = kwargs.get("session_id")
        return json.dumps({"ok": True})

    with patch(
        "tools.execution_backends.maybe_resolve_execution_task_id",
        return_value="execution-backend:cnb-a",
    ), patch(
        "hermes_cli.middleware.run_tool_execution_middleware",
        _fake_middleware,
    ), patch(
        "model_tools.registry.dispatch", _fake_dispatch
    ), patch(
        "model_tools._emit_post_tool_call_hook", lambda **kw: None
    ):
        result = handle_function_call(
            "terminal",
            {"command": "echo hi"},
            task_id=original_task,
            session_id=original_session,
        )

    # Middleware/hooks get ORIGINAL ids
    assert captured["middleware_task_id"] == original_task
    assert captured["middleware_session_id"] == original_session
    # Dispatch gets the EXECUTION key
    assert captured["dispatch_task_id"] == "execution-backend:cnb-a"
    assert captured["dispatch_session_id"] == original_session


def test_handle_function_call_non_environment_tool_preserves_task_id() -> None:
    """Non-environment tools do NOT change the dispatch task_id."""
    from model_tools import handle_function_call

    captured: dict = {}

    def _fake_middleware(
        name, args, dispatch, *, task_id, session_id, **kwargs
    ):
        captured["middleware_task_id"] = task_id
        return dispatch(args)

    def _fake_dispatch(name, args, **kwargs):
        captured["dispatch_task_id"] = kwargs.get("task_id")
        return json.dumps({"ok": True})

    with patch(
        "tools.execution_backends.maybe_resolve_execution_task_id",
        wraps=lambda fn, **kw: kw.get("task_id"),
    ), patch(
        "hermes_cli.middleware.run_tool_execution_middleware",
        _fake_middleware,
    ), patch(
        "model_tools.registry.dispatch", _fake_dispatch
    ), patch(
        "model_tools._emit_post_tool_call_hook", lambda **kw: None
    ):
        handle_function_call(
            "web_search",
            {"query": "hello"},
            task_id="original-task",
            session_id="original-session",
        )

    assert captured["dispatch_task_id"] == "original-task"


# =========================================================================
# 6. execute_code reads task overrides for env_type
# =========================================================================


def test_execute_code_resolves_task_overrides_for_env_type() -> None:
    """execute_code should use task-level overrides to determine env_type.

    When a CNB backend is active, env_type should be 'ssh' even though
    the global TERMINAL_ENV is 'local'.
    """
    # Patch the config reading and override resolution
    from tools.terminal_tool import _TASK_ENV_CONFIG_KEYS

    overrides = {
        "env_type": "ssh",
        "ssh_host": "cnb-host",
        "ssh_user": "dev",
        "ssh_sync_hermes_home": False,
    }

    with patch(
        "tools.terminal_tool.resolve_task_overrides",
        return_value=overrides,
    ), patch(
        "tools.code_execution_tool._execute_remote",
        return_value=json.dumps({"status": "ok", "output": ""}),
    ), patch(
        "tools.code_execution_tool.SANDBOX_AVAILABLE", True,
    ), patch(
        "tools.code_execution_tool._load_config",
        return_value={"timeout": 300, "max_tool_calls": 50},
    ), patch(
        "tools.approval.check_execute_code_guard",
        return_value={"approved": True},
    ):
        from tools.code_execution_tool import execute_code

        result = execute_code(
            code="print('hello')",
            task_id="execution-backend:cnb-a",
        )
        parsed = json.loads(result)
        assert parsed.get("status") == "ok"


def test_execute_code_local_path_still_works_without_overrides() -> None:
    """When no overrides exist, execute_code still takes local path."""
    import io

    class _FakeProc:
        pid = 99999
        returncode = 0
        stdout = io.BytesIO(b"hello from local\n")
        stderr = io.BytesIO(b"")

        def poll(self):
            return 0  # already exited

        def wait(self, timeout=None):
            return 0

    with patch(
        "tools.terminal_tool.resolve_task_overrides",
        return_value={},
    ), patch(
        "subprocess.Popen",
        return_value=_FakeProc(),
    ), patch(
        "tools.code_execution_tool.SANDBOX_AVAILABLE", True,
    ), patch(
        "tools.code_execution_tool._load_config",
        return_value={"timeout": 300, "max_tool_calls": 50},
    ), patch(
        "tools.approval.check_execute_code_guard",
        return_value={"approved": True},
    ):
        from tools.code_execution_tool import execute_code

        result = execute_code(code="print('hello')")
        parsed = json.loads(result)
        # Local path returns "success" status, not "ok"
        assert parsed.get("status") == "success"


# =========================================================================
# 7. Registry dispatch receives backend-qualified task_id for CNB
# =========================================================================


def test_dispatch_receives_backend_qualified_task_id_for_env_tools() -> None:
    """registry.dispatch gets backend-qualified task_key for env tools."""
    from model_tools import handle_function_call

    captured: dict = {}

    def _fake_middleware(
        name, args, dispatch, *, task_id, session_id, **kwargs
    ):
        return dispatch(args)

    def _fake_dispatch(name, args, **kwargs):
        captured["dispatch_task_id"] = kwargs.get("task_id")
        captured["dispatch_name"] = name
        return json.dumps({"ok": True})

    with patch(
        "tools.execution_backends.maybe_resolve_execution_task_id",
        return_value="execution-backend:cnb-a",
    ), patch(
        "hermes_cli.middleware.run_tool_execution_middleware",
        _fake_middleware,
    ), patch(
        "model_tools.registry.dispatch", _fake_dispatch
    ), patch(
        "model_tools._emit_post_tool_call_hook", lambda **kw: None
    ):
        handle_function_call(
            "terminal",
            {"command": "echo hi"},
            task_id="child-task",
            session_id="session-a",
        )

    assert captured["dispatch_task_id"] == "execution-backend:cnb-a"


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def store(tmp_path: Path) -> BackendStore:
    return BackendStore(tmp_path / "execution_backends.db")


def _running_cnb(backend_id: str, *, host: str) -> BackendRecord:
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
        created_at="2026-07-17T01:00:00+08:00",
    )
