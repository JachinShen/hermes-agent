from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tools.execution_backends import (
    BackendRecord,
    BackendStore,
    BackendLeaseMonitor,
    CNBCLIAdapter,
    BackendError,
    backend_tool,
    earliest_cnb_reclaim_at,
    parse_cnb_response,
    resolve_execution_task_id,
)


SHANGHAI = ZoneInfo("Asia/Shanghai")


@pytest.fixture
def store(tmp_path: Path) -> BackendStore:
    return BackendStore(tmp_path / "execution_backends.db")


def running_cnb(backend_id: str, *, host: str) -> BackendRecord:
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
        created_at="2026-07-17T19:00:00+08:00",
    )


def test_store_defaults_every_session_to_local(store: BackendStore) -> None:
    assert store.get_current("session-a").id == "local"
    assert store.get_current("session-b").id == "local"
    assert [record.id for record in store.list_backends()] == ["local"]


def test_two_sessions_can_select_different_backends(store: BackendStore) -> None:
    store.create_backend(running_cnb("cnb-a", host="a.example"))
    store.create_backend(running_cnb("cnb-b", host="b.example"))

    store.set_current("session-a", "cnb-a")
    store.set_current("session-b", "cnb-b")

    assert store.get_current("session-a").id == "cnb-a"
    assert store.get_current("session-b").id == "cnb-b"


def test_store_persists_registry_and_binding(tmp_path: Path) -> None:
    path = tmp_path / "execution_backends.db"
    first = BackendStore(path)
    first.create_backend(running_cnb("cnb-a", host="a.example"))
    first.set_current("session-a", "cnb-a")
    first.close()

    second = BackendStore(path)
    assert second.get_backend("cnb-a").ssh_host == "a.example"
    assert second.get_current("session-a").id == "cnb-a"


def test_delete_local_is_rejected(store: BackendStore) -> None:
    with pytest.raises(BackendError, match="local"):
        store.delete_backend("local")


def test_delete_selected_backend_resets_only_affected_bindings(store: BackendStore) -> None:
    store.create_backend(running_cnb("cnb-a", host="a.example"))
    store.create_backend(running_cnb("cnb-b", host="b.example"))
    store.set_current("session-a", "cnb-a")
    store.set_current("session-b", "cnb-b")

    store.delete_backend("cnb-a")

    assert store.get_current("session-a").id == "local"
    assert store.get_current("session-b").id == "cnb-b"


def test_router_preserves_task_id_for_local(store: BackendStore) -> None:
    assert resolve_execution_task_id(
        task_id="task-a", session_id="session-a", store=store
    ) == "task-a"


def test_router_uses_stable_backend_key_and_no_sync_ssh_override(
    store: BackendStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.create_backend(running_cnb("cnb-a", host="a.example"))
    store.set_current("session-a", "cnb-a")
    captured: dict[str, object] = {}

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


def test_router_fails_closed_for_unready_backend(store: BackendStore) -> None:
    record = running_cnb("cnb-a", host="a.example")
    store.create_backend(record)
    store.set_current("session-a", "cnb-a")
    store.update_backend_status("cnb-a", "expired")

    with pytest.raises(BackendError, match="expired"):
        resolve_execution_task_id(
            task_id="task-a", session_id="session-a", store=store
        )


def test_backend_tool_exposes_one_crud_surface(store: BackendStore) -> None:
    store.create_backend(running_cnb("cnb-a", host="a.example"))

    switched = json.loads(
        backend_tool(
            {"action": "update", "id": "cnb-a", "current": True},
            session_id="session-a",
            store=store,
        )
    )
    queried = json.loads(
        backend_tool(
            {"action": "get"}, session_id="session-a", store=store
        )
    )

    assert switched["current_backend"] == "cnb-a"
    assert queried["current_backend"] == "cnb-a"
    assert {item["id"] for item in queried["backends"]} == {"local", "cnb-a"}


def test_backend_tool_never_serializes_private_adapter_fields(store: BackendStore) -> None:
    record = running_cnb("cnb-a", host="a.example")
    record.metadata = {
        "access_url": "https://example.invalid",
        "token": "must-not-leak",
        "authorization": "must-not-leak",
    }
    store.create_backend(record)

    output = backend_tool(
        {"action": "get", "id": "cnb-a"},
        session_id="session-a",
        store=store,
    )

    assert "must-not-leak" not in output
    assert "token" not in output.lower()
    assert "authorization" not in output.lower()


def test_parse_cnb_response_requires_http_success_even_when_cli_succeeded() -> None:
    with pytest.raises(BackendError, match="status 400"):
        parse_cnb_response(
            json.dumps(
                {
                    "status": 400,
                    "data": {"error": {"message": "bad request"}},
                }
            )
        )


def test_cnb_adapter_requires_exact_unambiguous_workspace_match() -> None:
    envelope = {
        "status": 200,
        "data": {
            "list": [
                {
                    "sn": "one",
                    "pipeline_id": "p1",
                    "slug": "example/repo",
                    "branch": "main",
                    "status": "running",
                    "create_time": "2026-07-17T01:00:00+08:00",
                },
                {
                    "sn": "two",
                    "pipeline_id": "p2",
                    "slug": "example/repo",
                    "branch": "main",
                    "status": "running",
                    "create_time": "2026-07-17T02:00:00+08:00",
                },
            ]
        },
    }
    adapter = CNBCLIAdapter(runner=lambda _argv: json.dumps(envelope))

    with pytest.raises(BackendError, match="multiple"):
        adapter.find_workspace(repo="example/repo", branch="main")

    selected = adapter.find_workspace(
        repo="example/repo", branch="main", workspace_sn="two"
    )
    assert selected["sn"] == "two"


def test_cnb_adapter_parses_detail_into_secret_free_record() -> None:
    detail = {
        "status": 200,
        "data": {
            "sn": "sn-1",
            "pipelineId": "pipeline-1",
            "status": "running",
            "remoteSsh": "dev@ssh.example",
            "accessUrl": "https://webide.example",
            "token": "must-not-leak",
        },
    }
    adapter = CNBCLIAdapter(runner=lambda _argv: json.dumps(detail))

    record = adapter.get_backend_record(
        backend_id="cnb-a",
        repo="example/repo",
        branch="main",
        workspace_sn="sn-1",
        created_at="2026-07-17T19:00:00+08:00",
    )

    assert record.ssh_user == "dev"
    assert record.ssh_host == "ssh.example"
    assert record.metadata == {"access_url": "https://webide.example"}


@pytest.mark.parametrize(
    ("created_at", "expected"),
    [
        # Eight-hour threshold is inside the overnight window.
        (
            datetime(2026, 7, 17, 20, 30, tzinfo=SHANGHAI),
            datetime(2026, 7, 18, 4, 30, tzinfo=SHANGHAI),
        ),
        # Eight-hour threshold is after the window; the 18-hour cap wins.
        (
            datetime(2026, 7, 17, 23, 0, tzinfo=SHANGHAI),
            datetime(2026, 7, 18, 17, 0, tzinfo=SHANGHAI),
        ),
        # Eight-hour threshold is before 04:00; reclaim risk begins at 04:00.
        (
            datetime(2026, 7, 17, 19, 0, tzinfo=SHANGHAI),
            datetime(2026, 7, 18, 4, 0, tzinfo=SHANGHAI),
        ),
    ],
)
def test_earliest_cnb_reclaim_at(
    created_at: datetime, expected: datetime
) -> None:
    assert earliest_cnb_reclaim_at(created_at) == expected


def test_earliest_cnb_reclaim_requires_timezone_aware_input() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        earliest_cnb_reclaim_at(datetime(2026, 7, 17, 20, 0))


# ── BackendLeaseMonitor ──────────────────────────────────────────────────


def _cnb_backend(
    *,
    backend_id: str = "cnb-test",
    created_at: str,
    owner_session_id: str = "",
    host: str = "ssh.example",
) -> BackendRecord:
    return BackendRecord(
        id=backend_id,
        driver="cnb",
        status="running",
        repo="example/repo",
        branch="main",
        workspace_sn="sn-test",
        pipeline_id="pipeline-test",
        ssh_host=host,
        ssh_user="dev",
        ssh_port=22,
        cwd="/workspace",
        created_at=created_at,
        owner_session_id=owner_session_id,
    )


def test_monitor_local_backend_returns_ok_no_events(store: BackendStore) -> None:
    monitor = BackendLeaseMonitor(store)
    result = monitor.evaluate("local")
    assert result["status"] == "ok"
    assert result["events_fired"] == []
    assert result["deadline"] is None


def test_monitor_unknown_backend_returns_unknown(store: BackendStore) -> None:
    monitor = BackendLeaseMonitor(store)
    result = monitor.evaluate("does-not-exist")
    assert result["status"] == "unknown"
    assert result["events_fired"] == []


def test_monitor_non_cnb_backend_skips_lease(store: BackendStore) -> None:
    record = BackendRecord(id="custom", driver="custom", status="running")
    store._conn.execute(
        "INSERT INTO execution_backends "
        "(id, driver, status, created_at, updated_at, metadata_json) "
        "VALUES (?, ?, ?, '2026-07-17T10:00:00+00:00', '2026-07-17T10:00:00+00:00', '{}')",
        (record.id, record.driver, record.status),
    )
    monitor = BackendLeaseMonitor(store)
    result = monitor.evaluate("custom")
    assert result["status"] == "ok"
    assert result["events_fired"] == []
    assert result["deadline"] is None


def test_monitor_cnb_far_from_deadline_returns_ok(store: BackendStore) -> None:
    store.create_backend(
        _cnb_backend(
            created_at="2026-07-17T19:00:00+08:00",
        )
    )
    monitor = BackendLeaseMonitor(store)
    # Created at 19:00, deadline = min(
    #   19:00+18h=13:00 next day, 04:00 next day (overnight)
    # ) = 04:00 on July 18.
    # "now" at 21:00 on July 17 → 7 hours before deadline → "ok".
    now = datetime(2026, 7, 17, 21, 0, tzinfo=SHANGHAI)
    result = monitor.evaluate("cnb-test", now=now)
    assert result["status"] == "ok"
    assert result["events_fired"] == []
    assert result["deadline"] is not None


def test_monitor_warning_zone_fires_reclaim_warning(store: BackendStore) -> None:
    store.create_backend(
        _cnb_backend(
            created_at="2026-07-17T19:00:00+08:00",
        )
    )
    monitor = BackendLeaseMonitor(store)
    # Deadline = 2026-07-18 04:00 (next overnight window).
    # "Now" = 2026-07-18 03:40 → 20 min before deadline → within warning zone (30-10 min).
    now = datetime(2026, 7, 18, 3, 40, tzinfo=SHANGHAI)
    result = monitor.evaluate("cnb-test", now=now)
    assert result["status"] == "warning"
    assert result["events_fired"] == ["backend.reclaim_warning"]

def test_monitor_critical_zone_fires_critical(store: BackendStore) -> None:
    store.create_backend(
        _cnb_backend(
            created_at="2026-07-17T19:00:00+08:00",
        )
    )
    monitor = BackendLeaseMonitor(store)
    # Deadline = 2026-07-18 04:00.  "Now" = 03:55 → 5 min before deadline.
    now = datetime(2026, 7, 18, 3, 55, tzinfo=SHANGHAI)
    result = monitor.evaluate("cnb-test", now=now)
    assert result["status"] == "critical"
    assert result["events_fired"] == ["backend.reclaim_critical"]

def test_monitor_expired_fires_expired(store: BackendStore) -> None:
    store.create_backend(
        _cnb_backend(
            created_at="2026-07-17T19:00:00+08:00",
        )
    )
    monitor = BackendLeaseMonitor(store)
    # Deadline = 04:00 on July 18.  "Now" = 04:05 → past deadline.
    now = datetime(2026, 7, 18, 4, 5, tzinfo=SHANGHAI)
    result = monitor.evaluate("cnb-test", now=now)
    assert result["status"] == "expired"
    assert result["events_fired"] == ["backend.expired"]


def test_monitor_fires_only_one_event_type_per_evaluation(store: BackendStore) -> None:
    """When multiple thresholds are crossed, the highest-severity fires."""
    store.create_backend(
        _cnb_backend(
            created_at="2026-07-17T19:00:00+08:00",
        )
    )
    monitor = BackendLeaseMonitor(store)
    # Deadline = 04:00 July 18.  Expired has highest priority → fires expired, not critical.
    now = datetime(2026, 7, 18, 4, 5, tzinfo=SHANGHAI)
    result = monitor.evaluate("cnb-test", now=now)
    assert result["status"] == "expired"
    assert result["events_fired"] == ["backend.expired"]


def test_monitor_dedup_returns_empty_events_fired_on_second_call(store: BackendStore) -> None:
    """Persistent dedup: second evaluate with same clock should fire nothing."""
    store.create_backend(
        _cnb_backend(
            created_at="2026-07-17T19:00:00+08:00",
        )
    )
    monitor = BackendLeaseMonitor(store)
    now = datetime(2026, 7, 18, 3, 40, tzinfo=SHANGHAI)

    first = monitor.evaluate("cnb-test", now=now)
    assert first["events_fired"] == ["backend.reclaim_warning"]

    second = monitor.evaluate("cnb-test", now=now)
    assert second["status"] == "warning"
    assert second["events_fired"] == []  # already emitted


def test_monitor_dedup_survives_store_reopen(tmp_path: Path) -> None:
    """Persistent dedup across store instances (process restart)."""
    path = tmp_path / "execution_backends.db"
    first_store = BackendStore(path)
    first_store.create_backend(
        _cnb_backend(
            created_at="2026-07-17T19:00:00+08:00",
        )
    )
    first = BackendLeaseMonitor(first_store)
    now = datetime(2026, 7, 18, 3, 40, tzinfo=SHANGHAI)
    assert first.evaluate("cnb-test", now=now)["events_fired"] == [
        "backend.reclaim_warning"
    ]
    first_store.close()

    second_store = BackendStore(path)
    second = BackendLeaseMonitor(second_store)
    result = second.evaluate("cnb-test", now=now)
    assert result["status"] == "warning"
    assert result["events_fired"] == []  # dedup from DB


def test_monitor_carries_owner_session_id(store: BackendStore) -> None:
    store.create_backend(
        _cnb_backend(
            created_at="2026-07-17T19:00:00+08:00",
            owner_session_id="session-abc",
        )
    )
    monitor = BackendLeaseMonitor(store)
    now = datetime(2026, 7, 18, 4, 5, tzinfo=SHANGHAI)
    result = monitor.evaluate("cnb-test", now=now)
    assert result["owner_session_id"] == "session-abc"
    assert result["events_fired"] == ["backend.expired"]


def test_monitor_owner_session_stored_in_event_table(store: BackendStore) -> None:
    """Verify the owner_session_id is persisted alongside the event."""
    store.create_backend(
        _cnb_backend(
            created_at="2026-07-17T19:00:00+08:00",
            owner_session_id="session-xyz",
        )
    )
    monitor = BackendLeaseMonitor(store)
    now = datetime(2026, 7, 18, 3, 40, tzinfo=SHANGHAI)
    monitor.evaluate("cnb-test", now=now)

    row = store._conn.execute(
        "SELECT owner_session_id FROM execution_backend_events "
        "WHERE backend_id = ? AND event_type = ?",
        ("cnb-test", "backend.reclaim_warning"),
    ).fetchone()
    assert row is not None
    assert row["owner_session_id"] == "session-xyz"


def test_monitor_expired_transitions_from_warning_via_critical(store: BackendStore) -> None:
    """Simulate clock progression through all three phases."""
    store.create_backend(
        _cnb_backend(
            created_at="2026-07-17T19:00:00+08:00",
            owner_session_id="session-progress",
        )
    )
    monitor = BackendLeaseMonitor(store)
    deadline = datetime(2026, 7, 18, 4, 0, tzinfo=SHANGHAI)

    # Phase 1: ok (7 hours before deadline)
    r1 = monitor.evaluate("cnb-test", now=datetime(2026, 7, 17, 21, 0, tzinfo=SHANGHAI))
    assert r1["status"] == "ok"
    assert r1["events_fired"] == []

    # Phase 2: warning (20 min before deadline)
    r2 = monitor.evaluate("cnb-test", now=deadline - timedelta(minutes=20))
    assert r2["status"] == "warning"
    assert r2["events_fired"] == ["backend.reclaim_warning"]

    # Phase 3: critical (5 min before deadline)
    r3 = monitor.evaluate("cnb-test", now=deadline - timedelta(minutes=5))
    assert r3["status"] == "critical"
    assert r3["events_fired"] == ["backend.reclaim_critical"]

    # Phase 4: expired
    r4 = monitor.evaluate("cnb-test", now=deadline + timedelta(minutes=5))
    assert r4["status"] == "expired"
    assert r4["events_fired"] == ["backend.expired"]

    # Phase 5: re-evaluate expired → no duplicate events
    r5 = monitor.evaluate("cnb-test", now=deadline + timedelta(hours=1))
    assert r5["status"] == "expired"
    assert r5["events_fired"] == []
