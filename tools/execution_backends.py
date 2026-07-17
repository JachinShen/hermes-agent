"""Local control plane for session-selectable execution backends.

The agent loop, conversation state, model requests, memory, and skills remain on
this host.  This module stores only execution-backend coordinates and resolves
an environment task key for tools that touch a filesystem or run commands.
"""

from __future__ import annotations

import json
import re
import shlex
import sqlite3
import subprocess
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

from hermes_cli.config import get_hermes_home, load_config


CNB_TIMEZONE = ZoneInfo("Asia/Shanghai")
CNB_MAX_HEARTBEAT_LIFETIME = timedelta(hours=18)
CNB_OVERNIGHT_MIN_AGE = timedelta(hours=8)
CNB_OVERNIGHT_START = time(4, 0)
CNB_OVERNIGHT_END = time(6, 0)
_BACKEND_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SENSITIVE_METADATA_KEYS = frozenset(
    {"authorization", "cookie", "password", "secret", "token"}
)

ENVIRONMENT_TOOL_NAMES = frozenset(
    {
        "terminal",
        "process",
        "read_file",
        "search_files",
        "write_file",
        "patch",
        "execute_code",
    }
)


class BackendError(RuntimeError):
    """A safe, user-facing execution-backend failure."""


@dataclass
class BackendRecord:
    id: str
    driver: str
    status: str = "running"
    repo: str = ""
    branch: str = ""
    workspace_sn: str = ""
    pipeline_id: str = ""
    ssh_host: str = ""
    ssh_user: str = ""
    ssh_port: int = 22
    ssh_key: str = ""
    cwd: str = ""
    created_at: str = ""
    updated_at: str = ""
    owner_session_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def local(cls) -> "BackendRecord":
        return cls(id="local", driver="local", status="running")

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("ssh_key", None)
        value["metadata"] = _sanitize_metadata(value.get("metadata", {}))
        return value


def _sanitize_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize_metadata(item)
            for key, item in value.items()
            if str(key).lower() not in _SENSITIVE_METADATA_KEYS
        }
    if isinstance(value, list):
        return [_sanitize_metadata(item) for item in value]
    return value


def _utc_now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _validate_backend_id(backend_id: str) -> str:
    backend_id = str(backend_id or "").strip()
    if not _BACKEND_ID_RE.fullmatch(backend_id):
        raise BackendError(
            "backend id must be 1-64 characters using letters, numbers, '.', '_' or '-'"
        )
    return backend_id


class BackendStore:
    """Profile-local SQLite registry for backends and session bindings."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or (get_hermes_home() / "execution_backends.db"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS execution_backends (
                    id TEXT PRIMARY KEY,
                    driver TEXT NOT NULL,
                    status TEXT NOT NULL,
                    repo TEXT NOT NULL DEFAULT '',
                    branch TEXT NOT NULL DEFAULT '',
                    workspace_sn TEXT NOT NULL DEFAULT '',
                    pipeline_id TEXT NOT NULL DEFAULT '',
                    ssh_host TEXT NOT NULL DEFAULT '',
                    ssh_user TEXT NOT NULL DEFAULT '',
                    ssh_port INTEGER NOT NULL DEFAULT 22,
                    ssh_key TEXT NOT NULL DEFAULT '',
                    cwd TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    owner_session_id TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS execution_backend_bindings (
                    session_key TEXT PRIMARY KEY,
                    backend_id TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS execution_backend_events (
                    backend_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    emitted_at TEXT NOT NULL,
                    owner_session_id TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (backend_id, event_type)
                );
                try:
                    self._conn.execute(
                        "ALTER TABLE execution_backend_events ADD COLUMN owner_session_id TEXT NOT NULL DEFAULT ''"
                    )
                except sqlite3.OperationalError:
                    pass  # column already exists
                """
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def create_backend(self, record: BackendRecord) -> BackendRecord:
        record.id = _validate_backend_id(record.id)
        if record.id == "local":
            raise BackendError("local is a built-in backend and cannot be created")
        if record.driver != "cnb":
            raise BackendError(f"unsupported backend driver: {record.driver}")
        if not record.created_at:
            record.created_at = _utc_now_iso()
        record.updated_at = _utc_now_iso()
        record.metadata = _sanitize_metadata(record.metadata)
        values = asdict(record)
        metadata_json = json.dumps(values.pop("metadata"), ensure_ascii=False)
        with self._lock, self._conn:
            try:
                self._conn.execute(
                    """
                    INSERT INTO execution_backends (
                        id, driver, status, repo, branch, workspace_sn,
                        pipeline_id, ssh_host, ssh_user, ssh_port, ssh_key,
                        cwd, created_at, updated_at, owner_session_id, metadata_json
                    ) VALUES (
                        :id, :driver, :status, :repo, :branch, :workspace_sn,
                        :pipeline_id, :ssh_host, :ssh_user, :ssh_port, :ssh_key,
                        :cwd, :created_at, :updated_at, :owner_session_id, :metadata_json
                    )
                    """,
                    {**values, "metadata_json": metadata_json},
                )
            except sqlite3.IntegrityError as exc:
                raise BackendError(f"backend already exists: {record.id}") from exc
        return record

    def mark_event_once(self, backend_id: str, event_type: str, *, owner_session_id: str = "") -> bool:
        """Persist event de-duplication across monitor restarts.

        Returns True if the event was newly inserted (first time for this
        backend+event_type combination), or False if it already existed.
        When *owner_session_id* is non-empty it is stored alongside the event
        so downstream consumers can identify the owning conversation.
        """
        backend_id = _validate_backend_id(backend_id)
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                INSERT OR IGNORE INTO execution_backend_events(
                    backend_id, event_type, emitted_at, owner_session_id
                ) VALUES (?, ?, ?, ?)
                """,
                (backend_id, str(event_type), _utc_now_iso(), str(owner_session_id or "")),
            )
        return cursor.rowcount == 1

    def update_backend_status(self, backend_id: str, status: str) -> BackendRecord:
        backend_id = _validate_backend_id(backend_id)
        if backend_id == "local":
            raise BackendError("local backend status is immutable")
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "UPDATE execution_backends SET status = ?, updated_at = ? WHERE id = ?",
                (str(status), _utc_now_iso(), backend_id),
            )
            if cursor.rowcount != 1:
                raise BackendError(f"unknown backend: {backend_id}")
        return self.get_backend(backend_id)

    def get_backend(self, backend_id: str) -> BackendRecord:
        backend_id = _validate_backend_id(backend_id)
        if backend_id == "local":
            return BackendRecord.local()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM execution_backends WHERE id = ?", (backend_id,)
            ).fetchone()
        if row is None:
            raise BackendError(f"unknown backend: {backend_id}")
        return self._row_to_record(row)

    def list_backends(self) -> list[BackendRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM execution_backends ORDER BY id"
            ).fetchall()
        return [BackendRecord.local(), *(self._row_to_record(row) for row in rows)]

    def set_current(self, session_key: str, backend_id: str) -> BackendRecord:
        session_key = str(session_key or "default")
        record = self.get_backend(backend_id)
        if record.status != "running":
            raise BackendError(
                f"backend {record.id} is not ready (status={record.status})"
            )
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO execution_backend_bindings(session_key, backend_id, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_key) DO UPDATE SET
                    backend_id = excluded.backend_id,
                    updated_at = excluded.updated_at
                """,
                (session_key, record.id, _utc_now_iso()),
            )
        return record

    def get_current(self, session_key: str) -> BackendRecord:
        session_key = str(session_key or "default")
        with self._lock:
            row = self._conn.execute(
                "SELECT backend_id FROM execution_backend_bindings WHERE session_key = ?",
                (session_key,),
            ).fetchone()
        if row is None:
            return BackendRecord.local()
        try:
            return self.get_backend(str(row["backend_id"]))
        except BackendError:
            return BackendRecord.local()

    def delete_backend(self, backend_id: str) -> BackendRecord:
        backend_id = _validate_backend_id(backend_id)
        if backend_id == "local":
            raise BackendError("local backend cannot be deleted")
        record = self.get_backend(backend_id)
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM execution_backend_bindings WHERE backend_id = ?",
                (backend_id,),
            )
            self._conn.execute(
                "DELETE FROM execution_backend_events WHERE backend_id = ?",
                (backend_id,),
            )
            self._conn.execute(
                "DELETE FROM execution_backends WHERE id = ?", (backend_id,)
            )
        return record

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> BackendRecord:
        metadata = json.loads(row["metadata_json"] or "{}")
        return BackendRecord(
            id=row["id"],
            driver=row["driver"],
            status=row["status"],
            repo=row["repo"],
            branch=row["branch"],
            workspace_sn=row["workspace_sn"],
            pipeline_id=row["pipeline_id"],
            ssh_host=row["ssh_host"],
            ssh_user=row["ssh_user"],
            ssh_port=int(row["ssh_port"]),
            ssh_key=row["ssh_key"],
            cwd=row["cwd"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            owner_session_id=row["owner_session_id"],
            metadata=_sanitize_metadata(metadata),
        )


def is_execution_backends_enabled() -> bool:
    try:
        section = load_config().get("execution_backends", {})
    except Exception:
        return False
    return isinstance(section, dict) and bool(section.get("enabled", False))


_store_cache: dict[Path, BackendStore] = {}
_store_cache_lock = threading.Lock()


def get_backend_store() -> BackendStore:
    path = get_hermes_home() / "execution_backends.db"
    with _store_cache_lock:
        store = _store_cache.get(path)
        if store is None:
            store = BackendStore(path)
            _store_cache[path] = store
        return store


def _session_key(session_id: str | None, task_id: str | None) -> str:
    return str(session_id or task_id or "default")


def resolve_execution_task_id(
    *,
    task_id: str | None,
    session_id: str | None,
    store: BackendStore | None = None,
) -> str:
    """Resolve and prepare the environment key for the session's backend."""
    backend_store = store or get_backend_store()
    record = backend_store.get_current(_session_key(session_id, task_id))
    if record.id == "local":
        return str(task_id or "default")
    if record.status != "running":
        raise BackendError(
            f"backend {record.id} is not ready (status={record.status})"
        )
    if record.driver != "cnb":
        raise BackendError(f"unsupported backend driver: {record.driver}")
    if not record.ssh_host or not record.ssh_user:
        raise BackendError(f"backend {record.id} has no usable SSH coordinates")

    execution_key = f"execution-backend:{record.id}"
    from tools.terminal_tool import register_task_env_overrides

    register_task_env_overrides(
        execution_key,
        {
            "env_type": "ssh",
            "cwd": record.cwd or "/workspace",
            "ssh_host": record.ssh_host,
            "ssh_user": record.ssh_user,
            "ssh_port": record.ssh_port,
            "ssh_key": record.ssh_key,
            "ssh_persistent": True,
            "ssh_sync_hermes_home": False,
        },
    )
    return execution_key


def maybe_resolve_execution_task_id(
    function_name: str,
    *,
    task_id: str | None,
    session_id: str | None,
) -> str | None:
    if function_name not in ENVIRONMENT_TOOL_NAMES:
        return task_id
    if not is_execution_backends_enabled():
        return task_id
    return resolve_execution_task_id(task_id=task_id, session_id=session_id)


def parse_cnb_response(raw: str | bytes | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BackendError("CNB CLI returned invalid JSON") from exc
    else:
        payload = raw
    if not isinstance(payload, dict):
        raise BackendError("CNB CLI returned a non-object response")
    try:
        status = int(payload.get("status"))
    except (TypeError, ValueError) as exc:
        raise BackendError("CNB CLI response has no numeric status") from exc
    if not 200 <= status < 300:
        raise BackendError(f"CNB API returned status {status}")
    return payload


def _default_cnb_runner(argv: list[str]) -> str:
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=60,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise BackendError(f"CNB CLI execution failed: {type(exc).__name__}") from exc
    raw = result.stdout.strip() or result.stderr.strip()
    if result.returncode != 0:
        raise BackendError(f"CNB CLI exited with status {result.returncode}")
    return raw


def _first_value(mapping: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = mapping.get(name)
        if value not in (None, ""):
            return value
    return None


def _parse_remote_ssh(value: str) -> tuple[str, str, int]:
    value = str(value or "").strip()
    if not value:
        raise BackendError("CNB workspace detail has no remoteSsh target")
    tokens = shlex.split(value)
    port = 22
    target = ""
    if tokens and tokens[0] == "ssh":
        index = 1
        while index < len(tokens):
            token = tokens[index]
            if token in {"-p", "--port"} and index + 1 < len(tokens):
                port = int(tokens[index + 1])
                index += 2
                continue
            if token.startswith("-"):
                index += 2 if token in {"-i", "-o", "-F", "-J"} else 1
                continue
            target = token
            index += 1
    else:
        target = tokens[-1] if tokens else value
    if "@" not in target:
        raise BackendError("CNB remoteSsh target must include user@host")
    user, host = target.rsplit("@", 1)
    if not user or not host:
        raise BackendError("CNB remoteSsh target is incomplete")
    return user, host, port


class CNBCLIAdapter:
    """Small, injectable adapter over the official CNB workspace CLI."""

    def __init__(self, runner: Callable[[list[str]], str] | None = None):
        self._runner = runner or _default_cnb_runner

    def _run(self, argv: list[str]) -> dict[str, Any]:
        return parse_cnb_response(self._runner(argv))

    def list_workspaces(self, *, repo: str, branch: str) -> list[dict[str, Any]]:
        payload = self._run(
            [
                "cnb",
                "workspace",
                "list-workspaces",
                "--slug",
                repo,
                "--branch",
                branch,
                "--page-size",
                "20",
                "--verbose",
            ]
        )
        data = payload.get("data")
        rows = data.get("list", []) if isinstance(data, dict) else []
        if not isinstance(rows, list):
            raise BackendError("CNB workspace list has an invalid data.list field")
        return [row for row in rows if isinstance(row, dict)]

    def find_workspace(
        self, *, repo: str, branch: str, workspace_sn: str = ""
    ) -> dict[str, Any]:
        matches = [
            row
            for row in self.list_workspaces(repo=repo, branch=branch)
            if row.get("slug") == repo
            and row.get("branch") == branch
            and (not workspace_sn or str(row.get("sn")) == workspace_sn)
        ]
        if not matches:
            raise BackendError("no exact CNB workspace match")
        if len(matches) > 1:
            raise BackendError("multiple exact CNB workspaces match; provide workspace_sn")
        return matches[0]

    def get_backend_record(
        self,
        *,
        backend_id: str,
        repo: str,
        branch: str,
        workspace_sn: str,
        created_at: str = "",
        pipeline_id: str = "",
    ) -> BackendRecord:
        payload = self._run(
            [
                "cnb",
                "workspace",
                "get-workspace-detail",
                "--repo",
                repo,
                "--sn",
                workspace_sn,
                "--verbose",
            ]
        )
        data = payload.get("data")
        if not isinstance(data, dict):
            raise BackendError("CNB workspace detail has no data object")
        remote_ssh = _first_value(data, "remoteSsh", "remote_ssh")
        user, host, port = _parse_remote_ssh(str(remote_ssh or ""))
        access_url = _first_value(data, "accessUrl", "access_url", "webideUrl")
        status = str(_first_value(data, "status") or "unknown").lower()
        if status != "running":
            raise BackendError(f"CNB workspace is not running (status={status})")
        return BackendRecord(
            id=_validate_backend_id(backend_id),
            driver="cnb",
            status=status,
            repo=repo,
            branch=branch,
            workspace_sn=str(_first_value(data, "sn") or workspace_sn),
            pipeline_id=str(
                _first_value(data, "pipelineId", "pipeline_id") or pipeline_id
            ),
            ssh_host=host,
            ssh_user=user,
            ssh_port=port,
            cwd="/workspace",
            created_at=created_at or _utc_now_iso(),
            metadata={"access_url": access_url} if access_url else {},
        )

    def create_backend(
        self, *, backend_id: str, repo: str, branch: str
    ) -> BackendRecord:
        payload = self._run(
            [
                "cnb",
                "workspace",
                "start-workspace",
                "--repo",
                repo,
                "--branch",
                branch,
                "--verbose",
            ]
        )
        data = payload.get("data")
        data = data if isinstance(data, dict) else {}
        workspace_sn = str(_first_value(data, "sn", "workspaceSn") or "")
        pipeline_id = str(_first_value(data, "pipelineId", "pipeline_id") or "")
        created_at = str(_first_value(data, "create_time", "createTime") or _utc_now_iso())
        if not workspace_sn:
            row = self.find_workspace(repo=repo, branch=branch)
            workspace_sn = str(row.get("sn") or "")
            pipeline_id = str(row.get("pipeline_id") or pipeline_id)
            created_at = str(row.get("create_time") or created_at)
        if not workspace_sn:
            raise BackendError("CNB start response has no workspace serial number")
        return self.get_backend_record(
            backend_id=backend_id,
            repo=repo,
            branch=branch,
            workspace_sn=workspace_sn,
            pipeline_id=pipeline_id,
            created_at=created_at,
        )

    def stop_backend(self, record: BackendRecord) -> None:
        selector: list[str]
        if record.pipeline_id:
            selector = ["--pipelineId", record.pipeline_id]
        elif record.workspace_sn:
            selector = ["--sn", record.workspace_sn]
        else:
            raise BackendError(f"backend {record.id} has no CNB stop coordinate")
        self._run(
            ["cnb", "workspace", "workspace-stop", *selector, "--verbose"]
        )


def _backend_result(
    *, action: str, session_key: str, store: BackendStore, records: Iterable[BackendRecord]
) -> str:
    current = store.get_current(session_key)
    return json.dumps(
        {
            "action": action,
            "current_backend": current.id,
            "backends": [record.public_dict() for record in records],
        },
        ensure_ascii=False,
    )


def backend_tool(
    args: dict[str, Any],
    *,
    task_id: str | None = None,
    session_id: str | None = None,
    store: BackendStore | None = None,
    adapter: CNBCLIAdapter | None = None,
) -> str:
    backend_store = store or get_backend_store()
    cnb = adapter or CNBCLIAdapter()
    session_key = _session_key(session_id, task_id)
    action = str(args.get("action") or "").strip().lower()
    backend_id = str(args.get("id") or "").strip()

    if action == "get":
        records = (
            [backend_store.get_backend(backend_id)]
            if backend_id
            else backend_store.list_backends()
        )
        return _backend_result(
            action=action, session_key=session_key, store=backend_store, records=records
        )

    if action == "create":
        backend_id = _validate_backend_id(backend_id)
        repo = str(args.get("repo") or "").strip()
        branch = str(args.get("branch") or "").strip()
        if not repo or not branch:
            raise BackendError("create requires repo and branch")
        record = cnb.create_backend(
            backend_id=backend_id, repo=repo, branch=branch
        )
        record.owner_session_id = session_key
        backend_store.create_backend(record)
        return _backend_result(
            action=action,
            session_key=session_key,
            store=backend_store,
            records=[record],
        )

    if action == "update":
        backend_id = _validate_backend_id(backend_id)
        if args.get("current") is not True:
            raise BackendError("update currently requires current=true")
        record = backend_store.set_current(session_key, backend_id)
        return _backend_result(
            action=action,
            session_key=session_key,
            store=backend_store,
            records=[record],
        )

    if action == "delete":
        backend_id = _validate_backend_id(backend_id)
        record = backend_store.get_backend(backend_id)
        if record.driver == "cnb":
            cnb.stop_backend(record)
        backend_store.delete_backend(backend_id)
        return _backend_result(
            action=action,
            session_key=session_key,
            store=backend_store,
            records=[record],
        )

    raise BackendError("action must be one of: create, get, update, delete")


def earliest_cnb_reclaim_at(created_at: datetime) -> datetime:
    """Return the earliest predictable CNB hard-reclaim risk.

    CNB's ten-minute WebIDE inactivity rule cannot be derived from SSH state and
    is intentionally excluded.  This function models the documented 18-hour
    maximum and the >8-hour 04:00-06:00 Asia/Shanghai reclaim window.
    """
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("created_at must be timezone-aware")
    created_shanghai = created_at.astimezone(CNB_TIMEZONE)
    age_threshold = created_shanghai + CNB_OVERNIGHT_MIN_AGE
    absolute_deadline = created_shanghai + CNB_MAX_HEARTBEAT_LIFETIME

    threshold_clock = age_threshold.timetz().replace(tzinfo=None)
    if CNB_OVERNIGHT_START <= threshold_clock < CNB_OVERNIGHT_END:
        overnight_deadline = age_threshold
    elif threshold_clock < CNB_OVERNIGHT_START:
        overnight_deadline = datetime.combine(
            age_threshold.date(), CNB_OVERNIGHT_START, tzinfo=CNB_TIMEZONE
        )
    else:
        overnight_deadline = datetime.combine(
            age_threshold.date() + timedelta(days=1),
            CNB_OVERNIGHT_START,
            tzinfo=CNB_TIMEZONE,
        )

    return min(absolute_deadline, overnight_deadline)


class BackendLeaseMonitor:
    """Evaluate CNB workspace lease state and emit one-shot events.

    Each backend has a predictable reclaim deadline computed by
    *earliest_cnb_reclaim_at*.  The monitor checks the current time against
    that deadline and emits at most one occurrence of each event type
    (warning → critical → expired) using *BackendStore.mark_event_once* for
    persistent deduplication across process restarts.

    Event types (in order of severity):

    - ``backend.reclaim_warning``  — 30 minutes before the reclaim deadline.
    - ``backend.reclaim_critical`` — 10 minutes before the reclaim deadline.
    - ``backend.expired``          — past the reclaim deadline.

    Every emitted event stores the backend's *owner_session_id* so
    downstream notification logic knows which conversation to contact.
    """

    WARNING_MINUTES = 30
    CRITICAL_MINUTES = 10
    EVENT_WARNING = "backend.reclaim_warning"
    EVENT_CRITICAL = "backend.reclaim_critical"
    EVENT_EXPIRED = "backend.expired"

    def __init__(self, store: BackendStore) -> None:
        self._store = store

    def evaluate(self, backend_id: str, *, now: datetime | None = None) -> dict[str, Any]:
        """Check lease state for *backend_id* and emit any new events.

        Parameters
        ----------
        backend_id:
            Registered CNB backend to evaluate.
        now:
            Override wall clock (for deterministic testing).  Defaults to
            ``datetime.now().astimezone()``.

        Returns
        -------
        A dict with keys:

        - ``backend_id`` — the evaluated backend.
        - ``owner_session_id`` — the backend's recorded owner session (if any).
        - ``deadline`` — ISO-8601 string of the earliest reclaim deadline, or
          ``None`` when the backend has no meaningful deadline (local, unknown
          driver, or no parseable created_at).
        - ``status`` — ``"ok"``, ``"warning"``, ``"critical"``, ``"expired"``,
          or ``"unknown"``.
        - ``events_fired`` — list of event-type strings that were *newly*
          emitted during this evaluation (empty list on repeated calls).
        """
        try:
            record = self._store.get_backend(backend_id)
        except BackendError:
            return {
                "backend_id": backend_id,
                "owner_session_id": "",
                "deadline": None,
                "status": "unknown",
                "events_fired": [],
            }

        # Only CNB backends have a meaningful reclaim contract.
        if record.id == "local" or record.driver != "cnb":
            return {
                "backend_id": backend_id,
                "owner_session_id": record.owner_session_id,
                "deadline": None,
                "status": "ok",
                "events_fired": [],
            }

        try:
            created = datetime.fromisoformat(record.created_at)
        except (ValueError, TypeError):
            return {
                "backend_id": backend_id,
                "owner_session_id": record.owner_session_id,
                "deadline": None,
                "status": "ok",
                "events_fired": [],
            }

        if created.tzinfo is None:
            created = created.replace(tzinfo=CNB_TIMEZONE)

        deadline = earliest_cnb_reclaim_at(created)
        now = (now or datetime.now().astimezone()).astimezone(deadline.tzinfo)
        delta = deadline - now
        events_fired: list[str] = []

        if now >= deadline:
            if self._store.mark_event_once(
                backend_id,
                self.EVENT_EXPIRED,
                owner_session_id=record.owner_session_id,
            ):
                events_fired.append(self.EVENT_EXPIRED)
        elif delta <= timedelta(minutes=self.CRITICAL_MINUTES):
            if self._store.mark_event_once(
                backend_id,
                self.EVENT_CRITICAL,
                owner_session_id=record.owner_session_id,
            ):
                events_fired.append(self.EVENT_CRITICAL)
        elif delta <= timedelta(minutes=self.WARNING_MINUTES):
            if self._store.mark_event_once(
                backend_id,
                self.EVENT_WARNING,
                owner_session_id=record.owner_session_id,
            ):
                events_fired.append(self.EVENT_WARNING)

        if now >= deadline:
            status = "expired"
        elif delta <= timedelta(minutes=self.CRITICAL_MINUTES):
            status = "critical"
        elif delta <= timedelta(minutes=self.WARNING_MINUTES):
            status = "warning"
        else:
            status = "ok"

        return {
            "backend_id": backend_id,
            "owner_session_id": record.owner_session_id,
            "deadline": deadline.isoformat(),
            "status": status,
            "events_fired": events_fired,
        }
