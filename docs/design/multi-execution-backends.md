# Multi-execution backends

## Status

This document defines an incremental design for keeping the Hermes control plane local while routing environment tools to one local backend or one of several CNB workspaces.

The first implementation milestone covers the local registry, per-session binding, deterministic routing key, CNB CLI adapter contract, lifecycle deadline calculation, and behavior tests. Background lifecycle delivery and a live CNB end-to-end test are follow-up gates rather than prerequisites for the local control-plane seam.

## Goals

1. Keep conversation history, prompts, LLM requests, provider credentials, memory, skills, scheduling, and delivery on the local Hermes host.
2. Let one Hermes profile manage `local + N` CNB execution backends.
3. Let each conversation select its current execution backend without rebuilding the system prompt or invalidating prompt caching.
4. Route all environment-sensitive tools consistently.
5. Keep remote workspace persistence explicit: Git is the cross-backend code transport.
6. Reuse Hermes's existing environment abstraction instead of creating a second terminal/file execution stack.

## Non-goals

- Running the Hermes agent loop or LLM client inside CNB.
- Synchronizing local `HERMES_HOME`, memory, skills, credentials, or conversation state into CNB.
- Automatically committing, pushing, restoring, or merging Git work.
- Silently falling back to `local` when a CNB backend is unavailable.
- Automatically replaying a failed command on another backend.
- Treating SSH connectivity as CNB's documented workspace heartbeat.

## Architecture

```text
local Hermes process
├── conversation + LLM + memory + skills
├── BackendStore ($HERMES_HOME/execution_backends.db)
│   ├── backend records: local + N CNB
│   └── per-session current-backend bindings
├── backend CRUD tool
├── ExecutionBackendRouter
│   └── (session_id, task_id) -> stable execution task key
├── existing environment tools
│   ├── terminal / process
│   ├── read_file / search_files
│   ├── write_file / patch
│   └── execute_code
└── adapters
    ├── local (existing LocalEnvironment)
    └── CNB lifecycle adapter -> cnb CLI -> no-sync SSH environment
```

The router changes only the execution key passed to environment-sensitive tool handlers. Hooks, observability, approvals, and conversation history continue to use the original task/session identifiers.

## Backend resource and CRUD surface

Hermes exposes one service-gated model tool:

```text
backend(action="create|get|update|delete", ...)
```

- `create`: create and register a CNB workspace for an exact repository and branch.
- `get`: return the current backend and either one backend or all registered backends.
- `update`: select a backend for the current session. Mutable display metadata may be added later, but lifecycle extension is not promised.
- `delete`: stop a CNB workspace and remove its local registry record. `local` cannot be deleted.

Separate `list`, `current`, `status`, `switch`, `extend`, `stop`, or `acknowledge` tools are deliberately not added.

The tool is available only when multi-backend execution is enabled in `config.yaml`, preserving Hermes's core-tool footprint for users who do not configure the feature.

## Local state

Backend state is profile-local and must resolve through `get_hermes_home()`:

```text
$HERMES_HOME/execution_backends.db
```

The database contains no model/provider credential, CNB bearer token, conversation content, memory, or skill content. A CNB record contains only the coordinates required to reconcile or connect to the workspace: backend id, repository, branch, workspace serial number, status, remote SSH target, cwd, and timestamps.

`local` is a built-in immutable backend. A session with no binding resolves to `local`.

A stable execution key is derived from the selected backend:

```text
local selection: preserve the existing task_id behavior
CNB selection:   execution-backend:<backend-id>
```

Using the backend id means multiple conversations selecting the same CNB workspace share the same environment object and working directory. Conversation bindings remain independent.

## Tool routing

Environment-sensitive tools are:

- `terminal`
- `process`
- `read_file`
- `search_files`
- `write_file`
- `patch`
- `execute_code`

Before registry dispatch, the router resolves the selected backend and supplies the execution task key to these handlers. The original task/session ids continue to be supplied to middleware and post-tool hooks.

The backend CRUD tool itself always executes locally.

## CNB adapter contract

The adapter invokes only the installed official `cnb` CLI. Its command runner is injectable so behavior tests never require CNB credentials or create remote resources.

Expected command paths:

```text
cnb workspace list-workspaces --slug <repo> --branch <branch> --page-size 20 --verbose
cnb workspace start-workspace --repo <repo> --branch <branch> --verbose
cnb workspace get-workspace-detail --repo <repo> --sn <sn> --verbose
```

The adapter must:

1. parse the JSON response envelope;
2. require a 2xx response `status`, even when the CLI exit code is zero;
3. match repository, branch, and recorded workspace serial number exactly;
4. require a running workspace and a `remoteSsh` target before routing tools;
5. never persist or return bearer tokens, cookies, authorization headers, or model credentials;
6. use a CNB-specific no-sync SSH environment so local `.hermes` files never cross the boundary.

A live CNB E2E remains opt-in because it creates billable/ephemeral infrastructure. Contract tests use captured, secret-free response shapes.

## CNB lifecycle

CNB's public workspace-recycling documentation is authoritative:

- a newly created workspace may be reclaimed after ten minutes if VS Code is never entered;
- after the VS Code page is closed, more than ten minutes without activity may reclaim it;
- continuous heartbeat keeps a workspace for at most 18 hours by default (cluster configuration may differ);
- a workspace used for more than eight hours is forcibly reclaimed in the 04:00–06:00 window.

Reference: <https://docs.cnb.cool/zh/workspaces/workspace-recycling.md>

Hermes must not claim that SSH `ControlMaster` is the documented heartbeat. The lifecycle adapter reconciles actual CNB status. The local deadline monitor computes the earliest predictable hard-risk time in `Asia/Shanghai`:

```text
min(created_at + 18 hours,
    first time at/after created_at + 8 hours that falls in the 04:00–06:00 window)
```

The ten-minute WebIDE rule is not inferred from SSH activity. An early reclaim is detected by status polling.

Planned warning events are one-shot transitions, not prompt-prefix state:

- `backend.reclaim_warning`
- `backend.reclaim_critical`
- `backend.expired`

The owner conversation is notified before the predictable deadline so the agent can inspect Git state and decide whether to commit and push. Hermes never runs automatic `git add`, `commit`, or `push`.

## Failure semantics

- Unknown, deleted, stopped, or expired backend: fail closed with a structured error.
- CNB CLI error or non-2xx response: preserve the existing session binding and return the error.
- Backend deletion while selected: affected bindings return to `local` only as an explicit delete consequence recorded by the CRUD result; command execution is never silently retried.
- Remote command failure: report it from that backend; do not replay locally.
- Lost workspace: a new workspace may be created, but no remote filesystem recovery is claimed.
- Background process handles remain tied to the environment object that created them.

## Implementation status

The following describes the actual state of the codebase at commit `23ca94656`.
Items marked **CRITICAL** are bugs that prevent the feature from working.

### Implemented (Milestone 1 core)

| Component | Status | File(s) |
|---|---|---|
| BackendRecord dataclass | ✅ Complete | `tools/execution_backends.py:52-80` |
| BackendStore (SQLite CRUD) | ⚠️ Blocked by bug | `tools/execution_backends.py:107-322` |
| Session binding (set/get current) | ✅ Implemented | `tools/execution_backends.py:249-281` |
| Event dedup table | ✅ Implemented | `tools/execution_backends.py:145-151` |
| Execution key derivation | ✅ Implemented | `tools/execution_backends.py:351-387` |
| Env override registration | ✅ Implemented | `tools/execution_backends.py:372-386` |
| Tool dispatch routing (model_tools.py) | ✅ Implemented | `model_tools.py:1264-1268` |
| is_execution_backends_enabled guard | ✅ Implemented | `tools/execution_backends.py:325-330` |
| Backend CRUD tool registration | ✅ Implemented | `tools/backend_tool.py` |
| CNBCLIAdapter (parser + runner) | ✅ Implemented | `tools/execution_backends.py:480-622` |
| Lease deadline calculation | ✅ Implemented | `tools/execution_backends.py:708-735` |
| BackendLeaseMonitor (warning/critical/expired) | ✅ Implemented | `tools/execution_backends.py:738-868` |
| No-sync SSH env override | ✅ Implemented | `tools/execution_backends.py:384` |
| Metadata sanitization | ✅ Implemented | `tools/execution_backends.py:82-91` |

### Critical issues

**CRITICAL-1** — `tools/execution_backends.py:118-159` — Python code inside SQL `executescript()` string

The `_init_schema()` method embeds a `try:/except:` block of Python code within the multi-line string passed to `self._conn.executescript()`. SQLite's `executescript()` accepts only SQL text. The word `try:` on line 152 is passed to SQLite as SQL, producing `sqlite3.OperationalError: near "try": syntax error`. This prevents `BackendStore.__init__()` from completing, making the entire multi-execution-backend feature non-functional on first use.

**Fix required**: Move the ALTER TABLE migration (lines 152-157) OUTSIDE the `executescript()` call, placing it as a separate `self._conn.execute()` call after the schema string. Since `owner_session_id` is already in the CREATE TABLE definition (line 149), the ALTER TABLE is only needed for databases created before the column was added — consider removing it entirely if no such databases exist in production.

**CRITICAL-2** — `tools/execution_backends.py:152-157` — Migration is skipped even without the syntax error

The CREATE TABLE for `execution_backend_events` (line 145-151) already includes `owner_session_id TEXT NOT NULL DEFAULT ''`. The ALTER TABLE on lines 153-154 attempts to add the same column that already exists. On a fresh database, this would fail with `duplicate column name` (an sqlite3.OperationalError) and would have been caught by the intended `except sqlite3.OperationalError` block — if the try/except weren't inside the SQL string. This is harmless in intended use (migration for pre-existing databases), but the code path is unreachable due to CRITICAL-1.

### Gaps and observations

| Gap | Severity | Details |
|---|---|---|
| Wire routing to `terminal_tool` create path | 🟡 Medium | `resolve_execution_task_id()` registers overrides and returns `"execution-backend:<id>"` as the dispatch key. The terminal tool correctly picks up overrides via `resolve_task_overrides()`. However, `_resolve_container_task_id()` in terminal_tool.py collapses CWD-only overrides back to `"default"` — this could interfere if a CNB routing key is processed before `register_task_env_overrides` has been called. The current execution order in `model_tools.py` (call `maybe_resolve_execution_task_id` before `dispatch`) is correct, but the dependency on ordering is not documented. |
| No explicit test for no-sync behavior | 🟡 Medium | `test_router_uses_stable_backend_key_and_no_sync_ssh_override` checks that `ssh_sync_hermes_home=False` is passed in the overrides dict, but does NOT verify that the `SSHEnvironment` actually skips file sync when this flag is false. Add a test that inspects whether `_sync_manager` is None when `sync_hermes_home=False`. |
| No explicit test for original IDs in middleware | 🟡 Medium | The document specifies that "model/tool hooks retain original session/task ids after routing." The code in `model_tools.py` passes the original `session_id` and `task_id` to middleware while routing only `dispatch_task_id` to the handler. This is correct by inspection but has no dedicated test. |
| `_validate_backend_id` uppercase letters allowed | 🟢 Low | The regex `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$` allows uppercase letters, though the document doesn't specify otherwise. Consider whether SQLite's case-insensitive PRIMARY KEY could cause confusion between `cnb-a` and `CNB-A`. |

## Isolation and verification

Development and tests must not touch the installed Hermes instance:

- work in a separate clone and branch;
- use the clone's `.venv`;
- use pytest `tmp_path` as `HERMES_HOME`;
- never open the installed `~/.hermes/state.db` or `execution_backends.db`;
- do not restart or replace the running gateway;
- do not start a real CNB workspace in the default test suite.

### Behavior test coverage (test_execution_backends.py)

| # | Required behavior | Test | Status |
|---|---|---|---|
| 1 | Default selection is `local` | `test_store_defaults_every_session_to_local` (line 48) | ✅ |
| 2 | Two sessions can select different backends | `test_two_sessions_can_select_different_backends` (line 54) | ✅ |
| 3 | Switching changes only the environment execution key | `test_router_uses_stable_backend_key_and_no_sync_ssh_override` (line 100) | ✅ |
| 4 | CRUD state survives a new store instance | `test_store_persists_registry_and_binding` (line 65) | ✅ |
| 5 | Unknown/unready backends fail closed | `test_router_fails_closed_for_unready_backend` (line 132) | ✅ |
| 6a | CNB parsing rejects non-2xx envelopes | `test_parse_cnb_response_requires_http_success_even_when_cli_succeeded` (line 185) | ✅ |
| 6b | CNB parsing rejects ambiguous matches | `test_cnb_adapter_requires_exact_unambiguous_workspace_match` (line 197) | ✅ |
| 6c | CNB detail parses into secret-free record | `test_cnb_adapter_parses_detail_into_secret_free_record` (line 232) | ✅ |
| 7 | No-sync remote execution never syncs local Hermes files | `test_router_uses_stable_backend_key_and_no_sync_ssh_override` checks override dict (line 128); does NOT verify SSHEnvironment skips FileSyncManager creation | 🟡 Partial |
| 8a | 18-hour deadline boundaries | `test_earliest_cnb_reclaim_at` parametrized (line 259) | ✅ |
| 8b | Deadline requires timezone-aware input | `test_earliest_cnb_reclaim_requires_timezone_aware_input` (line 285) | ✅ |
| 9 | Model/tool hooks retain original session/task ids | No dedicated test; verified by inspection in `model_tools.py:1260-1295` | 🟡 Missing |

### Running the test suite

```bash
# All execution backend tests (no CNB credentials needed)
cd <hermes-agent-clone>
python -m pytest tests/tools/test_execution_backends.py -v --tb=short

# With coverage
python -m pytest tests/tools/test_execution_backends.py \
  --cov=tools.execution_backends --cov=tools.backend_tool \
  --cov-report=term-missing

# Test without the critical bug (after fix)
# See CRITICAL-1 above
```

**Note**: Due to **CRITICAL-1** (`tools/execution_backends.py:118-159`), `BackendStore.__init__()` raises `sqlite3.OperationalError`. Run the isolated test above to reproduce the failure. The fix must be applied before any test can pass.

## Isolation test procedure

To add a new behavior test:

1. **Fixture**: use `tests/tools/test_execution_backends.py`'s `store` fixture (line 27), which creates a `BackendStore` at `tmp_path / "execution_backends.db"`. This guarantees zero interaction with the real Hermes home.
2. **No CNB credentials**: use the injectable `CNBCLIAdapter(runner=lambda argv: ...)` pattern for all CLI interaction tests.
3. **No real SSH**: the routing test (`test_router_uses_stable_backend_key_and_no_sync_ssh_override`) monkeypatches `register_task_env_overrides` to capture overrides without creating a live SSH connection.
4. **Deterministic time**: `BackendLeaseMonitor.evaluate()` accepts a `now` parameter; use it with known `datetime` values.
5. **Runtime dependencies**: tests must not import or run any live `cnb` CLI commands, establish SSH connections, or access files outside `tmp_path`.

## Delivery phases

### Milestone 1: control-plane MVP

- ✅ local SQLite store and session bindings (blocked by CRITICAL-1)
- ✅ one CRUD tool
- ✅ deterministic environment-tool routing
- ✅ per-task environment type/SSH overrides
- ✅ no-sync SSH mode for CNB
- ✅ CNB CLI parsing contract and lifecycle pure functions
- ✅ isolated behavior tests (blocked by CRITICAL-1)

### Milestone 2: live lifecycle integration

- ☐ CNB create/detail/stop reconciliation against a dedicated test repository
- ☐ WebIDE heartbeat integration using a documented supported mechanism
- ☐ background polling and one-shot session events
- ☐ owner-session wake-up and Git-save guidance
- ☐ opt-in real CNB E2E evidence

### Milestone 3: operator UX

- ☐ `hermes tools` configuration UX
- ☐ TUI/Desktop backend indicator and picker
- ☐ lifecycle notifications across messaging platforms
- ☐ migration and recovery diagnostics
