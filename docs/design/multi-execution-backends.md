# Multi-execution backends

## Status

This document describes the final implementation of multi-execution backends in
Hermes Agent. The Hermes control plane (conversation history, prompts, LLM
requests, provider credentials, memory, skills, scheduling, delivery) stays
local; environment tools (terminal, process, file I/O) are routed to one local
backend or one of several CNB workspaces.

## Goals

1. Keep conversation history, prompts, LLM requests, provider credentials,
   memory, skills, scheduling, and delivery on the local Hermes host.
2. Let one Hermes profile manage `local + N` CNB execution backends.
3. Let each conversation select its current execution backend without rebuilding
   the system prompt or invalidating prompt caching.
4. Route all environment-sensitive tools consistently.
5. Keep remote workspace persistence explicit: Git is the cross-backend code
   transport.
6. Reuse Hermes's existing environment abstraction instead of creating a second
   terminal/file execution stack.

## Non-goals

- Running the Hermes agent loop or LLM client inside CNB.
- Synchronizing local `HERMES_HOME`, memory, skills, credentials, or
  conversation state into CNB.
- Automatically committing, pushing, restoring, or merging Git work.
- Silently falling back to `local` when a CNB backend is unavailable.
- Automatically replaying a failed command on another backend.
- Treating SSH connectivity as CNB's documented workspace heartbeat.

## Enablement

The feature is **off by default** and must be explicitly enabled in
`config.yaml`:

```yaml
execution_backends:
  enabled: true
```

When disabled, `backend` tool is not registered and
`maybe_resolve_execution_task_id` returns the original `task_id` unchanged for
all environment tools.

The `backend` tool registers only when `execution_backends.enabled` is `true`.
Additionally, the official `cnb` CLI must be installed and authenticated on the
Hermes host before any `backend(action="create")` call can succeed.

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

The router changes only the execution key passed to environment-sensitive tool
handlers. Hooks, observability, approvals, and conversation history continue to
use the original task/session identifiers.

## Backend resource and CRUD surface

Hermes exposes one service-gated model tool:

```text
backend(action="create|get|update|delete", ...)
```

- `create`: create and register a CNB workspace for an exact repository slug and
  git ref.
- `get`: return the current backend and either one backend or all registered
  backends.
- `update`: select a backend for the current session (`current=true` is
  required). Does not kill old background processes — only switches subsequent
  tool routing.
- `delete`: stop a CNB workspace and remove its local registry record. `local`
  cannot be deleted.

Separate `list`, `current`, `status`, `switch`, `extend`, `stop`, or
`acknowledge` tools are deliberately not added.

The tool is available only when `execution_backends.enabled` is `true`,
preserving Hermes's core-tool footprint for users who do not configure the
feature.

### CRUD detailed contract

**create** — inputs are validated:
- `backend_id`: 1–64 chars, `[A-Za-z0-9][A-Za-z0-9._-]{0,63}`.
- `repo`: must be a valid CNB slug (group/subgroup/repo, no leading hyphen).
- `branch`: must pass `git-check-ref-format` rules (no `..`, `@{`, `~`, `^`,
  `:`, `?`, `*`, `[`, `\`, whitespace, leading `.`, trailing `.`, trailing `/`,
  `.lock`, `//`).
- The CNB CLI `start-workspace` command is invoked; the returned workspace SN
  is validated before persisting.

**get** — without `id` returns all registered backends plus `local`. With `id`
returns the specific backend. The current selection (if any) is included in
every response.

**update** — switches the session's binding. The old backend's background
processes are **not** killed; subsequent tool calls use the new backend. The
target backend must be `status == "running"`.

**delete** — safe-dismantle sequence (per-backend lifecycle lock held):
1. Set status to `deleting` — concurrent routing attempts fail closed.
2. Stop the remote CNB workspace via `cnb workspace workspace-stop`.
   - On stop failure: restore original status, preserve
     DB/bindings/env/overrides/processes, re-raise.
3. `process_registry.retire_backend(backend_id)` — mark matching running
   sessions as exited/`backend_lost`. Does **not** send SSH kill signals.
4. `clear_backend_execution_env(backend_id)` — clean in-memory env keys
   (old format `execution-backend:<id>`, new format
   `execution-backend:<id>:session:<hash>`), overrides, cwd, activity, and
   creation locks.
5. Delete from DB (bindings, events, record).
6. Return dumped record with status=`deleted`.

If step 3 or 4 raises (programming error), the DB stays in `deleting` state —
fail-closed, record not removed. No force-delete path.

## Local state

Backend state is profile-local and must resolve through `get_hermes_home()`:

```text
$HERMES_HOME/execution_backends.db
```

The database contains no model/provider credential, CNB bearer token,
conversation content, memory, or skill content. A CNB record contains only the
coordinates required to reconcile or connect to the workspace: backend id,
repository, branch, workspace serial number, status, remote SSH target, cwd,
and timestamps.

`local` is a built-in immutable backend. A session with no binding resolves to
`local`.

A stable execution key is derived from the selected backend:

```text
local selection: preserve the existing task_id behavior
CNB selection:   execution-backend:<backend-id>:session:<sha256(session-key)[:16]>
```

The session key uses the first 16 hex digits of SHA-256 of the session ID or
task ID. This prevents raw session/task IDs from leaking into the environment
key namespace while providing 64 bits of collision entropy.

Same CNB workspace selected by different sessions → shared remote filesystem
(the CNB workspace is the same). But each session's local SSH environment
snapshot, cwd, file cache, and env overrides are independent.

## Tool routing

Environment-sensitive tools are:

- `terminal`
- `process`
- `read_file`
- `search_files`
- `write_file`
- `patch`
- `execute_code`

Before registry dispatch, `maybe_resolve_execution_task_id()`:
1. Skips non-environment tools → returns original `task_id`.
2. Skips when `execution_backends.enabled` is `false` → returns original
   `task_id`.
3. Calls `resolve_execution_task_id()` which:
   a. Reads the session's current backend from `BackendStore`.
   b. For `local`: returns `task_id` unchanged.
   c. For CNB: validates status == `running`, driver == `cnb`, SSH coordinates
      present.
   d. Derives the execution key
      `execution-backend:<id>:session:<sha256(session)[:16]>`.
   e. **Inside the per-backend lifecycle lock**: re-reads binding + status +
      workspace_sn. Rejects if the binding changed, status is no longer
      `running`, or workspace_sn was recreated (delete+recreate-with-same-id
      guard). Then registers env overrides (SSH host/user/port/key, cwd,
      `ssh_sync_hermes_home=False`, `ssh_persistent=True`).
   f. Returns the execution key.

The original task/session ids continue to be supplied to middleware and
post-tool hooks.

The backend CRUD tool itself always executes locally.

## CNB adapter contract

The adapter invokes only the installed official `cnb` CLI. Its command runner
is injectable so behavior tests never require CNB credentials or create remote
resources.

### Runner behavior

- `stdout` is captured and parsed as JSON.
- `stderr` is **never echoed** in error messages to prevent credential leakage.
- Non-zero exit → `BackendError("CNB CLI exited with status N")` — no stderr
  content exposed.
- Empty stdout on zero exit → `BackendError("CNB CLI produced no output on
  stdout")`.

### Expected command paths

```text
cnb workspace list-workspaces --slug <repo> --branch <branch> --page-size 20 --verbose
cnb workspace start-workspace --repo <repo> --branch <branch> --verbose
cnb workspace get-workspace-detail --repo <repo> --sn <sn> --verbose
cnb workspace workspace-stop --pipelineId <id> --verbose
cnb workspace workspace-stop --sn <sn> --verbose
```

### Parsing contract

1. Parse the JSON response envelope.
2. Require a 2xx response `status`, even when the CLI exit code is zero.
3. Match repository, branch, and recorded workspace serial number exactly.
4. Require a running workspace and a `remoteSsh` target before routing tools.
5. Metadata sanitization: recursively filter sensitive keys (case-insensitive,
   underscore/hyphen normalized). Blocked keys include: `authorization`,
   `cookie`, `password`, `secret`, `token`, `api_key`, `apikey`, `credential`,
   `credentials`, `private_key`, `passphrase`, `access_key`, `secret_key`.
6. Use a CNB-specific no-sync SSH environment (`ssh_sync_hermes_home=False`) so
   local `.hermes` files never cross the boundary.

A live CNB E2E remains opt-in because it creates billable/ephemeral
infrastructure. Contract tests use captured, secret-free response shapes.

## CNB lifecycle

CNB's public workspace-recycling documentation is authoritative:

- a newly created workspace may be reclaimed after ten minutes if VS Code is
  never entered;
- after the VS Code page is closed, more than ten minutes without activity may
  reclaim it;
- continuous heartbeat keeps a workspace for at most 18 hours by default
  (cluster configuration may differ);
- a workspace used for more than eight hours is forcibly reclaimed in the
  04:00–06:00 window.

Reference: <https://docs.cnb.cool/zh/workspaces/workspace-recycling.md>

Hermes must not claim that SSH `ControlMaster` is the documented heartbeat. The
local deadline monitor evaluates the earliest predictable hard-risk time in
`Asia/Shanghai`:

```text
min(created_at + 18 hours,
    first time at/after created_at + 8 hours that falls in the 04:00–06:00 window)
```

The ten-minute WebIDE rule is not inferred from SSH activity.

One-shot event types (persistent dedup via `BackendStore.mark_event_once`):

- `backend.reclaim_warning` — 30 minutes before the reclaim deadline.
- `backend.reclaim_critical` — 10 minutes before the reclaim deadline.
- `backend.expired` — past the reclaim deadline.

Events carry the backend's `owner_session_id` for downstream notification.

## Failure semantics

- Unknown, deleted, stopped, or expired backend: fail closed with a structured
  error.
- Orphaned session binding (DB row points to a non-existent backend):
  `BackendError("unknown backend")`.
- CNB CLI error or non-2xx response: preserve the existing session binding and
  return the error.
- Backend deletion while selected: the binding entry is removed from DB and the
  session falls back to `local`. Command execution is never silently retried.
- Remote command failure: report it from that backend; do not replay locally.
- Lost workspace: a new workspace may be created, but no remote filesystem
  recovery is claimed.
- Background process handles remain tied to the environment object that created
  them. After a backend is deleted, matching processes are marked
  `backend_lost` but **not** killed via SSH.
- Same backend ID with a changed `workspace_sn` (delete+recreate): the
  resolver's lifecycle-lock recheck detects the identity change and raises
  `BackendError` instead of routing to the recreated workspace.

## Implementation status

### Implemented (core)

| Component | Status | Notes |
|---|---|---|
| BackendRecord dataclass | ✅ Complete | Includes `public_dict()` with metadata sanitization |
| BackendStore (SQLite CRUD) | ✅ Implemented | Schema: backends, bindings, events tables; ALTER TABLE migration wrapped in try/except |
| Session binding (set/get current) | ✅ Implemented | `set_current`/`get_current` with fail-closed semantics |
| Event dedup table | ✅ Implemented | `mark_event_once` with `INSERT OR IGNORE` |
| Execution key derivation | ✅ Implemented | Format: `execution-backend:<id>:session:<sha256(session)[:16]>` |
| Session-scoped key (isolation) | ✅ Implemented | `_session_key_hash` — SHA-256 prefix, no raw session ID in key |
| Env override registration | ✅ Implemented | `ssh_sync_hermes_home=False`, `ssh_persistent=True` |
| Per-backend lifecycle lock | ✅ Implemented | `_backend_lifecycle_locks` — RLock per backend ID |
| workspace_sn identity guard | ✅ Implemented | Resolver re-reads inside lock, rejects changed workspace_sn |
| Tool dispatch routing | ✅ Implemented | `model_tools.py:maybe_resolve_execution_task_id` before registry dispatch |
| `is_execution_backends_enabled` guard | ✅ Implemented | Config-based enablement |
| Backend CRUD tool registration | ✅ Implemented | `tools/backend_tool.py` — `check_fn=is_execution_backends_enabled` |
| CNBCLIAdapter (parser + runner) | ✅ Implemented | stdout-only, stderr never echoed, metadata recursive sanitize |
| Input validation (repo/branch/backend_id) | ✅ Implemented | `_validate_repo_slug`, `_validate_git_ref`, `_validate_backend_id` |
| Fail-closed get_current | ✅ Implemented | Missing/not-running backend → `BackendError`, not local fallback |
| Lease deadline calculation | ✅ Implemented | `earliest_cnb_reclaim_at` pure function |
| BackendLeaseMonitor (evaluate) | ✅ Implemented | Pure evaluator — warning/critical/expired events with persistent dedup |
| No-sync SSH env override | ✅ Implemented | `ssh_sync_hermes_home=False` override |
| Metadata sanitization | ✅ Implemented | Recursive, case-insensitive, underscore/hyphen normalization |
| Delete cleanup (env/override/cwd) | ✅ Implemented | `clear_backend_execution_env` — matches old and new key formats |
| Delete process retirement | ✅ Implemented | `process_registry.retire_backend` — marks backend_lost, no SSH kill |
| Delete fail-closed (status → deleting) | ✅ Implemented | Stop failure restores status; programming error keeps `deleting` |
| process list includes backend_id | ✅ Implemented | `ProcessSession.backend_id` populated from `_backend_id_from_task_id` |

### Not implemented (gaps)

| Gap | Severity | Details |
|---|---|---|
| Background lifecycle polling/wakeup | 🟡 Medium | `BackendLeaseMonitor.evaluate()` is a pure function; no background loop or cron job drives it automatically. Event emission and session notification must be triggered externally. |
| Lifecycle notification delivery | 🟡 Medium | One-shot events are stored in DB but no delivery mechanism (session wake-up, Git-save guidance) is wired. |
| Opt-in live CNB E2E test | 🟢 Low | All tests use `tmp_path` + mock `CNBCLIAdapter` runners. No test creates a real CNB workspace. User must run E2E manually with real credentials. |
| CNB create timeout / retry | 🟢 Low | `create_backend` calls `cnb start-workspace` once; no retry on transient failure. |

## Isolation and verification

Development and tests must not touch the installed Hermes instance:

- work in a separate clone and branch;
- use the clone's `.venv`;
- use pytest `tmp_path` as `HERMES_HOME`;
- never open the installed `~/.hermes/state.db` or `execution_backends.db`;
- do not restart or replace the running gateway;
- do not start a real CNB workspace in the default test suite.

### Core test files

```bash
cd <hermes-agent-clone>

# BackendStore + CRUD + monitor + adapter
python -m pytest tests/tools/test_execution_backends.py -v --tb=short

# Routing, disable-by-default, process ownership
python -m pytest tests/tools/test_backend_routing.py -v --tb=short

# Session-scoped execution keys, RLock concurrency
python -m pytest tests/tools/test_backend_session_isolation.py -v --tb=short

# Input validation, metadata sanitization, fail-closed, CNB runner security
python -m pytest tests/tools/test_execution_backend_security.py -v --tb=short

# Delete cleanup: status→deleting, stop failure recovery,
# env/override/cwd retirement, lifecycle-lock serialization
python -m pytest tests/tools/test_backend_delete_cleanup.py -v --tb=short
```

All tests use `tmp_path` for `BackendStore`, injectable `CNBCLIAdapter` runners
(no real CLI, no network), and deterministic time for `BackendLeaseMonitor`.
No test suite touches the network or requires CNB credentials.

### Test characteristics

| Property | Value |
|---|---|
| Hermes home isolation | `tmp_path` via `store(tmp_path)` fixture |
| CNB CLI dependency | None — `CNBCLIAdapter(runner=...)` injectable |
| Network calls | None |
| SSH connections | None — routing tests monkeypatch `register_task_env_overrides` |
| Real clocks | Deterministic — `BackendLeaseMonitor.evaluate()` accepts `now=` |

## Delivery phases

### Milestone 1: control-plane MVP ✅ Completed

- ✅ local SQLite store and session bindings
- ✅ one CRUD tool (create/get/update/delete)
- ✅ deterministic environment-tool routing with lifecycle-lock serialization
- ✅ session-scoped execution keys (`sha256` prefix, no raw ID)
- ✅ workspace_sn identity guard against delete+recreate
- ✅ per-task environment type/SSH overrides
- ✅ no-sync SSH mode for CNB
- ✅ CNB CLI parsing contract and lifecycle pure functions
- ✅ input validation (repo slug, git ref, backend id)
- ✅ fail-closed semantics (missing/stopped backend, orphaned binding)
- ✅ delete safe-dismantle (status → deleting, stop, retire, clear env, DB prune)
- ✅ metadata sanitization (recursive, case-insensitive)
- ✅ isolated behavior tests (all tmp_path, no network)
- ✅ BackendLeaseMonitor pure evaluator

### Milestone 2: live lifecycle integration ☐

- ☐ Background polling loop or cron-driven `BackendLeaseMonitor.evaluate()`
- ☐ Session wake-up and Git-save guidance on reclaim events
- ☐ Opt-in real CNB E2E evidence

### Milestone 3: operator UX ☐

- ☐ `hermes tools` configuration UX
- ☐ TUI/Desktop backend indicator and picker
- ☐ Lifecycle notifications across messaging platforms
- ☐ Migration and recovery diagnostics
