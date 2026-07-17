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

## Isolation and verification

Development and tests must not touch the installed Hermes instance:

- work in a separate clone and branch;
- use the clone's `.venv`;
- use pytest `tmp_path` as `HERMES_HOME`;
- never open the installed `~/.hermes/state.db` or `execution_backends.db`;
- do not restart or replace the running gateway;
- do not start a real CNB workspace in the default test suite.

Required behavior tests:

1. default selection is `local`;
2. two sessions can select different backends;
3. switching changes only the environment execution key;
4. CRUD state survives a new store instance under the same temporary home;
5. unknown/unready backends fail closed;
6. CNB response parsing rejects non-2xx envelopes and ambiguous workspace matches;
7. no-sync remote execution never enumerates or transfers local Hermes files;
8. the 18-hour and overnight deadline calculations cover boundary times;
9. model/tool hooks retain original session/task ids after routing.

## Delivery phases

### Milestone 1: control-plane MVP

- local SQLite store and session bindings;
- one CRUD tool;
- deterministic environment-tool routing;
- per-task environment type/SSH overrides;
- no-sync SSH mode for CNB;
- CNB CLI parsing contract and lifecycle pure functions;
- isolated behavior tests.

### Milestone 2: live lifecycle integration

- CNB create/detail/stop reconciliation against a dedicated test repository;
- WebIDE heartbeat integration using a documented supported mechanism;
- background polling and one-shot session events;
- owner-session wake-up and Git-save guidance;
- opt-in real CNB E2E evidence.

### Milestone 3: operator UX

- `hermes tools` configuration UX;
- TUI/Desktop backend indicator and picker;
- lifecycle notifications across messaging platforms;
- migration and recovery diagnostics.
