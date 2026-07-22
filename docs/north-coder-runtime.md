# Hermes → North Coder runtime bridge

This worktree adds an opt-in alternate agent runtime to Hermes Gateway. Hermes
continues to own channel adapters, session-key routing, and delivery. North
Coder owns conversations, invocations, tools, skills, queueing, and runtime
state.

## Configuration

```yaml
agent_runtime:
  kind: north_coder
  base_url: http://127.0.0.1:8848
  workspace_id: home-default
  agent_profile_id: hermes:default
  model_id: ng-gpt-5.6-sol
  agent_yaml_path: /Users/jachinshen/.hermes/north-coder-profile/agent.yaml
```

The adapter creates one North conversation per Hermes `session_key`, persists
the mapping in `<HERMES_HOME>/north_coder_conversations.json`, posts messages
to the North REST control plane, and consumes the North WebSocket event stream.

## Profile migration

`hermes_cli.north_coder_profile.export_hermes_profile()` exports only the
current Hermes profile's prompt context (`SOUL.md`, `memories/MEMORY.md`, and
`memories/USER.md`), agent limits, core North builtin tools, and skill paths.
It intentionally does not copy `.env`, auth, sessions, logs, caches, or
provider credentials.

Memory compatibility is currently a startup snapshot, not full Hermes memory
parity: `MEMORY.md` and `USER.md` are included in `system_prompt.md`, but North
is not yet given Hermes' dynamic memory search/write/replace/delete tools.

The Hermes default profile is exported locally to:

```text
~/.hermes/north-coder-profile/agent.yaml
```

It is registered in North as `hermes:default`; the old `cnb-preview` North
profile is not used.

The TUI can switch the runtime without restarting:

```text
/runtime          # show current runtime
/runtime ncoder   # use North Coder
/runtime native   # use Hermes AIAgent
```

The switch is session-local and preserves the Hermes transcript. It requires the
`agent_runtime.kind: north_coder` configuration for the `ncoder` target.

## TUI launch from this worktree

The source checkout has no committed TUI bundle or local `node_modules`. Use the
worktree development path, which borrows dependencies from the canonical Hermes
checkout:

```bash
cd /Users/jachinshen/north-coder-workspace/hermes-agent-north-runtime
unset HERMES_TUI_DIR
HERMES_MAIN_CHECKOUT=/Users/jachinshen/.hermes/hermes-agent \
  python -m hermes_cli.main --profile default --tui --dev
```

If the canonical checkout's Python environment is unavailable, run the same
entrypoint through the project's managed environment (for example `uv run` with
its normal dependencies). Do not use the installed `hermes` binary directly:
it would load the installed Hermes checkout instead of this worktree's TUI
server changes.

## Compatibility boundary

This is a protocol adapter, not a North runtime fork. North-specific tool
catalog bindings and UI-only product tools remain North-owned. Hermes-specific
plugins are not silently translated.
