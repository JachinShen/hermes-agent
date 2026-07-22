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

Memory compatibility now uses the NexAU session/task builtin surface:
`save_memory` and `complete_task` are exported in the artifact. `ToolSearch` is
conditional in NexAU and is deliberately **not** declared in `tools:`; NexAU
mounts it when required. The initial `MEMORY.md` and `USER.md` are also
included in `system_prompt.md`.

This is runtime capability parity, but not yet storage parity with Hermes'
local memory database/files: writes made by North are owned by North's session
backend and are not automatically synchronized back to Hermes `~/.hermes/memories`.

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

## Hermes parity matrix

| Hermes-native capability | North bridge behavior | Status |
|---|---|---|
| Channel/session/thread routing | Hermes owns routing; one durable North conversation per `session_key` | aligned |
| Final text response | North text deltas are assembled and returned through the normal Gateway result path | aligned |
| Streaming edits | North text deltas feed Hermes `GatewayStreamConsumer` when platform streaming is enabled | aligned |
| Tool progress/history | North tool events are forwarded to Gateway observers and retained in the result | aligned at protocol level |
| Cancellation / `/stop` | Active invocation is tracked and cancelled through North REST; TUI cancellation is cross-thread safe | aligned on primary path |
| Permission approval | North `requires_action` / permission events are surfaced as a paused result | bridge present; Hermes approval UI resume still pending |
| `ask_user` | North required-action payload is preserved | bridge present; Gateway answer transport still pending |
| Queued follow-up / busy input | North owns conversation queue; Hermes busy-input policy is not yet mapped one-for-one | partial |
| Hermes memory read/search/write/update/delete | Startup snapshot is injected; North `save_memory` remains North-owned | partial; no Hermes file sync |
| Plugins / hooks | Hermes Gateway hooks still run around the turn; plugin-specific agent callbacks are not translated | partial |
| Images / attachments | Source metadata and workdir are passed; provider-specific binary attachment parity remains North-dependent | partial |
| Session reset / branch / compress | North conversation lifecycle is authoritative; Hermes slash semantics do not automatically map to every North control-plane verb | partial |
| Native fallback | `/runtime native` remains available for TUI session-local switching | aligned |

“Aligned” means the user-visible primary path is exercised by tests or real
smoke. “Partial” is intentional: the adapter does not claim semantics that
North does not expose through its REST/WebSocket contract.
