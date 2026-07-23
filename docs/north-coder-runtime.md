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
current Hermes profile's prompt context (`SOUL.md` plus sanitized, frozen
`MEMORY.md` / `USER.md` snapshots), agent limits, core North builtin tools,
and active skill paths. Archived, dependency, cache, and skill support-package
paths are excluded through Hermes' shared skill-discovery rule.
It intentionally does not copy `.env`, auth, sessions, logs, caches, or
provider credentials.

Memory compatibility uses a Python custom tool named `memory` generated from
Hermes' canonical schema. It delegates add/replace/remove/batch operations for
both `target=memory` and `target=user` to Hermes `MemoryStore`, preserving its
limits, locking, threat scanning, and atomic persistence. The generated prompt
respects the independent `memory_enabled` / `user_profile_enabled` switches and
their character budgets. A managed profile is refreshed immediately before a
new North conversation is created; an existing conversation keeps its frozen
snapshot. `complete_task` remains disabled, and conditional `ToolSearch` is not
declared explicitly because NexAU mounts it when required.

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
| Permission approval | North `requires_action` / permission events are surfaced as a paused result; pending permission actions are registered under the Hermes session key and `/approve`/`/deny` resolve the North invocation | primary permission route aligned |
| `ask_user` / `complete_task` | Disabled in the exported Hermes North profile while the upstream North action-persistence and terminal-tool issues are unresolved; Gateway protocol support remains in code for profiles that explicitly enable them | intentionally disabled in default profile; ask_user tracked in north-coder#981 |
| Queued follow-up / busy input | North owns conversation queue; Hermes busy-input policy is not yet mapped one-for-one | partial |
| Hermes memory read/search/write/update/delete | A sanitized Hermes `MemoryStore` snapshot is injected with independent MEMORY/USER switches and budgets; North calls the canonical `memory` custom tool for both targets; each new North conversation refreshes the managed profile while existing conversations remain frozen | aligned; real write → fresh-conversation read verified for both targets |
| Hermes `/learn` / skill self-learning | North profile now mounts a real `skill_manage` Python custom tool backed by Hermes' `tools.skill_manager_tool`; it writes the active Hermes profile's `skills/` directory. `LoadSkill` discovers new skills after the next conversation/bootstrap; `~/.skills/hermes` is a compat link to the active skills directory | native create → new-session LoadSkill verified |
| Images / attachments | Source metadata and workdir are passed; provider-specific binary attachment parity remains North-dependent | partial |
| Session reset / branch / compress | North conversation lifecycle is authoritative; Hermes slash semantics do not automatically map to every North control-plane verb | partial |
| Native fallback | `/runtime native` remains available for TUI session-local switching | aligned |

“Aligned” means the user-visible primary path is exercised by tests or real
smoke. “Partial” is intentional: the adapter does not claim semantics that
North does not expose through its REST/WebSocket contract.
