"""North Coder Agent Runtime adapter.

This module keeps Hermes channel/session delivery and canonical transcript
ownership intact while delegating one turn to a running North Coder control
plane. A North conversation is disposable runtime backing state for one Hermes
session, not the source of truth for that session's history.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

EventCallback = Callable[[dict[str, Any]], Awaitable[None] | None]
DeltaCallback = Callable[[str], None]


async def _interrupt_north_host(
    host_agent: Any,
    session_key: str,
    interrupt_reason: str = "",
) -> None:
    """Interrupt both the Hermes host agent and the North runtime for session_key.

    Gateway's interrupt paths (monitor_for_interrupt, backup checks, inactivity
    timeout) all reach a running Hermes agent first, but a North Coder foreground
    turn also owns a process-wide runtime invocation that the host agent's
    ``interrupt()`` does not touch.  This helper bridges the gap so all paths
    consistently cancel both boundaries.

    ``host_agent`` may be ``None`` (no Hermes host running); only the North
    runtime is cancelled in that case.  ``runtime_from_raw`` failures are
    caught and logged, never propagated.
    """
    if host_agent is not None and hasattr(host_agent, "interrupt"):
        host_agent.interrupt(interrupt_reason)
    try:
        from gateway.run import _gateway_config_home, _load_gateway_config

        _runtime = runtime_from_raw(
            _load_gateway_config(),
            _gateway_config_home(),
        )
        if _runtime is not None:
            await _runtime.cancel_session_async(session_key)
    except Exception:
        logger.exception(
            "Failed to cancel North invocation for session %s", session_key
        )


@dataclass(frozen=True)
class NorthCoderRuntimeConfig:
    base_url: str = "http://127.0.0.1:8848"
    workspace_id: str = "home-default"
    agent_profile_id: str = "hermes:default"
    agent_yaml_path: Optional[str] = None
    model_id: Optional[str] = "ng-gpt-5.6-sol"
    timeout_seconds: float = 1800.0
    state_file: Optional[str] = None
    seed_token_budget: int = 2000
    tui_variant: bool = False
    workspace_switch_supported: bool = False

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> Optional["NorthCoderRuntimeConfig"]:
        section = raw.get("agent_runtime")
        if not isinstance(section, dict):
            section = raw.get("agent", {}).get("north_coder") if isinstance(raw.get("agent"), dict) else None
            if section is None:
                return None
            section = dict(section)
            section.setdefault("kind", "north_coder")
        if str(section.get("kind", "")).strip().lower() not in {"north_coder", "north-coder"}:
            return None
        return cls(
            base_url=str(section.get("base_url") or section.get("endpoint") or cls.base_url).rstrip("/"),
            workspace_id=str(section.get("workspace_id") or cls.workspace_id),
            agent_profile_id=str(section.get("agent_profile_id") or section.get("profile_id") or cls.agent_profile_id),
            agent_yaml_path=(str(section["agent_yaml_path"]) if section.get("agent_yaml_path") else None),
            model_id=(str(section["model_id"]) if section.get("model_id") else None),
            timeout_seconds=float(section.get("timeout_seconds") or cls.timeout_seconds),
            state_file=(str(section["state_file"]) if section.get("state_file") else None),
            seed_token_budget=int(section.get("seed_token_budget") or cls.seed_token_budget),
        )


class NorthCoderRuntime:
    """Translate one Hermes Gateway turn to North REST + WebSocket APIs."""

    def __init__(self, config: NorthCoderRuntimeConfig, hermes_home: Path):
        self.config = config
        self.hermes_home = hermes_home
        self.state_file = Path(config.state_file) if config.state_file else hermes_home / "north_coder_conversations.json"
        self._state_lock = threading.RLock()
        # Process-local ownership for cross-thread Gateway/TUI interrupts.
        self._active_invocations: dict[str, str] = {}
        self._inflight_sessions: set[str] = set()
        self._cancel_requested_sessions: set[str] = set()
        # Idempotent-cancel guard: invocation_ids for which we have already
        # POSTed a cancel request.  Prevents duplicate POSTs from repeated
        # cancel_session_async calls for the same in-flight invocation.
        self._cancel_posted: set[str] = set()
        self._active_lock = threading.RLock()

    async def run_turn(
        self,
        *,
        message: str,
        session_key: str,
        hermes_session_id: str,
        context_prompt: str = "",
        source: Any = None,
        conversation_history: Optional[list[dict[str, Any]]] = None,
        workdir: Optional[str] = None,
        workspace_id: Optional[str] = None,
        event_message_id: Optional[str] = None,
        on_delta: Optional[DeltaCallback] = None,
        on_event: Optional[EventCallback] = None,
        metadata_extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Run one North turn with exception-safe interrupt ownership."""
        with self._active_lock:
            self._inflight_sessions.add(session_key)
            # A cancellation belongs to one concrete in-flight turn. Never let
            # a stale request poison the next turn for the same Hermes session.
            self._cancel_requested_sessions.discard(session_key)
        try:
            return await self._run_turn_impl(
                message=message,
                session_key=session_key,
                hermes_session_id=hermes_session_id,
                context_prompt=context_prompt,
                source=source,
                conversation_history=conversation_history,
                workdir=workdir,
                workspace_id=workspace_id,
                event_message_id=event_message_id,
                on_delta=on_delta,
                on_event=on_event,
                metadata_extra=metadata_extra,
            )
        finally:
            with self._active_lock:
                old_inv = self._active_invocations.pop(session_key, None)
                if old_inv:
                    self._cancel_posted.discard(old_inv)
                self._inflight_sessions.discard(session_key)
                self._cancel_requested_sessions.discard(session_key)

    async def _run_turn_impl(
        self,
        *,
        message: str,
        session_key: str,
        hermes_session_id: str,
        context_prompt: str = "",
        source: Any = None,
        conversation_history: Optional[list[dict[str, Any]]] = None,
        workdir: Optional[str] = None,
        workspace_id: Optional[str] = None,
        event_message_id: Optional[str] = None,
        on_delta: Optional[DeltaCallback] = None,
        on_event: Optional[EventCallback] = None,
        metadata_extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        import aiohttp

        requested_workspace_id = str(workspace_id or "").strip() or None
        # A caller-selected North workspace and an arbitrary filesystem workdir
        # are mutually exclusive routing modes. Gateway defaults select the
        # former; TUI/project sessions select the latter.
        effective_workdir = None if requested_workspace_id else (
            workdir or getattr(source, "workdir", None)
        )
        if effective_workdir:
            effective_workdir = str(Path(effective_workdir).expanduser().resolve())
        existing_id, existing_workdir, existing_workspace_id = await self._conversation_binding(
            session_key
        )
        composite_run = bool(
            (
                requested_workspace_id
                and (
                    not existing_id
                    or existing_workspace_id != requested_workspace_id
                    or bool(existing_workdir)
                )
            )
            or (
                effective_workdir
                and (
                    not existing_id
                    or existing_workdir != effective_workdir
                    or not existing_workspace_id
                )
            )
        )
        if composite_run:
            conversation_id, created = "", True
        elif existing_id:
            conversation_id, created = existing_id, False
        else:
            conversation_id, created = await self._conversation_for_turn(session_key)
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds, sock_read=self.config.timeout_seconds)
        headers = {"Content-Type": "application/json", "X-Hermes-Session-Id": hermes_session_id}
        client_message_id = event_message_id or f"hermes-{uuid.uuid4().hex}"
        payload = {
            "content": self._seeded_message(message, conversation_history, self.config.seed_token_budget) if created else message,
            "client_message_id": client_message_id,
            "agent_profile_id": self.config.agent_profile_id,
            "metadata": {
                "hermes_session_key": session_key,
                "hermes_session_id": hermes_session_id,
                "platform": getattr(getattr(source, "platform", None), "value", None),
                "chat_id": getattr(source, "chat_id", None),
                "thread_id": getattr(source, "thread_id", None),
                "hermes_context_prompt": context_prompt or None,
            },
        }
        if metadata_extra:
            payload["metadata"].update(metadata_extra)
        if self.config.model_id:
            payload["model_id"] = self.config.model_id
        if self.config.agent_yaml_path:
            payload["agent_yaml_path"] = self.config.agent_yaml_path
        if effective_workdir:
            payload["workdir"] = str(effective_workdir)

        full_response: list[str] = []
        invocation_id: Optional[str] = None
        terminal: dict[str, Any] = {"status": "completed", "tools": []}

        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as client:
            ws = None
            if composite_run:
                self._refresh_managed_profile()
                run_payload: dict[str, Any] = {
                    "content": payload["content"],
                    "agent_profile_id": self.config.agent_profile_id,
                    "metadata": payload["metadata"],
                    "conversation_options": {
                        "title": f"Hermes {session_key[-80:]}",
                        "agent_config": {"agent_profile_id": self.config.agent_profile_id},
                    },
                }
                if effective_workdir:
                    # Hermes project sessions must be navigable from the North
                    # sidebar. A bare workdir creates a workspace-less
                    # conversation that North Studio intentionally cannot open.
                    run_payload["workdir"] = effective_workdir
                    run_payload["register_workdir"] = True
                elif requested_workspace_id:
                    run_payload["workspace_id"] = requested_workspace_id
                if self.config.model_id:
                    run_payload["model_id"] = self.config.model_id
                if self.config.agent_yaml_path:
                    run_payload["agent_yaml_path"] = self.config.agent_yaml_path
                request_url = f"{self.config.base_url}/api/run"
                request_payload = run_payload
            else:
                try:
                    ws = await self._connect_events(client, conversation_id)
                except Exception as exc:
                    logger.warning("North event stream unavailable; falling back to invocation polling: %s", exc)
                request_url = f"{self.config.base_url}/api/conversations/{conversation_id}/messages"
                request_payload = payload

            registration_degraded_to_workdir = False

            async def _post_request(request_body: dict[str, Any]) -> tuple[int, str]:
                async with client.post(request_url, json=request_body) as response:
                    return response.status, await response.text()

            response_status, body = await _post_request(request_payload)
            if (
                response_status >= 400
                and composite_run
                and effective_workdir
                and request_payload.get("register_workdir") is True
                and self._is_workdir_register_failed(response_status, body)
            ):
                # North 0.4 does not discover detached-HEAD linked worktrees because
                # its porcelain parser only registers entries carrying `branch`.
                # `workdir_register_failed` is emitted before invocation creation,
                # so retrying workspace-less cannot duplicate agent side effects.
                degraded_payload = dict(request_payload)
                degraded_payload.pop("register_workdir", None)
                logger.warning(
                    "North could not register workdir %s as a sidebar workspace; "
                    "retrying with an exact workspace-less workdir binding",
                    effective_workdir,
                )
                response_status, body = await _post_request(degraded_payload)
                registration_degraded_to_workdir = True

            if response_status >= 400:
                operation = "run" if composite_run else "message"
                raise RuntimeError(
                    f"North {operation} request failed ({response_status}): {body[:500]}"
                )
            accepted = json.loads(body) if body else {}
            if composite_run:
                conversation_id = str(accepted.get("conversation_id") or "")
                if not conversation_id:
                    raise RuntimeError("North composite run response omitted conversation_id")
                workspace_id = str(accepted.get("workspace_id") or "")
                if not workspace_id and not registration_degraded_to_workdir:
                    raise RuntimeError("North composite run response omitted workspace_id")
                if requested_workspace_id and workspace_id != requested_workspace_id:
                    raise RuntimeError(
                        "North composite run response returned unexpected workspace_id "
                        f"{workspace_id!r}; expected {requested_workspace_id!r}"
                    )
                await self._record_conversation_binding(
                    session_key,
                    conversation_id,
                    str(effective_workdir) if effective_workdir else None,
                    workspace_id or None,
                    workdir_only=registration_degraded_to_workdir,
                )
            invocation_id = accepted.get("invocation_id")
            if invocation_id:
                with self._active_lock:
                    self._active_invocations[session_key] = str(invocation_id)
                    cancel_requested = session_key in self._cancel_requested_sessions
                    already_posted = str(invocation_id) in self._cancel_posted
                if cancel_requested and not already_posted:
                    with self._active_lock:
                        self._cancel_posted.add(str(invocation_id))
                    await self.cancel(str(invocation_id))

            if composite_run:
                try:
                    ws = await self._connect_events(client, conversation_id)
                except Exception as exc:
                    logger.warning("North event stream unavailable; falling back to invocation polling: %s", exc)

            if ws is not None:
                ws_task = asyncio.ensure_future(
                    self._consume_events(
                        ws,
                        full_response,
                        on_delta,
                        on_event,
                        process_replay=composite_run,
                        stream_text=not bool(invocation_id and self.config.tui_variant),
                    )
                )
                if invocation_id:
                    # 并发：WS消费 vs REST轮询。谁先terminal就用谁。
                    # 此设计解决North静默WS导致无限卡住的问题：
                    #   - REST先terminal（failed/cancelled/completed/requires_action）
                    #     时，结束WS消费并快速返回
                    #   - WS先terminal时，仍等待REST reconcile（不立即cancel poll）
                    #   - 双方都永久running时，_poll_result的timeout_seconds硬deadline触发
                    poll_task = asyncio.ensure_future(
                        self._poll_result(client, invocation_id, full_response, on_event, on_delta=on_delta)
                    )
                    # 等待最先完成的一方
                    done, pending = await asyncio.wait(
                        [ws_task, poll_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    # REST先terminal → 取消WS
                    if poll_task in done and not poll_task.cancelled():
                        ws_task.cancel()
                        try:
                            await ws_task
                        except (asyncio.CancelledError, Exception):
                            pass
                        await ws.close()

                        try:
                            authoritative = poll_task.result()
                        except (TimeoutError, asyncio.TimeoutError):
                            raise
                        except Exception as exc:
                            logger.warning(
                                "North REST poll finished first but failed: %s", exc
                            )
                            raise
                        else:
                            # REST finished first — build terminal from authoritative result
                            rest_tools = self._extract_result_tool_blocks(authoritative)
                            terminal = {
                                "status": str(authoritative.get("status") or "completed"),
                                "tools": rest_tools,
                            }
                            if authoritative.get("required_action"):
                                terminal["required_action"] = authoritative["required_action"]
                            terminal["result"] = authoritative
                            # Incremental delta from REST text (silent WS/slow provider):
                            # only append new suffix not yet in full_response.
                            # When WS already made progress with different content
                            # (REST text doesn't start with WS text), keep WS content.
                            rest_text = self._extract_result_text(authoritative)
                            if rest_text:
                                current = "".join(full_response)
                                if not full_response:
                                    full_response.append(rest_text)
                                    if on_delta:
                                        on_delta(rest_text)
                                elif rest_text.startswith(current) and len(rest_text) > len(current):
                                    suffix = rest_text[len(current):]
                                    if suffix:
                                        full_response.append(suffix)
                                        if on_delta:
                                            on_delta(suffix)
                    else:
                        # WS先terminal → 等待REST poll完成以reconcile
                        terminal = ws_task.result()
                        if terminal.get("transport_error"):
                            full_response.clear()
                        # 等待REST poll完成（有超时保护：剩余timeout_seconds硬deadline）
                        try:
                            rest_done, rest_pending = await asyncio.wait(
                                [poll_task],
                                timeout=self.config.timeout_seconds,
                            )
                        finally:
                            await ws.close()
                        # Cancel REST poll if still running (shouldn't happen
                        # since _poll_result has its own deadline, but safety net)
                        for t in (rest_pending or []):
                            t.cancel()
                            try:
                                await t
                            except (asyncio.CancelledError, Exception):
                                pass

                        if rest_done and not poll_task.cancelled():
                            try:
                                authoritative = poll_task.result()
                            except Exception as exc:
                                logger.warning(
                                    "North invocation status reconciliation unavailable: %s",
                                    exc,
                                )
                                # REST reconcile failed/excepted — if WS also
                                # had transport_error, surface structured failed
                                if terminal.get("transport_error"):
                                    terminal["status"] = "failed"
                                    terminal.setdefault("error", str(exc))
                            else:
                                if authoritative.get("status"):
                                    terminal["status"] = authoritative["status"]
                                if authoritative.get("required_action"):
                                    terminal["required_action"] = authoritative["required_action"]
                                terminal["result"] = authoritative
                                rest_tools = self._extract_result_tool_blocks(authoritative)
                                if rest_tools:
                                    terminal.setdefault("tools", []).extend(rest_tools)
                        elif terminal.get("transport_error"):
                            # WS errored and no REST result — return structured failed
                            terminal["status"] = "failed"
                            terminal.setdefault("error", terminal["transport_error"])
                else:
                    # No invocation_id to poll — consume WS only
                    try:
                        terminal = await ws_task
                    finally:
                        await ws.close()
                    if terminal.get("transport_error"):
                        terminal["status"] = "failed"
                        terminal.setdefault("error", terminal["transport_error"])
            elif invocation_id:
                terminal = await self._poll_result(client, invocation_id, full_response, on_event, on_delta=on_delta)
                # REST-only path: extract tool_use blocks from raw result into tools
                if terminal.get("blocks") and not terminal.get("tools"):
                    rest_tools = self._extract_result_tool_blocks(terminal)
                    if rest_tools:
                        terminal["tools"] = rest_tools

        with self._active_lock:
            if self._active_invocations.get(session_key) == str(invocation_id or ""):
                self._active_invocations.pop(session_key, None)

        text = "".join(full_response)
        status = str(terminal.get("status") or "completed")
        # North 0.3.3 may emit an ask_user tool result that says it is
        # waiting, without persisting a resumable requires_action invocation.
        # Do not synthesize a pause in that case: the answer endpoint would
        # reject it with required_action_not_found. Surface the mismatch as a
        # failed bridge turn until the backend exposes a real action state.
        if status != "requires_action":
            _tool_events = terminal.get("tools", []) or []
            _ask_user_waiting = any(
                event.get("type") == "tool_call_start"
                and str(event.get("toolCallName") or event.get("tool_name") or "") == "ask_user"
                for event in _tool_events
            ) and any(
                event.get("type") == "tool_call_result"
                and "waiting for user" in str(event.get("content") or "").lower()
                for event in _tool_events
            )
            if _ask_user_waiting:
                status = "error"
                terminal["status"] = status
                terminal["error"] = "North emitted an ask_user wait result without a resumable requires_action invocation"
        requires_action = status == "requires_action"
        # error after a terminal tool (notably complete_task) already returned
        # visible content. Hermes treats that tool result as the turn result;
        # preserve it instead of converting a successful turn into silence.
        if not text and status in {"completed", "error", "failed"}:
            for tool_event in reversed(terminal.get("tools", [])):
                if tool_event.get("type") != "tool_call_result" or tool_event.get("isError"):
                    continue
                candidate = tool_event.get("content") or tool_event.get("result")
                if isinstance(candidate, str) and candidate:
                    text = candidate
                    status = "completed"
                    break
        requires_action = status == "requires_action"
        # Gateway direct path (workspace_switch_supported=False): detect and
        # reject workspace_switch intents with a structured unsupported response.
        if not self.config.workspace_switch_supported:
            ws = self.extract_workspace_switch(terminal)
            if ws is not None:
                logger.warning(
                    "Workspace switch intent detected in Gateway direct runtime "
                    "for session %s — workspace_switch_supported=False; rejected",
                    session_key,
                )
                return {
                    "final_response": (
                        "Project switch is not supported in this runtime mode. "
                        "Use the terminal or a North TUI session to switch projects."
                    ),
                    "messages": [{"role": "user", "content": message}, {"role": "assistant", "content": ""}],
                    "api_calls": 1,
                    "tools": terminal.get("tools", []),
                    "completed": False,
                    "interrupted": False,
                    "failed": True,
                    "status": "unsupported",
                    "session_id": hermes_session_id,
                    "north_conversation_id": conversation_id,
                    "north_invocation_id": invocation_id,
                    "history_offset": 0,
                    "response_previewed": False,
                }
        return {
            "final_response": text or ("⏸️ North Coder is waiting for user action." if requires_action else "(No response from North Coder)"),
            "messages": [{"role": "user", "content": message}, {"role": "assistant", "content": text}],
            "api_calls": 1,
            "tools": terminal.get("tools", []),
            "completed": status == "completed",
            "interrupted": status in {"cancelled", "canceled"},
            "failed": status in {"failed", "error"},
            "requires_action": requires_action,
            "status": status,
            "required_action": terminal.get("required_action"),
            "session_id": hermes_session_id,
            "north_conversation_id": conversation_id,
            "north_invocation_id": invocation_id,
            "history_offset": 0,
            "response_previewed": bool(on_delta and text),
        }

    def extract_workspace_switch(self, terminal: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Extract a workspace_switch intent from tool events in the result.

        Two detection paths:

        Phase 1-2 (WS events via tool_call_start+tool_call_result):
          Correlates tool_call_start events (carrying the tool name) with
          tool_call_result events (carrying the result content) by tool call id.
          Returns the parsed ``workspace_switch`` dict on success, or None on
          any validation failure.

        Phase 3 (REST tool_use blocks — stop-tool format):
          Real North 0.4 stop tools appear as ``tool_use`` blocks in the REST
          ``result.blocks`` or ``terminal["tools"]``. Content is a JSON string
          with ``name`` and ``input`` keys. No tool_call_start/result pairing.

        Rejection rules (must all pass):
          - Tool name must be ``project_switch``
          - Result must not have ``isError`` set (Phase 1-2 only)
          - Content must be valid JSON
          - JSON must have ``success: true`` and ``workspace_switch`` key (Phase 1-2)
            OR ``name: "project_switch"`` with ``input.project`` (Phase 3)
          - When ``workspace_switch.path`` present, must be absolute (Phase 1-2)
        """
        tools = terminal.get("tools") or []
        result_blocks = ((terminal.get("result") or {}).get("blocks") or []) if not tools else []
        if not tools and not result_blocks:
            return None

        # Phase 1: collect tool call name mappings by id
        tool_names: dict[str, str] = {}
        for event in tools:
            if not isinstance(event, dict):
                continue
            kind = str(event.get("type") or "")
            call_id = str(event.get("toolCallId") or event.get("tool_call_id") or "")
            if not call_id:
                continue
            if kind == "tool_call_start":
                name = str(event.get("toolCallName") or event.get("tool_name") or "")
                if name:
                    tool_names[call_id] = name

        # Phase 2: inspect tool_call_result events
        for event in tools:
            if not isinstance(event, dict):
                continue
            kind = str(event.get("type") or "")
            if kind not in {"tool_call_result", "tool_call_end"}:
                continue
            if event.get("isError"):
                continue
            call_id = str(event.get("toolCallId") or event.get("tool_call_id") or "")
            tool_name = tool_names.get(call_id) or str(event.get("toolCallName") or event.get("tool_name") or "")
            if tool_name != "project_switch":
                continue
            content = event.get("content") or event.get("result")
            if not isinstance(content, str) or not content:
                continue
            try:
                data = json.loads(content)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            if not data.get("success"):
                continue
            ws = data.get("workspace_switch")
            if not isinstance(ws, dict):
                continue
            path = ws.get("path")
            if not isinstance(path, str) or not path:
                continue
            if not os.path.isabs(path):
                continue
            if not Path(path).expanduser().resolve().is_dir():
                continue
            return {
                "project_id": str(ws.get("project_id", "")),
                "project_name": str(ws.get("project_name", "")),
                "path": str(Path(path).expanduser().resolve()),
            }

        # Phase 3: inspect tool_use blocks (REST stop-tool format, North 0.4)
        # These may be in terminal["tools"] or terminal["result"]["blocks"]
        tool_use_events: list[dict[str, Any]] = [
            e
            for e in tools
            if isinstance(e, dict)
            and str(e.get("block_type") or e.get("type") or "") == "tool_use"
        ]
        if not tool_use_events:
            result_blocks = (terminal.get("result") or {}).get("blocks") or []
            for blk in result_blocks:
                if (
                    isinstance(blk, dict)
                    and str(blk.get("block_type") or blk.get("type") or "") == "tool_use"
                ):
                    tool_use_events.append(blk)

        for event in tool_use_events:
            content = event.get("content")
            if not isinstance(content, str) or not content:
                continue
            try:
                data = json.loads(content)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            if data.get("name") != "project_switch":
                continue
            inp = data.get("input", {})
            if not isinstance(inp, dict):
                continue
            project = inp.get("project", "")
            if not project:
                continue
            return {
                "project_id": str(project),
                "project_name": str(project),
                "name": "project_switch",
                "input": inp,
            }
        return None

    async def cancel(self, invocation_id: str) -> None:
        import aiohttp

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as client:
            async with client.post(f"{self.config.base_url}/api/invocations/{invocation_id}/cancel") as response:
                if response.status >= 400:
                    raise RuntimeError(f"North cancel failed ({response.status}): {(await response.text())[:300]}")

    def active_invocation(self, session_key: str) -> Optional[str]:
        with self._active_lock:
            return self._active_invocations.get(session_key)

    def _request_session_cancel(self, session_key: str) -> Optional[str]:
        """Record a turn cancellation and return an accepted invocation id."""
        with self._active_lock:
            invocation_id = self._active_invocations.get(session_key)
            if not invocation_id and session_key in self._inflight_sessions:
                self._cancel_requested_sessions.add(session_key)
            return invocation_id

    async def cancel_session_async(self, session_key: str) -> None:
        """Cancel a North turn from an async Gateway command handler.

        Idempotent: at most one POST per invocation_id, even when called
        repeatedly for the same in-flight turn.
        """
        invocation_id = self._request_session_cancel(session_key)
        if invocation_id:
            with self._active_lock:
                if invocation_id in self._cancel_posted:
                    return
                self._cancel_posted.add(invocation_id)
            await self.cancel(invocation_id)

    def cancel_session(self, session_key: str) -> None:
        """Best-effort synchronous cancellation hook for Gateway/TUI threads.

        Idempotent: at most one POST per invocation_id.
        """
        invocation_id = self._request_session_cancel(session_key)
        if not invocation_id:
            return
        with self._active_lock:
            if invocation_id in self._cancel_posted:
                return
            self._cancel_posted.add(invocation_id)
        try:
            asyncio.run(self.cancel(invocation_id))
        except Exception:
            logger.exception("Failed to cancel North invocation %s", invocation_id)

    async def resolve_permission(
        self,
        invocation_id: str,
        tool_call_id: str,
        decision: str,
    ) -> dict[str, Any]:
        """Resolve a North permission pause and return the resume response."""
        if decision not in {"allow", "allow_once", "deny"}:
            raise ValueError(f"unsupported permission decision: {decision}")
        import aiohttp

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as client:
            async with client.post(
                f"{self.config.base_url}/api/invocations/{invocation_id}/permissions/{tool_call_id}/resolve",
                json={"decision": decision},
            ) as response:
                body = await response.text()
                if response.status >= 400:
                    raise RuntimeError(f"North permission resolve failed ({response.status}): {body[:500]}")
                return json.loads(body) if body else {}

    async def answer_ask_user(
        self,
        invocation_id: str,
        answers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Submit Hermes-compatible ask_user answers and resume the run."""
        import aiohttp

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as client:
            async with client.get(f"{self.config.base_url}/api/invocations/{invocation_id}") as response:
                invocation_body = await response.text()
                if response.status >= 400:
                    raise RuntimeError(f"North invocation lookup failed ({response.status}): {invocation_body[:500]}")
                invocation = json.loads(invocation_body)
            if invocation.get("status") != "requires_action":
                raise RuntimeError(f"North invocation is not waiting for action: {invocation.get('status')}")
            conversation_id = str(invocation.get("conversation_id") or "")
            action = invocation.get("required_action") or {}
            tool_call_id = str(action.get("action_id") or action.get("tool_call_id") or "")
            if not conversation_id or not tool_call_id:
                raise RuntimeError("North requires_action response lacks conversation_id or action_id")
            lines = ["[Ask User Response]"]
            normalized: list[dict[str, Any]] = []
            for index, answer in enumerate(answers):
                item = dict(answer)
                item.setdefault("questionIndex", index)
                item.setdefault("header", f"Question {index + 1}")
                item.setdefault("type", "text")
                item.setdefault("value", "")
                normalized.append(item)
                lines.append(f"{index + 1}. {item['header']}: {item['value']}")
            payload = {
                "content": "\n".join(lines),
                "metadata": {
                    "ask_user_response": {
                        "tool_call_id": tool_call_id,
                        "answers": normalized,
                    }
                },
            }
            async with client.post(
                f"{self.config.base_url}/api/conversations/{conversation_id}/messages",
                json=payload,
            ) as response:
                body = await response.text()
                if response.status >= 400:
                    raise RuntimeError(f"North ask_user answer failed ({response.status}): {body[:500]}")
                return json.loads(body) if body else {}

    async def resume_ask_user_turn(
        self,
        invocation_id: str,
        answers: list[dict[str, Any]],
        *,
        session_key: str,
        hermes_session_id: str,
        source: Any = None,
        on_delta: Optional[DeltaCallback] = None,
        on_event: Optional[EventCallback] = None,
    ) -> dict[str, Any]:
        """Submit a pending ask_user action and consume the resumed invocation."""
        import aiohttp

        conversation_id = await self._conversation_id(session_key)
        full_response: list[str] = []
        terminal: dict[str, Any] = {"status": "completed", "tools": []}
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds, sock_read=self.config.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as client:
            try:
                ws = await self._connect_events(client, conversation_id)
            except Exception:
                ws = None
            # Subscribe before posting the answer. answer_ask_user performs the
            # authoritative status check and POST in a separate client, while
            # this subscription closes the fast-run replay race.
            accepted = await self.answer_ask_user(invocation_id, answers)
            resumed_id = str(accepted.get("invocation_id") or invocation_id)
            if ws is not None:
                try:
                    terminal = await self._consume_events(ws, full_response, on_delta, on_event)
                finally:
                    await ws.close()
            else:
                terminal = await self._poll_result(client, resumed_id, full_response, on_event, on_delta=on_delta)
        text = "".join(full_response)
        status = str(terminal.get("status") or "completed")
        return {
            "final_response": text or ("⏸️ North Coder is waiting for user action." if status == "requires_action" else "(No response from North Coder)"),
            "messages": [{"role": "user", "content": "[Ask User Response]"}, {"role": "assistant", "content": text}],
            "api_calls": 1,
            "tools": terminal.get("tools", []),
            "completed": status == "completed",
            "interrupted": status in {"cancelled", "canceled"},
            "failed": status in {"failed", "error"},
            "requires_action": status == "requires_action",
            "required_action": terminal.get("required_action"),
            "north_conversation_id": conversation_id,
            "north_invocation_id": resumed_id,
        }

    def resolve_permission_sync(self, invocation_id: str, tool_call_id: str, decision: str) -> dict[str, Any]:
        return asyncio.run(self.resolve_permission(invocation_id, tool_call_id, decision))

    def answer_ask_user_sync(self, invocation_id: str, answers: list[dict[str, Any]]) -> dict[str, Any]:
        return asyncio.run(self.answer_ask_user(invocation_id, answers))

    async def _connect_events(self, client: Any, conversation_id: str) -> Any:
        ws_url = self.config.base_url.replace("http://", "ws://", 1).replace("https://", "wss://", 1)
        return await client.ws_connect(f"{ws_url}/ws/conversation/{conversation_id}", heartbeat=30)

    async def _consume_events(
        self,
        ws: Any,
        full_response: list[str],
        on_delta: Optional[DeltaCallback],
        on_event: Optional[EventCallback],
        *,
        process_replay: bool = False,
        stream_text: bool = True,
    ) -> dict[str, Any]:
        replaying = False
        terminal: dict[str, Any] = {"status": "completed", "tools": []}
        while True:
            try:
                item = await asyncio.wait_for(
                    ws.receive(),
                    timeout=self.config.timeout_seconds,
                )
            except (asyncio.TimeoutError, TimeoutError):
                logger.warning(
                    "North event stream timed out (no message for %.0fs); "
                    "falling back to REST polling for invocation",
                    self.config.timeout_seconds,
                )
                terminal["transport_error"] = f"event stream timeout ({self.config.timeout_seconds}s)"
                break
            if item.type.name in {"CLOSED", "CLOSE", "CLOSING"}:
                break
            if item.type.name == "ERROR":
                logger.warning(
                    "North event stream error for session; falling back to REST polling: %s",
                    ws.exception(),
                )
                terminal["transport_error"] = str(ws.exception() or "North event stream error")
                break
            if item.type.name != "TEXT":
                continue
            event = json.loads(item.data)
            kind = event.get("type")
            if kind == "replay_start":
                replaying = True
            elif kind == "replay_end":
                replaying = False
            if on_event:
                result = on_event(event)
                if asyncio.iscoroutine(result):
                    await result
            if (replaying and not process_replay) or kind in {"history_ref", "replay_start", "replay_end"}:
                continue
            if kind == "text_message_content" and stream_text:
                delta = str(event.get("delta") or "")
                if delta:
                    full_response.append(delta)
                    if on_delta:
                        on_delta(delta)
            if kind in {"tool_call_start", "tool_call_args", "tool_call_end", "tool_call_result"}:
                terminal.setdefault("tools", []).append(event)
            if kind in {"requires_action", "run_requires_action", "invocation_requires_action", "permission_request", "ask_user"}:
                terminal["status"] = "requires_action"
                terminal["required_action"] = event.get("required_action") or event
            if kind in {"run_finished", "run_error", "run_start_failure", "cancelled"}:
                if terminal.get("status") != "requires_action":
                    terminal["status"] = {
                        "run_finished": "completed",
                        "run_error": "error",
                        "run_start_failure": "error",
                        "cancelled": "cancelled",
                    }[kind]
                if event.get("error"):
                    terminal["error"] = event["error"]
                break
        return terminal

    @staticmethod
    def _extract_ask_user_questions(result: dict[str, Any]) -> list[dict[str, Any]]:
        blocks = result.get("blocks")
        if not isinstance(blocks, list):
            return []
        for block in reversed(blocks):
            if (
                not isinstance(block, dict)
                or str(block.get("block_type") or block.get("type") or "") != "tool_use"
            ):
                continue
            payload = block
            if not payload.get("name") and isinstance(payload.get("content"), str):
                try:
                    parsed = json.loads(payload["content"])
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    payload = parsed
            if payload.get("name") != "ask_user":
                continue
            input_payload = payload.get("input")
            if isinstance(input_payload, dict) and isinstance(input_payload.get("questions"), list):
                return [item for item in input_payload["questions"] if isinstance(item, dict)]
        return []

    @staticmethod
    def _extract_result_tool_blocks(result: dict[str, Any]) -> list[dict[str, Any]]:
        """Return North REST tool-use blocks across 0.3/0.4 field names."""
        blocks = result.get("blocks")
        if not isinstance(blocks, list):
            return []
        return [
            block
            for block in blocks
            if isinstance(block, dict)
            and str(block.get("block_type") or block.get("type") or "") == "tool_use"
        ]

    @staticmethod
    def _extract_result_text(result: dict[str, Any]) -> str:
        direct = result.get("text") or result.get("content") or result.get("response")
        if isinstance(direct, str) and direct:
            return direct
        blocks = result.get("blocks")
        if not isinstance(blocks, list):
            return ""
        parts: list[str] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            role = str(block.get("role") or "")
            block_type = str(block.get("block_type") or block.get("type") or "")
            content = block.get("content") or block.get("text")
            if role == "assistant" and block_type == "text" and isinstance(content, str):
                parts.append(content)
        return "".join(parts)

    @staticmethod
    def _extract_final_result_text(result: dict[str, Any]) -> str:
        """Return only the terminal assistant block, excluding process narration."""
        blocks = result.get("blocks")
        if not isinstance(blocks, list):
            return NorthCoderRuntime._extract_result_text(result)
        for index in range(len(blocks) - 1, -1, -1):
            block = blocks[index]
            if not isinstance(block, dict):
                continue
            role = str(block.get("role") or "")
            block_type = str(block.get("block_type") or block.get("type") or "")
            if role != "assistant" or block_type != "text":
                continue
            # A text block followed by another assistant action is progress
            # narration, not the terminal answer.
            has_later_action = any(
                isinstance(later, dict)
                and str(later.get("role") or "") == "assistant"
                and str(later.get("block_type") or later.get("type") or "")
                in {"text", "tool_use", "tool_result", "background_task"}
                for later in blocks[index + 1 :]
            )
            if not has_later_action:
                content = block.get("content") or block.get("text")
                return content if isinstance(content, str) else ""
        return ""

    async def _emit_rest_process_events(
        self,
        result: dict[str, Any],
        on_event: Optional[EventCallback],
        seen: set[tuple[str, str]],
    ) -> None:
        """Map newly persisted North blocks onto Hermes process/tool events."""
        if on_event is None:
            return
        blocks = result.get("blocks")
        if not isinstance(blocks, list):
            return
        for index, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            role = str(block.get("role") or "")
            block_type = str(block.get("block_type") or block.get("type") or "")
            position = str(block.get("position") if block.get("position") is not None else index)
            key = (position, block_type)
            if key in seen or role != "assistant":
                continue

            event: Optional[dict[str, Any]] = None
            content = block.get("content") or block.get("text") or ""
            if block_type == "text" and isinstance(content, str) and content:
                has_later_action = any(
                    isinstance(later, dict)
                    and str(later.get("role") or "") == "assistant"
                    and str(later.get("block_type") or later.get("type") or "")
                    in {"text", "tool_use", "tool_result", "background_task"}
                    for later in blocks[index + 1 :]
                )
                if has_later_action:
                    event = {"type": "assistant_process", "text": content, "position": position}
            elif block_type == "tool_use":
                payload = block
                if isinstance(content, str):
                    try:
                        parsed = json.loads(content)
                    except (TypeError, ValueError):
                        parsed = None
                    if isinstance(parsed, dict):
                        payload = parsed
                event = {
                    "type": "tool_call_start",
                    "toolCallId": str(payload.get("id") or payload.get("tool_call_id") or position),
                    "toolCallName": payload.get("name") or payload.get("tool_name"),
                    "input": payload.get("input") or payload.get("args") or {},
                }
            elif block_type == "tool_result":
                payload = block
                if isinstance(content, str):
                    try:
                        parsed = json.loads(content)
                    except (TypeError, ValueError):
                        parsed = None
                    if isinstance(parsed, dict):
                        payload = parsed
                event = {
                    "type": "tool_call_result",
                    "toolCallId": str(
                        payload.get("toolUseId")
                        or payload.get("tool_call_id")
                        or position
                    ),
                    "content": payload.get("content") or payload.get("result") or "",
                    "isError": bool(payload.get("isError") or payload.get("is_error")),
                }
            if event is None:
                continue
            seen.add(key)
            callback_result = on_event(event)
            if asyncio.iscoroutine(callback_result):
                await callback_result

    async def _poll_result(self, client: Any, invocation_id: str, full_response: list[str], on_event: Optional[EventCallback], *, on_delta: Optional[DeltaCallback] = None) -> dict[str, Any]:
        deadline = time.monotonic() + self.config.timeout_seconds
        seen_process_events: set[tuple[str, str]] = set()
        while time.monotonic() < deadline:
            async with client.get(f"{self.config.base_url}/api/invocations/{invocation_id}/result") as response:
                if response.status >= 400:
                    raise RuntimeError(f"North result request failed ({response.status})")
                result = json.loads(await response.text())
            if on_event:
                event = {"type": "invocation_result", "result": result}
                callback_result = on_event(event)
                if asyncio.iscoroutine(callback_result):
                    await callback_result
            if self.config.tui_variant:
                await self._emit_rest_process_events(result, on_event, seen_process_events)
            text = (
                self._extract_final_result_text(result)
                if self.config.tui_variant
                else self._extract_result_text(result)
            )
            status = str(result.get("status") or "")
            terminal_status = status in {
                "completed", "requires_action", "failed", "cancelled", "error"
            } or bool(result.get("finished"))
            if text and (terminal_status or not self.config.tui_variant):
                # Incremental delta: only append new suffix not yet in full_response
                current = "".join(full_response)
                if not current.startswith(text) and not text.startswith(current):
                    # Different content or first chunk — use text as-is
                    if not full_response:
                        full_response.append(text)
                        if on_delta:
                            on_delta(text)
                elif len(text) > len(current):
                    # Text has grown — only emit the new suffix
                    suffix = text[len(current):]
                    if suffix:
                        full_response.append(suffix)
                        if on_delta:
                            on_delta(suffix)
            required_action = result.get("required_action")
            if isinstance(required_action, dict) and required_action.get("type") == "ask_user":
                questions = self._extract_ask_user_questions(result)
                if questions and not required_action.get("questions"):
                    required_action = {**required_action, "questions": questions}
                    result["required_action"] = required_action
            if status in {"completed", "requires_action", "failed", "cancelled", "error"} or result.get("finished"):
                if status not in {"completed", ""}:
                    return result
                return result
            await asyncio.sleep(0.25)
        raise TimeoutError(f"North invocation {invocation_id} timed out")

    async def _conversation_id(self, session_key: str) -> str:
        conversation_id, _ = await self._conversation_for_turn(session_key)
        return conversation_id

    @staticmethod
    def _is_workdir_register_failed(status: int, body: str) -> bool:
        """Recognize North's pre-invocation workdir registration failure."""
        if status != 500:
            return False
        try:
            payload = json.loads(body)
        except (TypeError, ValueError):
            return False
        if not isinstance(payload, dict):
            return False
        detail = payload.get("detail")
        return isinstance(detail, dict) and detail.get("code") == "workdir_register_failed"

    @staticmethod
    def _decode_binding(
        value: Any,
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        if isinstance(value, str) and value:
            return value, None, None
        if isinstance(value, dict):
            conversation_id = str(value.get("conversation_id") or "")
            workdir = str(value.get("workdir") or "")
            workspace_id = str(value.get("workspace_id") or "")
            if value.get("workdir_only") is True:
                workspace_id = "__hermes_workdir_only__"
            return conversation_id or None, workdir or None, workspace_id or None
        return None, None, None

    async def _conversation_binding(
        self,
        session_key: str,
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        with self._state_lock:
            return self._decode_binding(self._read_state().get(session_key))

    async def _record_conversation_binding(
        self,
        session_key: str,
        conversation_id: str,
        workdir: Optional[str],
        workspace_id: Optional[str],
        *,
        workdir_only: bool = False,
    ) -> None:
        with self._state_lock:
            state = self._read_state()
            binding = {
                "conversation_id": conversation_id,
                "workdir": workdir,
                "workspace_id": workspace_id,
            }
            if workdir_only:
                binding["workdir_only"] = True
            state[session_key] = binding
            self._write_state(state)

    async def _conversation_for_turn(self, session_key: str) -> tuple[str, bool]:
        with self._state_lock:
            state = self._read_state()
            existing, _, _ = self._decode_binding(state.get(session_key))
            if existing:
                return existing, False

        conversation_id = await self._create_conversation(session_key)
        with self._state_lock:
            state = self._read_state()
            existing, _, _ = self._decode_binding(state.get(session_key))
            if existing:
                return existing, False
            state[session_key] = conversation_id
            self._write_state(state)
        return conversation_id, True

    def detach_session(self, session_key: str) -> None:
        """Forget provider state so the next North turn seeds Gateway history."""
        with self._state_lock:
            state = self._read_state()
            if session_key in state:
                state.pop(session_key, None)
                self._write_state(state)

    @staticmethod
    def _seeded_message(
        message: str,
        conversation_history: Optional[list[dict[str, Any]]],
        seed_token_budget: int = 2000,
    ) -> str:
        """Project the canonical Gateway transcript into a fresh provider lane.

        Only called for fresh conversations (created=True). From back to
        front, retains the most recent user/assistant pairs up to
        seed_token_budget. Uses rough estimate (len+3)//4 ≈ 4 chars/token.
        Never returns empty — at minimum the current message is preserved.
        """
        if not conversation_history:
            return message
        projected: list[tuple[str, str]] = []
        budget = seed_token_budget
        for item in reversed(conversation_history):
            role = str(item.get("role") or "")
            if role not in {"user", "assistant"}:
                continue
            content = item.get("content")
            if not isinstance(content, str) or not content:
                continue
            entry_text = f"{role}:\n{content}"
            tok = (len(entry_text) + 3) // 4
            if tok > budget:
                if not projected:
                    projected.append((role, content[:budget * 4]))
                break
            budget -= tok
            projected.append((role, content))
        projected.reverse()
        if not projected:
            return message
        history_text = "\n\n".join(
            f"{role}:\n{content}" for role, content in projected
        )
        return (
            "[Prior Hermes conversation; context only]\n"
            f"{history_text}\n\n"
            "[Current user message]\n"
            f"{message}"
        )

    def _refresh_managed_profile(self) -> Optional[Path]:
        """Refresh the generated Hermes artifact at a new-conversation boundary.

        Only the canonical profile path under this Hermes home is managed. An
        arbitrary user-supplied North ``agent_yaml_path`` is never overwritten.

        When ``tui_variant`` is True, writes ``agent-tui.yaml`` (with project
        switch tools).  Otherwise writes ``agent.yaml`` (Gateway direct path).

        Returns the path to the generated YAML, or None if the profile is not
        managed (user-specified path).
        """
        if not self.config.agent_yaml_path:
            return None
        configured = Path(self.config.agent_yaml_path).expanduser().resolve()
        managed_dir = self.hermes_home / "north-coder-profile"
        managed = (managed_dir / "agent.yaml").resolve()

        # Non-managed agent_yaml_path: fail closed — don't inject project tools
        # or fake a variant that wasn't configured.
        if configured == managed and not self.config.tui_variant:
            pass  # normal managed path, write agent.yaml
        elif configured == (managed_dir / "agent-tui.yaml").resolve() and self.config.tui_variant:
            pass  # TUI managed path, write agent-tui.yaml
        else:
            # User explicitly configured a non-managed path — do not overwrite.
            if not str(configured).startswith(str(managed_dir.resolve())):
                return None

        from hermes_cli.north_coder_profile import export_hermes_profile

        profile_name = self.config.agent_profile_id.replace(":", "-") or "hermes-default"
        export_hermes_profile(
            self.hermes_home, managed_dir, name=profile_name,
            tui_variant=self.config.tui_variant,
        )
        return Path(str(self.config.agent_yaml_path))

    async def _create_conversation(self, session_key: str) -> str:
        import aiohttp

        self._refresh_managed_profile()
        agent_config = {"agent_profile_id": self.config.agent_profile_id}
        if self.config.model_id:
            agent_config["model_id"] = self.config.model_id
        if self.config.agent_yaml_path:
            agent_config["agent_yaml_path"] = self.config.agent_yaml_path
        payload = {
            "title": f"Hermes {session_key[-80:]}",
            "agent_config": agent_config,
            "metadata": {"hermes_session_key": session_key, "source": "hermes-gateway"},
        }
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as client:
            async with client.post(f"{self.config.base_url}/api/workspaces/{self.config.workspace_id}/conversations", json=payload) as response:
                body = await response.text()
                if response.status >= 400:
                    raise RuntimeError(f"North conversation creation failed ({response.status}): {body[:500]}")
                data = json.loads(body)
        conversation_id = data.get("id") or data.get("conversation_id")
        if not conversation_id:
            raise RuntimeError(f"North conversation response lacks id: {data}")
        return str(conversation_id)

    def _read_state(self) -> dict[str, Any]:
        try:
            if self.state_file.exists():
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
        except Exception:
            logger.warning("Ignoring unreadable North conversation map %s", self.state_file)
        return {}

    def _write_state(self, state: dict[str, Any]) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        temp = self.state_file.with_suffix(self.state_file.suffix + ".tmp")
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, self.state_file)


_RUNTIME_CACHE: dict[tuple[str, str, str], "NorthCoderRuntime"] = {}


def runtime_from_raw(raw: dict[str, Any], hermes_home: Path) -> Optional[NorthCoderRuntime]:
    config = NorthCoderRuntimeConfig.from_raw(raw)
    if not config:
        return None
    key = (config.base_url, config.workspace_id, str(Path(config.state_file) if config.state_file else hermes_home / "north_coder_conversations.json"))
    runtime = _RUNTIME_CACHE.get(key)
    if runtime is None:
        runtime = NorthCoderRuntime(config, hermes_home)
        _RUNTIME_CACHE[key] = runtime
    return runtime


class NorthCoderTUIAgent:
    """A small AIAgent-shaped facade for the stdio TUI.

    The TUI invokes ``run_conversation`` synchronously from its worker thread;
    the external runtime remains async and owns the real conversation state.

    The TUI variant enables workspace switch support and uses agent-tui.yaml.
    """

    def __init__(self, runtime: NorthCoderRuntime, session_key: str,
                 _review_host_factory=None, workspace_switch_callback=None):
        import dataclasses
        # TUI explicitly supports workspace switching and uses the TUI variant
        self.runtime = runtime
        # Patch the runtime config to use tui_variant and workspace_switch_supported
        if not runtime.config.tui_variant or not runtime.config.workspace_switch_supported:
            self.runtime = NorthCoderRuntime(
                dataclasses.replace(
                    runtime.config,
                    tui_variant=True,
                    workspace_switch_supported=True,
                    agent_yaml_path=str(runtime.hermes_home / "north-coder-profile" / "agent-tui.yaml") if (
                        runtime.config.agent_yaml_path is None
                        or str(Path(runtime.config.agent_yaml_path).expanduser().resolve())
                        == str((runtime.hermes_home / "north-coder-profile" / "agent.yaml").resolve())
                    ) else runtime.config.agent_yaml_path,
                ),
                runtime.hermes_home,
            )
        self.session_key = session_key
        self.session_id = session_key
        self.model = "north-coder"
        self.provider = "north_coder"
        self.runtime_override = "ncoder"
        # Native AIAgent persists its own transcript.  North does not, so the
        # TUI gateway must append this facade's canonical messages to SessionDB.
        self.gateway_managed_history_persistence = True
        self.history: list[dict[str, Any]] = []
        self._last_invocation_id: Optional[str] = None
        self._interrupted = False
        # workspace_switch_callback is a callable that receives the extracted
        # workspace_switch dict and returns None on success or an error string
        # on failure.  Set by _make_agent in tui_gateway/server.py for North
        # TUI sessions; None in Gateway direct-runtime paths.
        self._workspace_switch_callback = workspace_switch_callback
        # Cache the last extracted workspace_switch so run_conversation can
        # apply it after the turn completes but before background review.
        self._pending_workspace_switch: Optional[dict[str, Any]] = None
        # Lazy factory for the Hermes AIAgent review host.  Only invoked when
        # nudge thresholds are met — never per-turn — to avoid building a
        # second runtime on every North foreground turn.
        self._review_host_factory = _review_host_factory
        # Cached background review host — built lazily on first threshold met,
        # then reused across turns until facade.close().
        self._review_host = None

    def run_conversation(self, message: Any, *, conversation_history=None, stream_callback=None,
                         tool_start_callback=None, tool_complete_callback=None,
                         tool_progress_callback=None, reasoning_callback=None,
                         task_id=None, **kwargs):
        text = message if isinstance(message, str) else str(message)
        if conversation_history:
            self.history = list(conversation_history)

        tool_names: dict[str, Any] = {}

        def on_event(event: dict[str, Any]) -> None:
            kind = str(event.get("type") or "")
            if kind == "assistant_process":
                process_text = str(event.get("text") or "")
                if process_text and reasoning_callback:
                    reasoning_callback(process_text)
                return
            if not (tool_start_callback or tool_complete_callback or tool_progress_callback):
                return
            tool_name = event.get("toolCallName") or event.get("tool_name")
            call_id = str(event.get("toolCallId") or event.get("tool_call_id") or "")
            if not tool_name:
                tool_name = tool_names.get(call_id)
            if kind == "tool_call_start":
                tool_names[call_id] = tool_name
                if tool_start_callback:
                    tool_start_callback(call_id, tool_name, event)
                if tool_progress_callback:
                    tool_progress_callback(
                        "tool.started", name=tool_name, args=event,
                    )
            elif kind in {"tool_call_result", "tool_call_end"}:
                call_id = str(event.get("toolCallId") or event.get("tool_call_id") or "")
                content = str(event.get("content") or "")
                if tool_complete_callback:
                    tool_complete_callback(call_id, tool_name, event, content)
                if tool_progress_callback:
                    tool_progress_callback(
                        "tool.completed", name=tool_name,
                        preview=content[:240], args=event,
                    )

        from agent.runtime_cwd import resolve_agent_cwd

        result = asyncio.run(
            self.runtime.run_turn(
                message=text,
                session_key=self.session_key,
                hermes_session_id=self.session_id,
                conversation_history=self.history,
                workdir=str(resolve_agent_cwd()),
                on_delta=stream_callback,
                on_event=on_event,
            )
        )
        self._last_invocation_id = result.get("north_invocation_id")
        self.history.extend(result.get("messages", []))
        # TUI server persists ``result["messages"]`` as the canonical Hermes
        # transcript. North's run result contains only the current exchange,
        # so return the facade's full history instead of letting the server
        # overwrite prior turns with the latest pair.
        result["messages"] = list(self.history)

        # ── Workspace switch intent handling ─────────────────────────────
        # Extract workspace_switch from tool results before background review.
        # Callback is injected by _make_agent in tui_gateway/server.py for
        # North TUI sessions; None in Gateway direct-runtime paths.
        self._pending_workspace_switch = self.runtime.extract_workspace_switch(result)
        if self._pending_workspace_switch and self._workspace_switch_callback:
            cb_error = self._workspace_switch_callback(self._pending_workspace_switch)
            if cb_error is None:
                selected_path = str(
                    self._pending_workspace_switch.get("selected_path") or ""
                )
                project_name = str(
                    self._pending_workspace_switch.get("project_name") or "项目"
                )
                success_text = (
                    "已切换工作区\n\n"
                    f"项目：{project_name}\n"
                    f"路径：{selected_path}\n\n"
                    "后续消息将在该路径中执行。"
                )
                result["final_response"] = success_text
                self.history.append({"role": "assistant", "content": success_text})
                result["messages"] = list(self.history)
                # Callback succeeded: detach old North binding so the next turn
                # creates a new North conversation with the correct workdir.
                self.runtime.detach_session(self.session_key)
                self._pending_workspace_switch = None
            else:
                # Callback failed: keep old binding, mark turn as failed.
                error_msg = str(cb_error)
                result["completed"] = False
                result["failed"] = True
                result["status"] = "failed"
                result["final_response"] = (
                    f"⚠️ Project switch blocked: {error_msg}\n\n"
                    "The workspace switch could not be applied. "
                    "Your session remains in the previous context."
                )
                logger.warning(
                    "Workspace switch callback failed for session %s: %s",
                    self.session_key, error_msg,
                )
        elif self._pending_workspace_switch and not self._workspace_switch_callback:
            # Gateway direct-runtime path (no callback): reject as unsupported.
            logger.warning(
                "Project_switch intent detected in Gateway direct runtime "
                "for session %s — no workspace_switch_callback configured; rejected",
                self.session_key,
            )
            result["failed"] = True
            result["status"] = "unsupported"
            result["final_response"] = (
                "⚠️ Project switch is not supported in this runtime mode. "
                "Use the terminal or a North TUI session to switch projects."
            )

        # Schedule background memory/skill review after a successful turn.
        # This runs as a Hermes sidecar — it does NOT create a North background
        # conversation.  The canonical transcript (self.history) MUST include
        # the just-completed exchange so the review sees the full session.
        if result.get("completed") and not result.get("interrupted") and not result.get("requires_action"):
            try:
                self.schedule_background_review(
                    list(self.history),
                    result,
                    background_review_callback=getattr(self, "background_review_callback", None),
                    memory_notifications=getattr(self, "memory_notifications", "on"),
                )
            except Exception:
                logger.warning("North background review scheduling failed", exc_info=True)
        return result

    def interrupt(self) -> None:
        self._interrupted = True
        self.runtime.cancel_session(self.session_key)

    def clear_interrupt(self) -> None:
        self._interrupted = False

    def close(self) -> None:
        """AIAgent compatibility hook; North owns transport cleanup per turn."""
        if self._review_host is not None:
            try:
                self._review_host.close()
            except Exception:
                pass
            self._review_host = None
        with _NORTH_REVIEW_LOCK:
            _NORTH_REVIEW_STATE.pop(self.session_id, None)

    def schedule_background_review(
        self,
        canonical_history: list[dict],
        north_result: dict,
        *,
        background_review_callback=None,
        memory_notifications: str = "on",
    ) -> None:
        """Schedule a Hermes background review after a successful North turn.
        Uses the lazy review host factory (only built when thresholds are met).
        Caches the built host for reuse across turns."""
        def _caching_builder():
            if self._review_host is None and self._review_host_factory is not None:
                self._review_host = self._review_host_factory()
            return self._review_host

        try:
            schedule_north_background_review(
                canonical_history=canonical_history,
                north_result=north_result,
                session_id=self.session_id,
                review_host=None,
                tui_host_builder=_caching_builder,
                background_review_callback=background_review_callback,
                memory_notifications=memory_notifications,
            )
        except Exception:
            logger.warning("North background review scheduling failed", exc_info=True)


def tui_agent_from_raw(
    raw: dict[str, Any],
    hermes_home: Path,
    session_key: str,
    _review_host_factory=None,
    workspace_switch_callback=None,
) -> Optional[NorthCoderTUIAgent]:
    runtime = runtime_from_raw(raw, hermes_home)
    return NorthCoderTUIAgent(runtime, session_key,
                              _review_host_factory=_review_host_factory,
                              workspace_switch_callback=workspace_switch_callback) if runtime else None


# ---------------------------------------------------------------------------
# North background review — Hermes sidecar that runs after every successful
# North foreground turn and determines whether a memory/skill review should
# be triggered, reusing the existing agent/background_review infrastructure.
# ---------------------------------------------------------------------------

# Process-local nudge state keyed by Hermes session_key.
# Stores last-seen canonical turn counts and accumulated deltas.
_NORTH_REVIEW_STATE: dict[str, dict] = {}
_NORTH_REVIEW_LOCK = threading.Lock()


def _count_north_tool_iterations(north_tools: list[dict]) -> int:
    """Count tool iterations from North result tool events.

    Each ``tool_call_start`` event represents one tool iteration.  When
    ``toolCallId`` / ``tool_call_id`` is present, events are deduplicated so
    paired start+result events don't double-count.  When absent (some North
    run formats), each ``tool_call_start`` counts as 1.
    """
    seen = set()
    count = 0
    for event in north_tools or []:
        if not isinstance(event, dict):
            continue
        if event.get("type") == "tool_call_start":
            tcid = str(event.get("toolCallId") or event.get("tool_call_id") or "")
            if tcid:
                if tcid not in seen:
                    seen.add(tcid)
                    count += 1
            else:
                count += 1
    return count


def _count_user_turns(canonical_history: list[dict]) -> int:
    """Count user-role messages in the canonical transcript."""
    return sum(
        1 for m in (canonical_history or [])
        if isinstance(m, dict) and m.get("role") == "user"
    )


def _north_review_snapshot(
    canonical_history: list[dict],
    north_tools: list[dict],
) -> list[dict]:
    """Add truthful North tool activity to the review-only transcript.

    North persists user/assistant messages as the Gateway canonical history,
    while its tool telemetry arrives separately. Native Hermes skill review
    normally sees assistant ``tool_calls`` followed by ``tool`` messages, so
    reconstruct that shape for the reviewer without mutating session history.
    """
    tool_messages: list[dict] = []
    started: set[str] = set()
    for event in north_tools or []:
        if not isinstance(event, dict):
            continue
        kind = str(event.get("type") or "")
        call_id = str(event.get("toolCallId") or event.get("tool_call_id") or "")
        if kind == "tool_call_start":
            if not call_id:
                call_id = f"north-tool-{len(started) + 1}"
            if call_id in started:
                continue
            started.add(call_id)
            name = str(event.get("toolCallName") or event.get("tool_name") or "unknown")
            arguments = event.get("arguments", event.get("args", event.get("input", {})))
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments or {}, ensure_ascii=False, default=str)
            tool_messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }],
            })
        elif kind in {"tool_call_result", "tool_call_end"} and call_id in started:
            content = event.get("content", event.get("result", ""))
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, default=str)
            tool_messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": content,
            })

    snapshot = list(canonical_history)
    if not tool_messages:
        return snapshot
    insert_at = len(snapshot)
    if snapshot and isinstance(snapshot[-1], dict) and snapshot[-1].get("role") == "assistant":
        insert_at -= 1
    snapshot[insert_at:insert_at] = tool_messages
    return snapshot


def schedule_north_background_review(
    *,
    canonical_history: list[dict],
    north_result: dict,
    session_id: str,
    review_host: Any = None,
    tui_host_builder: Optional[Callable[[], Any]] = None,
    background_review_callback=None,
    memory_notifications: str = "on",
) -> None:
    """Schedule a background memory/skill review after a successful North turn.

    Uses diff-based accumulation: the function stores the *last seen* user-turn
    count from the canonical transcript and only accumulates the *delta* each
    call, preventing double-counting when the same history is passed repeatedly.
    Tool iterations are counted from the North result tool events only (not
    from canonical transcript tool_calls), since the Hermes native background
    review path already consumes canonical tool_calls for its own decision.

    When ``review_host`` (a Hermes AIAgent) is provided, the review runs
    directly via ``host._spawn_background_review`` — the same path Hermes
    native uses, with full thread context propagation, auxiliary routing,
    OAuth/live credentials, memory store, approval/tool whitelist, and
    self-improvement summary delivery. When ``tui_host_builder`` is provided
    instead (TUI lazy factory), the host is built on demand only when nudge
    thresholds are met — never per-turn.

    Args:
        canonical_history: Fully canonical Hermes session transcript
            (including the just-completed exchange).
        north_result: result dict from ``NorthCoderRuntime.run_turn``.
        session_id: Hermes session key for nudge state tracking.
        review_host: Hermes AIAgent instance (Gateway path).
        tui_host_builder: Callable returning a Hermes AIAgent (TUI lazy path).
        background_review_callback: Callback for review summary delivery.
        memory_notifications: ``"on"``, ``"off"``, or ``"verbose"``.
    """
    # Only trigger on successfully completed turns
    if not north_result.get("completed"):
        return
    if north_result.get("interrupted"):
        return
    if north_result.get("requires_action"):
        return
    if north_result.get("failed"):
        return

    # Read config for nudge intervals.
    # Hermes canonical keys (not agent.memory_nudge_interval):
    #   memory.nudge_interval (default 10)
    #   skills.creation_nudge_interval (default 10)
    from hermes_cli.config import DEFAULT_CONFIG, load_config
    cfg = load_config()
    mem_cfg = cfg.get("memory", {}) or {}
    raw_mem = mem_cfg.get("nudge_interval")
    if raw_mem is None:
        raw_mem = DEFAULT_CONFIG.get("memory", {}).get("nudge_interval", 10)
    memory_nudge = int(raw_mem)
    skills_cfg = cfg.get("skills", {}) or {}
    raw_skill = skills_cfg.get("creation_nudge_interval")
    if raw_skill is None:
        raw_skill = DEFAULT_CONFIG.get("skills", {}).get("creation_nudge_interval", 10)
    skill_nudge = int(raw_skill)

    # Diff-based counting: compute delta from last-seen canonical counts
    total_users = _count_user_turns(canonical_history)
    with _NORTH_REVIEW_LOCK:
        now = time.time()
        stale_before = now - 86_400
        for stale_id, stale_state in list(_NORTH_REVIEW_STATE.items()):
            if stale_id != session_id and stale_state.get("last_active", 0.0) < stale_before:
                _NORTH_REVIEW_STATE.pop(stale_id, None)
        state = _NORTH_REVIEW_STATE.setdefault(session_id, {
            "accum_user_turns": 0,
            "accum_tool_iters": 0,
            "last_seen_user_count": 0,
            "last_seen_tool_count": 0,
            "last_active": 0.0,
        })
        # Delta = total - last_seen; only the growth since last call counts
        # First baseline uses modulo so a resumed session with many prior
        # user turns does not immediately trigger review.
        old_seen = state.get("last_seen_user_count", 0)
        if old_seen == 0 and total_users > 0:
            # First entry: modulo to avoid immediate trigger on session resume
            if memory_nudge > 0:
                remainder = total_users % memory_nudge
                new_user_turns = memory_nudge if remainder == 0 else remainder
            else:
                new_user_turns = 0
        else:
            new_user_turns = max(0, total_users - old_seen)
        state["last_seen_user_count"] = total_users

        # Tool iterations: only count current North result tool_call_start events.
        # Canonical transcript tool_calls are NOT counted here — the Hermes native
        # review path internally scans the snapshot history for its own decision.
        north_tools = north_result.get("tools", []) or []
        new_tool_iters = _count_north_tool_iterations(north_tools)

        # Accumulate
        state["accum_user_turns"] = state.get("accum_user_turns", 0) + new_user_turns
        state["accum_tool_iters"] = state.get("accum_tool_iters", 0) + new_tool_iters
        state["last_active"] = now

        total_user_turns = state["accum_user_turns"]
        total_tool_iters = state["accum_tool_iters"]

    review_memory = memory_nudge > 0 and total_user_turns >= memory_nudge
    review_skills = skill_nudge > 0 and total_tool_iters >= skill_nudge

    if not review_memory and not review_skills:
        return

    # Determine and validate the review host (Hermes AIAgent).
    host = review_host
    if host is None and tui_host_builder is not None:
        try:
            host = tui_host_builder()
        except Exception:
            host = None
    if host is None:
        logger.warning(
            "North background review skipped: no review host available for session %s",
            session_id,
        )
        return

    # Native-style gating on the real host.
    if review_memory:
        mem_enabled = bool(getattr(host, "_memory_enabled", False) or
                           getattr(host, "_user_profile_enabled", False))
        if not mem_enabled:
            review_memory = False
    if review_skills:
        vt = getattr(host, "valid_tool_names", None)
        skills_enabled = vt is not None and "skill_manage" in vt
        if not skills_enabled:
            review_skills = False

    if not review_memory and not review_skills:
        return

    # Wire callbacks.
    if background_review_callback is not None:
        host.background_review_callback = background_review_callback
    if memory_notifications:
        host.memory_notifications = memory_notifications

    # Spawn the background review daemon thread.
    # The host MUST NOT be closed here — _spawn_background_review only starts
    # the daemon; the caller (TUIAgent facade or Gateway runner) owns close().
    try:
        host._spawn_background_review(
            messages_snapshot=_north_review_snapshot(
                canonical_history,
                north_result.get("tools", []) or [],
            ),
            review_memory=review_memory,
            review_skills=review_skills,
        )
        with _NORTH_REVIEW_LOCK:
            if session_id in _NORTH_REVIEW_STATE:
                if review_memory:
                    _NORTH_REVIEW_STATE[session_id]["accum_user_turns"] = 0
                if review_skills:
                    _NORTH_REVIEW_STATE[session_id]["accum_tool_iters"] = 0
    except Exception:
        logger.warning(
            "North background review failed for session %s", session_id,
            exc_info=True,
        )


# Exported symbols for the North background review integration
__all__ = [
    "_count_north_tool_iterations",
    "_count_user_turns",
    "_north_review_snapshot",
    "schedule_north_background_review",
]
