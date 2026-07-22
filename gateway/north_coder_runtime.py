"""North Coder Agent Runtime adapter.

This module keeps Hermes channel/session delivery intact while delegating the
agent loop to a running North Coder control plane. North owns the durable
conversation and invocation history; Hermes only keeps the deterministic
session-key -> conversation-id bridge required for channel routing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)

EventCallback = Callable[[dict[str, Any]], Awaitable[None] | None]
DeltaCallback = Callable[[str], None]


@dataclass(frozen=True)
class NorthCoderRuntimeConfig:
    base_url: str = "http://127.0.0.1:8848"
    workspace_id: str = "home-default"
    agent_profile_id: str = "hermes:default"
    agent_yaml_path: Optional[str] = None
    model_id: Optional[str] = "ng-gpt-5.6-sol"
    timeout_seconds: float = 1800.0
    state_file: Optional[str] = None

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
        )


class NorthCoderRuntime:
    """Translate one Hermes Gateway turn to North REST + WebSocket APIs."""

    def __init__(self, config: NorthCoderRuntimeConfig, hermes_home: Path):
        self.config = config
        self.hermes_home = hermes_home
        self.state_file = Path(config.state_file) if config.state_file else hermes_home / "north_coder_conversations.json"
        self._state_lock = asyncio.Lock()
        # Process-local ownership for cross-thread Gateway/TUI interrupts.
        self._active_invocations: dict[str, str] = {}
        self._active_lock = asyncio.Lock()

    async def run_turn(
        self,
        *,
        message: str,
        session_key: str,
        hermes_session_id: str,
        context_prompt: str = "",
        source: Any = None,
        event_message_id: Optional[str] = None,
        on_delta: Optional[DeltaCallback] = None,
        on_event: Optional[EventCallback] = None,
    ) -> dict[str, Any]:
        import aiohttp

        conversation_id = await self._conversation_id(session_key)
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds, sock_read=self.config.timeout_seconds)
        headers = {"Content-Type": "application/json", "X-Hermes-Session-Id": hermes_session_id}
        client_message_id = event_message_id or f"hermes-{uuid.uuid4().hex}"
        payload = {
            "content": message,
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
        if self.config.model_id:
            payload["model_id"] = self.config.model_id
        if self.config.agent_yaml_path:
            payload["agent_yaml_path"] = self.config.agent_yaml_path
        if getattr(source, "workdir", None):
            payload["workdir"] = source.workdir

        full_response: list[str] = []
        invocation_id: Optional[str] = None
        terminal: dict[str, Any] = {"status": "completed", "tools": []}

        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as client:
            ws = None
            try:
                ws = await self._connect_events(client, conversation_id)
            except Exception as exc:
                logger.warning("North event stream unavailable; falling back to invocation polling: %s", exc)

            async with client.post(
                f"{self.config.base_url}/api/conversations/{conversation_id}/messages",
                json=payload,
            ) as response:
                body = await response.text()
                if response.status >= 400:
                    raise RuntimeError(f"North message request failed ({response.status}): {body[:500]}")
                accepted = json.loads(body) if body else {}
                invocation_id = accepted.get("invocation_id")
                if invocation_id:
                    async with self._active_lock:
                        self._active_invocations[session_key] = str(invocation_id)

            if ws is not None:
                try:
                    terminal = await self._consume_events(ws, full_response, on_delta, on_event)
                finally:
                    await ws.close()
            elif invocation_id:
                terminal = await self._poll_result(client, invocation_id, full_response, on_event)

        async with self._active_lock:
            if self._active_invocations.get(session_key) == str(invocation_id or ""):
                self._active_invocations.pop(session_key, None)

        text = "".join(full_response)
        status = str(terminal.get("status") or "completed")
        # NexAU can emit a synthetic "No response content or tool calls"
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

    async def cancel(self, invocation_id: str) -> None:
        import aiohttp

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as client:
            async with client.post(f"{self.config.base_url}/api/invocations/{invocation_id}/cancel") as response:
                if response.status >= 400:
                    raise RuntimeError(f"North cancel failed ({response.status}): {(await response.text())[:300]}")

    def active_invocation(self, session_key: str) -> Optional[str]:
        return self._active_invocations.get(session_key)

    def cancel_session(self, session_key: str) -> None:
        """Best-effort synchronous cancellation hook for Gateway/TUI threads."""
        invocation_id = self.active_invocation(session_key)
        if not invocation_id:
            return
        try:
            asyncio.run(self.cancel(invocation_id))
        except Exception:
            logger.exception("Failed to cancel North invocation %s", invocation_id)

    async def _connect_events(self, client: Any, conversation_id: str) -> Any:
        ws_url = self.config.base_url.replace("http://", "ws://", 1).replace("https://", "wss://", 1)
        return await client.ws_connect(f"{ws_url}/ws/conversation/{conversation_id}", heartbeat=30)

    async def _consume_events(
        self,
        ws: Any,
        full_response: list[str],
        on_delta: Optional[DeltaCallback],
        on_event: Optional[EventCallback],
    ) -> dict[str, Any]:
        replaying = False
        terminal: dict[str, Any] = {"status": "completed", "tools": []}
        while True:
            item = await ws.receive()
            if item.type.name in {"CLOSED", "CLOSE", "CLOSING"}:
                break
            if item.type.name == "ERROR":
                raise RuntimeError(f"North event stream error: {ws.exception()}")
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
            if replaying or kind in {"history_ref", "replay_start", "replay_end"}:
                continue
            if kind == "text_message_content":
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

    async def _poll_result(self, client: Any, invocation_id: str, full_response: list[str], on_event: Optional[EventCallback]) -> dict[str, Any]:
        deadline = time.monotonic() + self.config.timeout_seconds
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
            text = result.get("text") or result.get("content") or result.get("response")
            if text and not full_response:
                full_response.append(str(text))
            status = str(result.get("status") or "")
            if status in {"completed", "failed", "cancelled", "error"} or result.get("finished"):
                if status not in {"completed", ""}:
                    return result
                return result
            await asyncio.sleep(0.25)
        raise TimeoutError(f"North invocation {invocation_id} timed out")

    async def _conversation_id(self, session_key: str) -> str:
        async with self._state_lock:
            state = self._read_state()
            existing = state.get(session_key)
            if existing:
                return str(existing)
            conversation_id = await self._create_conversation(session_key)
            state[session_key] = conversation_id
            self._write_state(state)
            return conversation_id

    async def _create_conversation(self, session_key: str) -> str:
        import aiohttp

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

    def _read_state(self) -> dict[str, str]:
        try:
            if self.state_file.exists():
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
        except Exception:
            logger.warning("Ignoring unreadable North conversation map %s", self.state_file)
        return {}

    def _write_state(self, state: dict[str, str]) -> None:
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
    """

    def __init__(self, runtime: NorthCoderRuntime, session_key: str):
        self.runtime = runtime
        self.session_key = session_key
        self.session_id = session_key
        self.model = "north-coder"
        self.provider = "north_coder"
        self.runtime_override = "ncoder"
        self.history: list[dict[str, Any]] = []
        self._last_invocation_id: Optional[str] = None
        self._interrupted = False

    def run_conversation(self, message: Any, *, conversation_history=None, stream_callback=None, task_id=None, **kwargs):
        text = message if isinstance(message, str) else str(message)
        if conversation_history:
            self.history = list(conversation_history)
        result = asyncio.run(
            self.runtime.run_turn(
                message=text,
                session_key=self.session_key,
                hermes_session_id=self.session_id,
                on_delta=stream_callback,
            )
        )
        self._last_invocation_id = result.get("north_invocation_id")
        self.history.extend(result.get("messages", []))
        return result

    def interrupt(self) -> None:
        self._interrupted = True
        self.runtime.cancel_session(self.session_key)

    def clear_interrupt(self) -> None:
        self._interrupted = False

    def close(self) -> None:
        """AIAgent compatibility hook; North owns transport cleanup per turn."""
        return None


def tui_agent_from_raw(raw: dict[str, Any], hermes_home: Path, session_key: str) -> Optional[NorthCoderTUIAgent]:
    runtime = runtime_from_raw(raw, hermes_home)
    return NorthCoderTUIAgent(runtime, session_key) if runtime else None
