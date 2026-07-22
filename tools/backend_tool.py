"""Service-gated CRUD tool for local + CNB execution backends."""

from __future__ import annotations

import json
from typing import Any

from tools.execution_backends import (
    BackendError,
    backend_tool as _backend_tool,
    is_execution_backends_enabled,
)
from tools.registry import registry


BACKEND_SCHEMA = {
    "name": "backend",
    "description": (
        "Manage the execution backend for environment tools. The Hermes agent, "
        "conversation, memory, skills, and LLM requests remain local. Use one "
        "CRUD action: create/get/update/delete. update with current=true switches "
        "this session; get without id lists all backends and the current selection."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "get", "update", "delete"],
            },
            "id": {
                "type": "string",
                "description": "Stable backend id. Optional only for get-all.",
            },
            "repo": {
                "type": "string",
                "description": "Exact CNB repository slug for create.",
            },
            "branch": {
                "type": "string",
                "description": "Exact CNB repository branch for create.",
            },
            "current": {
                "type": "boolean",
                "description": "Set true with update to select this backend.",
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


def _dispatch_backend(args: dict[str, Any], **kwargs: Any) -> str:
    try:
        return _backend_tool(
            args,
            task_id=kwargs.get("task_id"),
            session_id=kwargs.get("session_id"),
        )
    except BackendError as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


registry.register(
    name="backend",
    toolset="terminal",
    schema=BACKEND_SCHEMA,
    handler=_dispatch_backend,
    check_fn=is_execution_backends_enabled,
    description="Manage local and CNB execution backends",
    emoji="🧭",
)
