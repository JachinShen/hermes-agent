"""Export the active Hermes profile as a North Coder agent artifact.

The export is deliberately local and deterministic: credentials, sessions,
logs, caches, and provider auth are never copied. North receives the resulting
agent.yaml path in the conversation message request.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from agent.skill_utils import is_excluded_skill_path
from hermes_constants import get_hermes_home


def _skill_manage_tool_yaml() -> str:
    return """type: tool
name: skill_manage
description: >-
  Manage Hermes procedural-memory skills. Use create, patch, edit, delete,
  write_file, or remove_file. New skills are stored in the active Hermes
  profile's skills directory.
input_schema:
  type: object
  properties:
    action:
      type: string
      enum: [create, patch, edit, delete, write_file, remove_file]
    name:
      type: string
    content:
      type: string
    category:
      type: string
    file_path:
      type: string
    file_content:
      type: string
    old_string:
      type: string
    new_string:
      type: string
    replace_all:
      type: boolean
    absorbed_into:
      type: string
  required: [action, name]
  additionalProperties: false
"""


def _skill_manage_bridge_py(hermes_home: Path) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    venv_python = Path(sys.executable)
    helper = r'''import json, os, subprocess


def skill_manage(action, name, content=None, category=None, file_path=None,
                 file_content=None, old_string=None, new_string=None,
                 replace_all=False, absorbed_into=None, **_ignored):
    payload = {
        "action": action, "name": name, "content": content,
        "category": category, "file_path": file_path,
        "file_content": file_content, "old_string": old_string,
        "new_string": new_string, "replace_all": replace_all,
        "absorbed_into": absorbed_into,
    }
    payload = {key: value for key, value in payload.items() if value is not None}
    env = os.environ.copy()
    env["HERMES_HOME"] = __HERMES_HOME__
    env["PYTHONPATH"] = __REPO_ROOT__ + os.pathsep + env.get("PYTHONPATH", "")
    code = """
import json, sys
from tools.skill_manager_tool import skill_manage
args = json.load(sys.stdin)
print(skill_manage(**args))
"""
    proc = subprocess.run(
        [__VENV_PYTHON__, "-c", code], input=json.dumps(payload), text=True,
        capture_output=True, env=env,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or "Hermes skill_manage helper failed")
    return proc.stdout.strip()
'''
    return (
        helper.replace("__HERMES_HOME__", repr(str(hermes_home)))
        .replace("__REPO_ROOT__", repr(str(repo_root)))
        .replace("__VENV_PYTHON__", repr(str(venv_python)))
    )


def _memory_tool_yaml() -> str:
    """Render North's Python-tool shape from Hermes' canonical memory schema."""
    import yaml
    from tools.memory_tool import MEMORY_SCHEMA

    spec = {
        "type": "tool",
        "name": MEMORY_SCHEMA["name"],
        "description": MEMORY_SCHEMA["description"],
        "input_schema": MEMORY_SCHEMA["parameters"],
    }
    return yaml.safe_dump(spec, sort_keys=False, allow_unicode=True)


def _memory_bridge_py(hermes_home: Path) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    venv_python = Path(sys.executable)
    helper = r'''import json, os, subprocess


def memory(action=None, target="memory", content=None, old_text=None,
           operations=None, **_ignored):
    payload = {
        "action": action, "target": target, "content": content,
        "old_text": old_text, "operations": operations,
    }
    payload = {key: value for key, value in payload.items() if value is not None}
    env = os.environ.copy()
    env["HERMES_HOME"] = __HERMES_HOME__
    env["PYTHONPATH"] = __REPO_ROOT__ + os.pathsep + env.get("PYTHONPATH", "")
    code = """
import json, sys
from tools.memory_tool import load_on_disk_store, memory_tool
args = json.load(sys.stdin)
print(memory_tool(store=load_on_disk_store(), **args))
"""
    proc = subprocess.run(
        [__VENV_PYTHON__, "-c", code], input=json.dumps(payload), text=True,
        capture_output=True, env=env,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or "Hermes memory helper failed")
    return proc.stdout.strip()
'''
    return (
        helper.replace("__HERMES_HOME__", repr(str(hermes_home)))
        .replace("__REPO_ROOT__", repr(str(repo_root)))
        .replace("__VENV_PYTHON__", repr(str(venv_python)))
    )


def _memory_prompt_blocks(hermes_home: Path, config: dict[str, Any]) -> list[str]:
    """Build the same sanitized, frozen memory blocks as a native session."""
    from tools.memory_tool import MemoryStore

    raw = config.get("memory")
    memory_cfg = raw if isinstance(raw, dict) else {}
    store = MemoryStore(
        memory_char_limit=int(memory_cfg.get("memory_char_limit", 2200)),
        user_char_limit=int(memory_cfg.get("user_char_limit", 1375)),
    )
    store.load_from_disk(hermes_home / "memories")
    blocks: list[str] = []
    if memory_cfg.get("memory_enabled", True):
        block = store.format_for_system_prompt("memory")
        if block:
            blocks.append(block)
    if memory_cfg.get("user_profile_enabled", True):
        block = store.format_for_system_prompt("user")
        if block:
            blocks.append(block)
    return blocks


_CORE_NORTH_TOOLS = (
    ("read_file", "file.read_file"),
    ("write_file", "file.write_file"),
    ("replace", "file.replace"),
    ("apply_patch", "file.apply_patch"),
    ("multiedit", "file.multiedit"),
    ("glob", "file.glob"),
    ("list_directory", "file.list_directory"),
    ("read_many_files", "file.read_many_files"),
    ("search_file_content", "file.search_content"),
    ("run_shell_command", "shell.run_command"),
    ("background_task_manage", "shell.background_task_manage"),
    ("web_search", "web.search"),
    ("web_read", "web.read"),
    ("write_todos", "session.write_todos"),
)


def export_hermes_profile(
    hermes_home: Path,
    output_dir: Path,
    *,
    name: str = "hermes-profile",
    config: dict[str, Any] | None = None,
) -> Path:
    """Write a North-compatible artifact and return its ``agent.yaml`` path."""
    config = config or _read_config(hermes_home)
    agent_cfg_raw = config.get("agent")
    agent_cfg: dict[str, Any] = agent_cfg_raw if isinstance(agent_cfg_raw, dict) else {}
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_parts: list[str] = [
        "# Hermes profile compatibility layer",
        "",
        "This agent is running under North Coder but preserves the user's Hermes profile context.",
        "The injected Hermes Agent Profile owns the North/NexAU sandbox-aware tools; Hermes Gateway does not manage execution backends for this runtime.",
    ]
    soul_path = hermes_home / "SOUL.md"
    if soul_path.is_file():
        prompt_parts.extend(["", "## SOUL", "", soul_path.read_text(encoding="utf-8")])
    for block in _memory_prompt_blocks(hermes_home, config):
        prompt_parts.extend(["", block])
    (output_dir / "system_prompt.md").write_text("\n".join(prompt_parts).rstrip() + "\n", encoding="utf-8")
    custom_tools_dir = output_dir / "custom_tools"
    tool_specs_dir = output_dir / "tools"
    custom_tools_dir.mkdir(parents=True, exist_ok=True)
    tool_specs_dir.mkdir(parents=True, exist_ok=True)
    (tool_specs_dir / "skill_manage.tool.yaml").write_text(_skill_manage_tool_yaml(), encoding="utf-8")
    (tool_specs_dir / "memory.tool.yaml").write_text(_memory_tool_yaml(), encoding="utf-8")
    (custom_tools_dir / "skill_manage_bridge.py").write_text(
        _skill_manage_bridge_py(hermes_home), encoding="utf-8"
    )
    (custom_tools_dir / "memory_bridge.py").write_text(
        _memory_bridge_py(hermes_home), encoding="utf-8"
    )

    skill_paths = sorted(
        str(path.parent)
        for path in (hermes_home / "skills").rglob("SKILL.md")
        if path.is_file() and not is_excluded_skill_path(path)
    ) if (hermes_home / "skills").is_dir() else []

    max_context = int(agent_cfg.get("max_context_tokens") or 200_000)
    max_iterations = int(agent_cfg.get("max_turns") or 150)
    lines = [
        "type: agent",
        f"name: {name}",
        f"max_context_tokens: {max_context}",
        "system_prompt: ./system_prompt.md",
        "system_prompt_type: file",
        f"max_iterations: {max_iterations}",
        "",
        "# Provider credentials and model are supplied by North runtime settings.",
        "llm_config:",
        "  model: placeholder",
        "  api_key: placeholder",
        "  base_url: https://api.openai.com/v1",
        "",
        "# North/NexAU uses one local sandbox shared by filesystem and shell tools.",
        "# Hermes Gateway does not mount remote execution backends into this Profile.",
        "sandbox_config:",
        "  type: local",
        "",
        "tools:",
    ]
    for tool_name, builtin in _CORE_NORTH_TOOLS:
        lines.extend([f"  - name: {tool_name}", f"    builtin: {builtin}"])
    lines.extend([
        "  - name: memory",
        "    yaml_path: ./tools/memory.tool.yaml",
        "    binding: ./custom_tools/memory_bridge.py:memory",
        "  - name: skill_manage",
        "    yaml_path: ./tools/skill_manage.tool.yaml",
        "    binding: ./custom_tools/skill_manage_bridge.py:skill_manage",
    ])
    if skill_paths:
        lines.extend(["", "skills:"])
        lines.extend(f"  - {path}" for path in skill_paths)
    (output_dir / "agent.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_dir / "agent.yaml"


def _read_config(hermes_home: Path) -> dict[str, Any]:
    path = hermes_home / "config.yaml"
    if not path.is_file():
        return {}
    try:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except ImportError as exc:
        raise RuntimeError("Profile export requires Hermes' PyYAML dependency") from exc


def profile_export_manifest(hermes_home: Path, artifact_path: Path) -> dict[str, str]:
    """Return a redaction-safe manifest suitable for logs and tests."""
    return {
        "hermes_home": str(hermes_home),
        "agent_yaml_path": str(artifact_path),
        "profile_name": artifact_path.parent.name,
    }
