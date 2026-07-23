"""Export the active Hermes profile as a North Coder agent artifact.

The export is deliberately local and deterministic: credentials, sessions,
logs, caches, and provider auth are never copied. North receives the resulting
agent.yaml path in the conversation message request.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


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
    venv_python = hermes_home / "hermes-agent" / "venv" / "bin" / "python"
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


def _backend_tool_yaml() -> str:
    return """type: tool
name: backend
description: >-
  Manage Hermes local/CNB execution backends. Use get, create, update with
  current=true, or delete. This controls where terminal commands execute.
input_schema:
  type: object
  properties:
    action: {type: string, enum: [create, get, update, delete]}
    id: {type: string}
    repo: {type: string}
    branch: {type: string}
    current: {type: boolean}
  required: [action]
  additionalProperties: false
"""


def _terminal_tool_yaml() -> str:
    return """type: tool
name: terminal
description: >-
  Execute a command through Hermes' selected execution backend. Use this for
  environment commands when a CNB backend is selected; do not substitute the
  North-local run_shell_command tool.
input_schema:
  type: object
  properties:
    command: {type: string}
    background: {type: boolean}
    timeout: {type: integer}
    workdir: {type: string}
    pty: {type: boolean}
    notify_on_complete: {type: boolean}
    watch_patterns: {type: array, items: {type: string}}
  required: [command]
  additionalProperties: false
"""


def _execution_bridge_py(hermes_home: Path) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    venv_python = hermes_home / "hermes-agent" / "venv" / "bin" / "python"
    helper = r'''import json, os, subprocess


def _call(module, function, payload):
    env = os.environ.copy()
    env["HERMES_HOME"] = __HERMES_HOME__
    env["PYTHONPATH"] = __REPO_ROOT__ + os.pathsep + env.get("PYTHONPATH", "")
    code = f"""
import json, sys
from {module} import {function}
args = json.load(sys.stdin)
print({function}(**args))
"""
    proc = subprocess.run([__VENV_PYTHON__, "-c", code], input=json.dumps(payload),
                          text=True, capture_output=True, env=env)
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or "Hermes execution helper failed")
    return proc.stdout.strip()

def backend(action, id=None, repo=None, branch=None, current=None,
            session_id=None, task_id=None, **_ignored):
    payload = {"action": action, "id": id, "repo": repo, "branch": branch, "current": current}
    call = {"args": {k: v for k, v in payload.items() if v is not None}}
    if session_id is not None:
        call["session_id"] = session_id
    if task_id is not None:
        call["task_id"] = task_id
    return _call("tools.backend_tool", "_dispatch_backend", call)


def terminal(command, background=False, timeout=None, workdir=None, pty=False,
            notify_on_complete=False, watch_patterns=None, session_id=None,
            task_id=None, **_ignored):
    payload = {"command": command, "background": background, "timeout": timeout,
               "workdir": workdir, "pty": pty, "notify_on_complete": notify_on_complete,
               "watch_patterns": watch_patterns}
    call = {k: v for k, v in payload.items() if v is not None}
    if session_id is not None:
        call["session_id"] = session_id
    if task_id is not None:
        call["task_id"] = task_id
    return _call("tools.terminal_tool", "terminal_tool", call)
'''
    return (helper.replace("__HERMES_HOME__", repr(str(hermes_home)))
                  .replace("__REPO_ROOT__", repr(str(repo_root)))
                  .replace("__VENV_PYTHON__", repr(str(venv_python))))


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
    ("web_search", "web.search"),
    ("web_read", "web.read"),
    ("write_todos", "session.write_todos"),
    ("save_memory", "session.save_memory"),
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
        "Environment commands must use the Hermes terminal tool so the selected local/CNB execution backend is honored; North's run_shell_command is local to the North workspace.",
    ]
    for title, filename in (("SOUL", "SOUL.md"), ("Memory", "memories/MEMORY.md"), ("User profile", "memories/USER.md")):
        path = hermes_home / filename
        if path.is_file():
            prompt_parts.extend(["", f"## {title}", "", path.read_text(encoding="utf-8")])
    (output_dir / "system_prompt.md").write_text("\n".join(prompt_parts).rstrip() + "\n", encoding="utf-8")
    custom_tools_dir = output_dir / "custom_tools"
    tool_specs_dir = output_dir / "tools"
    custom_tools_dir.mkdir(parents=True, exist_ok=True)
    tool_specs_dir.mkdir(parents=True, exist_ok=True)
    (tool_specs_dir / "skill_manage.tool.yaml").write_text(_skill_manage_tool_yaml(), encoding="utf-8")
    (tool_specs_dir / "backend.tool.yaml").write_text(_backend_tool_yaml(), encoding="utf-8")
    (tool_specs_dir / "terminal.tool.yaml").write_text(_terminal_tool_yaml(), encoding="utf-8")
    (custom_tools_dir / "skill_manage_bridge.py").write_text(
        _skill_manage_bridge_py(hermes_home), encoding="utf-8"
    )
    (custom_tools_dir / "execution_bridge.py").write_text(
        _execution_bridge_py(hermes_home), encoding="utf-8"
    )

    skill_paths = sorted(
        str(path.parent)
        for path in (hermes_home / "skills").rglob("SKILL.md")
        if path.is_file()
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
        "tools:",
    ]
    for tool_name, builtin in _CORE_NORTH_TOOLS:
        lines.extend([f"  - name: {tool_name}", f"    builtin: {builtin}"])
    lines.extend([
        "  - name: skill_manage",
        "    yaml_path: ./tools/skill_manage.tool.yaml",
        "    binding: ./custom_tools/skill_manage_bridge.py:skill_manage",
        "  - name: backend",
        "    yaml_path: ./tools/backend.tool.yaml",
        "    binding: ./custom_tools/execution_bridge.py:backend",
        "  - name: terminal",
        "    yaml_path: ./tools/terminal.tool.yaml",
        "    binding: ./custom_tools/execution_bridge.py:terminal",
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
