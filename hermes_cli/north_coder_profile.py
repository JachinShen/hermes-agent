"""Export the active Hermes profile as a North Coder agent artifact.

The export is deliberately local and deterministic: credentials, sessions,
logs, caches, and provider auth are never copied. North receives the resulting
agent.yaml path in the conversation message request.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


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
    ("ask_user", "session.ask_user"),
    ("complete_task", "session.complete_task"),
    ("save_memory", "session.save_memory"),
    ("ToolSearch", "system.tool_search"),
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
    ]
    for title, filename in (("SOUL", "SOUL.md"), ("Memory", "memories/MEMORY.md"), ("User profile", "memories/USER.md")):
        path = hermes_home / filename
        if path.is_file():
            prompt_parts.extend(["", f"## {title}", "", path.read_text(encoding="utf-8")])
    (output_dir / "system_prompt.md").write_text("\n".join(prompt_parts).rstrip() + "\n", encoding="utf-8")

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
