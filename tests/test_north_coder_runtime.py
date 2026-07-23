import json
from pathlib import Path
from typing import Any, cast

import pytest
from aiohttp import web

from gateway.north_coder_runtime import NorthCoderRuntime, NorthCoderRuntimeConfig, NorthCoderTUIAgent
from hermes_cli.north_coder_profile import export_hermes_profile


def test_north_runtime_syncs_save_memory_to_local_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    events = [
        {"type": "tool_call_start", "toolCallId": "mem-1", "toolCallName": "save_memory"},
        {"type": "tool_call_args", "toolCallId": "mem-1", "delta": '{"action":"add","target":"memory","content":"North local memory probe"}'},
        {"type": "tool_call_result", "toolCallId": "mem-1", "content": "saved"},
    ]
    NorthCoderRuntime._sync_local_memory(events)
    assert "North local memory probe" in (tmp_path / "hermes" / "memories" / "MEMORY.md").read_text()


def test_north_tui_agent_forwards_tool_lifecycle_callbacks():
    class FakeRuntime:
        async def run_turn(self, **kwargs):
            kwargs["on_event"]({"type": "tool_call_start", "toolCallId": "t-1", "toolCallName": "read_file"})
            kwargs["on_event"]({"type": "tool_call_result", "toolCallId": "t-1", "content": "ok"})
            return {"messages": [], "north_invocation_id": "inv-1"}

    events = []
    agent = NorthCoderTUIAgent(cast(Any, FakeRuntime()), "session-1")
    agent.run_conversation(
        "probe",
        tool_start_callback=lambda *args: events.append(("start", args)),
        tool_complete_callback=lambda *args: events.append(("complete", args)),
    )
    assert [kind for kind, _ in events] == ["start", "complete"]
    assert events[0][1][0:2] == ("t-1", "read_file")
    assert events[1][1][0:2] == ("t-1", "read_file")


@pytest.mark.asyncio
async def test_north_runtime_translates_conversation_message_and_events(tmp_path, aiohttp_server):
    conversation_id = "conv-test"

    async def create_conversation(request):
        payload = await request.json()
        assert payload["agent_config"]["agent_profile_id"] == "builtin:general"
        assert payload["agent_config"]["agent_yaml_path"] == "/tmp/hermes-agent.yaml"
        assert payload["agent_config"]["model_id"] == "ng-test-model"
        assert payload["metadata"]["source"] == "hermes-gateway"
        return web.json_response({"id": conversation_id, "workspace_id": "home"})

    async def send_message(request):
        payload = await request.json()
        assert payload["content"] == "hello"
        assert payload["agent_profile_id"] == "builtin:general"
        assert payload["model_id"] == "ng-test-model"
        assert payload["metadata"]["hermes_context_prompt"] == "channel context"
        assert payload["metadata"]["ask_user_response"]["tool_call_id"] == "ask-1"
        await request.app["ws"].send_json({"type": "history_ref"})
        await request.app["ws"].send_json({"type": "replay_start"})
        await request.app["ws"].send_json({"type": "text_message_content", "messageId": "old-message", "delta": "stale answer"})
        await request.app["ws"].send_json({"type": "run_finished", "messageId": "old-message"})
        await request.app["ws"].send_json({"type": "replay_end"})
        await request.app["ws"].send_json({"type": "run_started", "messageId": "m1"})
        await request.app["ws"].send_json({"type": "tool_call_start", "toolCallId": "tool-1", "toolCallName": "read_file"})
        await request.app["ws"].send_json({"type": "tool_call_result", "toolCallId": "tool-1", "content": "ok"})
        await request.app["ws"].send_json({"type": "text_message_content", "messageId": "m1", "delta": "hello back"})
        await request.app["ws"].send_json({"type": "run_finished", "messageId": "m1"})
        return web.json_response({"invocation_id": "inv-test", "status": "running"})

    async def websocket(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        request.app["ws"] = ws
        await ws.receive()
        return ws

    app = web.Application()
    app["ws"] = None
    app.router.add_post("/api/workspaces/home-default/conversations", create_conversation)
    app.router.add_post(f"/api/conversations/{conversation_id}/messages", send_message)
    app.router.add_get(f"/ws/conversation/{conversation_id}", websocket)
    server = await aiohttp_server(app)

    runtime = NorthCoderRuntime(
        NorthCoderRuntimeConfig(
            base_url=f"http://{server.host}:{server.port}",
            agent_profile_id="builtin:general",
            agent_yaml_path="/tmp/hermes-agent.yaml",
            model_id="ng-test-model",
        ),
        tmp_path,
    )
    result = await runtime.run_turn(
        message="hello",
        session_key="slack:C:thread",
        hermes_session_id="hermes-session",
        context_prompt="channel context",
        metadata_extra={"ask_user_response": {"tool_call_id": "ask-1"}},
    )

    assert result["final_response"] == "hello back"
    assert len(result["tools"]) == 2
    assert result["status"] == "completed"
    assert result["north_conversation_id"] == conversation_id
    assert json.loads((tmp_path / "north_coder_conversations.json").read_text()) == {
        "slack:C:thread": conversation_id
    }


def test_profile_export_omits_credentials_and_preserves_profile_context(tmp_path):
    home = tmp_path / "profile"
    (home / "memories").mkdir(parents=True)
    (home / "skills" / "demo").mkdir(parents=True)
    (home / "config.yaml").write_text("agent:\n  max_turns: 42\n", encoding="utf-8")
    (home / "memories" / "USER.md").write_text("用户偏好中文", encoding="utf-8")
    (home / "skills" / "demo" / "SKILL.md").write_text("# Demo", encoding="utf-8")

    agent_yaml = export_hermes_profile(home, tmp_path / "north-profile", name="hermes-test")
    yaml_text = agent_yaml.read_text(encoding="utf-8")
    prompt_text = agent_yaml.with_name("system_prompt.md").read_text(encoding="utf-8")

    assert "name: hermes-test" in yaml_text
    assert "max_iterations: 42" in yaml_text
    assert "api_key" in yaml_text  # placeholder is required by North schema
    assert "用户偏好中文" in prompt_text
    assert str(home / "skills" / "demo") in yaml_text
    assert "sk-" not in yaml_text
    for tool_name in ("apply_patch", "save_memory"):
        assert f"name: {tool_name}" in yaml_text
    assert "name: ask_user" not in yaml_text
    assert "name: complete_task" not in yaml_text
    assert "name: skill_manage" in yaml_text
    assert "binding: ./custom_tools/skill_manage_bridge.py:skill_manage" in yaml_text
    assert "name: backend" not in yaml_text
    assert "name: terminal" not in yaml_text
    assert "name: run_shell_command" in yaml_text
    assert "name: background_task_manage" in yaml_text
    assert "sandbox_config:" in yaml_text
    assert "work_dir: ${env.NORTH_CODER_WORKSPACE_ROOT}" in yaml_text
    assert (agent_yaml.parent / "tools" / "skill_manage.tool.yaml").is_file()
    bridge = (agent_yaml.parent / "custom_tools" / "skill_manage_bridge.py").read_text()
    assert str(home) in bridge


@pytest.mark.asyncio
async def test_north_runtime_resolves_permission_and_answers_ask_user(tmp_path, aiohttp_server):
    seen: list[tuple[str, dict]] = []

    async def invocation(request):
        return web.json_response({
            "status": "requires_action",
            "conversation_id": "conv-action",
            "required_action": {"action_id": "ask-1"},
        })

    async def permission(request):
        seen.append(("permission", await request.json()))
        return web.json_response({"invocation_id": "inv-resumed", "status": "running"})

    async def answer(request):
        seen.append(("answer", await request.json()))
        return web.json_response({"invocation_id": "inv-answer", "status": "running"})

    app = web.Application()
    app.router.add_get("/api/invocations/inv-action", invocation)
    app.router.add_post("/api/invocations/inv-permission/permissions/tool-1/resolve", permission)
    app.router.add_post("/api/conversations/conv-action/messages", answer)
    server = await aiohttp_server(app)
    runtime = NorthCoderRuntime(
        NorthCoderRuntimeConfig(base_url=f"http://{server.host}:{server.port}"),
        tmp_path,
    )

    assert await runtime.resolve_permission("inv-permission", "tool-1", "allow_once") == {
        "invocation_id": "inv-resumed", "status": "running"
    }
    assert await runtime.answer_ask_user("inv-action", [{"header": "Mode", "value": "safe"}]) == {
        "invocation_id": "inv-answer", "status": "running"
    }
    assert seen[0] == ("permission", {"decision": "allow_once"})
    answer_payload = seen[1][1]
    assert answer_payload["metadata"]["ask_user_response"]["tool_call_id"] == "ask-1"
    assert answer_payload["metadata"]["ask_user_response"]["answers"][0]["type"] == "text"
