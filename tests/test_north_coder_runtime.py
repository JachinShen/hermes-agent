import json
from pathlib import Path

import pytest
from aiohttp import web

from gateway.north_coder_runtime import NorthCoderRuntime, NorthCoderRuntimeConfig
from hermes_cli.north_coder_profile import export_hermes_profile


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
        await request.app["ws"].send_json({"type": "history_ref"})
        await request.app["ws"].send_json({"type": "replay_start"})
        await request.app["ws"].send_json({"type": "text_message_content", "messageId": "old-message", "delta": "stale answer"})
        await request.app["ws"].send_json({"type": "run_finished", "messageId": "old-message"})
        await request.app["ws"].send_json({"type": "replay_end"})
        await request.app["ws"].send_json({"type": "run_started", "messageId": "m1"})
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
    )

    assert result["final_response"] == "hello back"
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
    for tool_name in ("apply_patch", "save_memory", "complete_task", "ToolSearch"):
        assert f"name: {tool_name}" in yaml_text
