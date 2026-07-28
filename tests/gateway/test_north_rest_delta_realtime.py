"""RED tests: North 0.4 契约修复 — REST/WS增量流式 & stop-tool project_switch.

Seam A: REST polling 成为增量源并与 WS 共享前缀去重 accumulator。
真实 North 0.4 REST /result 在 running 期间逐步返回 assistant text blocks
（每次增加内容），且 WS 可能静默。当前代码:
  - _poll_result 只在整个 full_response 为空时 append 一次 (line 904)
  - 不调用 on_delta
  - terminal 构造时 tools: [] 丢弃真实 blocks
修复后: _poll_result 每次 poll 只 append 新增 suffix, 调用 on_delta,
REST+WS 前缀去重。

Seam B: stop-tool project_switch 从 REST tool_use blocks 提取。
真实 North 0.4 中 project_switch 是 stop tool — 只有 tool_use block
(content = JSON string with name/input), 无 tool_call_start/result。
当前 extract_workspace_switch 只支持 tool_call_start+result 配对,
永不触发。修复后支持 tool_use type blocks。
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Optional

import pytest
from aiohttp import web

from gateway.north_coder_runtime import (
    NorthCoderRuntime,
    NorthCoderRuntimeConfig,
    NorthCoderTUIAgent,
)


# ======================================================================
# Seam A: REST/WS 增量 delta 流式
# ======================================================================


@pytest.mark.asyncio
async def test_rest_poll_incremental_delta(aiohttp_server, tmp_path):
    """REST poll 逐步返回更长 text blocks → on_delta 只收到新增 suffix.

    RED 条件:
      - 断言 on_delta 被调用多次 (每次只新增部分)
      - 断言 final full_response 为完整文本
      - 断言没有前缀重复
    """
    invoked = False
    poll_count = 0

    async def composite_run(request):
        nonlocal invoked
        invoked = True
        return web.json_response({
            "conversation_id": "conv-incr",
            "workspace_id": "home-default",
            "invocation_id": "inv-incr",
            "status": "running",
        }, status=202)

    # Simulate North 0.4 REST /result growing blocks per poll
    async def invocation_result(request):
        nonlocal poll_count
        poll_count += 1
        if poll_count == 1:
            return web.json_response({
                "status": "running",
                "blocks": [
                    {"role": "assistant", "block_type": "text", "content": "北京"},
                ],
            })
        elif poll_count == 2:
            return web.json_response({
                "status": "running",
                "blocks": [
                    {"role": "assistant", "block_type": "text", "content": "北京市"},
                ],
            })
        elif poll_count == 3:
            return web.json_response({
                "status": "running",
                "blocks": [
                    {"role": "assistant", "block_type": "text", "content": "北京市今天"},
                ],
            })
        else:
            return web.json_response({
                "status": "completed",
                "blocks": [
                    {"role": "assistant", "block_type": "text", "content": "北京市今天天气晴朗"},
                ],
            })

    async def websocket(request):
        # WS 静默 — 永不发送事件
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await asyncio.Event().wait()
        return ws

    app = web.Application()
    app.router.add_post("/api/run", composite_run)
    app.router.add_get("/api/invocations/{inv_id}/result", invocation_result)
    app.router.add_get("/ws/conversation/{conv_id}", websocket)
    server = await aiohttp_server(app)

    deltas: list[str] = []
    runtime = NorthCoderRuntime(
        NorthCoderRuntimeConfig(
            base_url=f"http://{server.host}:{server.port}",
            timeout_seconds=10.0,
            seed_token_budget=2000,
        ),
        tmp_path,
    )

    result = await runtime.run_turn(
        message="天气",
        session_key="test-incr-delta",
        hermes_session_id="h-incr-delta",
        workspace_id="home-default",
        on_delta=lambda d: deltas.append(d),
    )

    # ── RED 断言: on_delta 收到多个仅新增 suffix ──
    assert len(deltas) >= 3, (
        f"on_delta 应被多次调用 (增量 poll), 实际 {len(deltas)} 次: {deltas}"
    )
    # delta 序列应为 ["北京", "市", "今天", "天气晴朗"] 或类似分段
    # 关键是累计拼接后等于完整文本，且无任何前缀重复
    full_delta = "".join(deltas)
    assert full_delta == "北京市今天天气晴朗", (
        f"完整 delta 拼接应为完整文本, got {full_delta!r}"
    )

    # 断言 final_response 也是完整文本
    assert result.get("final_response") == "北京市今天天气晴朗", (
        f"final_response 应为完整文本, got {result.get('final_response')!r}"
    )
    assert result.get("completed") is True
    assert invoked is True
    assert poll_count >= 4, f"应至少 poll 4 次, got {poll_count}"


@pytest.mark.asyncio
async def test_ws_and_rest_same_delta_dedup(aiohttp_server, tmp_path):
    """WS 和 REST 提供相同 delta 内容 → 必须去重，不重复。

    真实场景: North WS 正常发 text_message_content delta，
    REST /result 也返回已包含相同 text 的 blocks。
    修复后: WS delta 在 on_delta/full_response 中优先，
    REST poll 只 append 超出 WS accumulator 的新增部分。
    """
    async def composite_run(request):
        return web.json_response({
            "conversation_id": "conv-dedup",
            "workspace_id": "home-default",
            "invocation_id": "inv-dedup",
            "status": "running",
        }, status=202)

    poll_count = 0

    async def invocation_result(request):
        nonlocal poll_count
        poll_count += 1
        # REST 从第1次poll就返回完整文本
        return web.json_response({
            "status": "completed" if poll_count >= 3 else "running",
            "blocks": [
                {"role": "assistant", "block_type": "text", "content": "Hello North!"},
            ],
        })

    async def websocket(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "run_started", "messageId": "m1"})
        await ws.send_json({"type": "text_message_content", "messageId": "m1", "delta": "Hello "})
        await ws.send_json({"type": "text_message_content", "messageId": "m1", "delta": "North!"})
        # REST 可能在 WS 完成前返回同样内容
        await asyncio.sleep(0.3)
        await ws.send_json({"type": "run_finished", "messageId": "m1"})
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_post("/api/run", composite_run)
    app.router.add_get("/api/invocations/{inv_id}/result", invocation_result)
    app.router.add_get("/ws/conversation/{conv_id}", websocket)
    server = await aiohttp_server(app)

    deltas: list[str] = []
    runtime = NorthCoderRuntime(
        NorthCoderRuntimeConfig(
            base_url=f"http://{server.host}:{server.port}",
            timeout_seconds=10.0,
            seed_token_budget=2000,
        ),
        tmp_path,
    )

    result = await runtime.run_turn(
        message="hello",
        session_key="test-dedup",
        hermes_session_id="h-dedup",
        workspace_id="home-default",
        on_delta=lambda d: deltas.append(d),
    )

    # WS delta 优先且不应被 REST 重复 prefix
    full_delta = "".join(deltas)
    assert full_delta == "Hello North!", (
        f"delta 应完整不重复: {full_delta!r}"
    )
    # final_response 不应有重复 prefix
    fr = result.get("final_response", "")
    assert fr == "Hello North!" or fr.count("Hello") == 1, (
        f"final_response 不应重复: {fr!r}"
    )
    assert result.get("completed") is True


@pytest.mark.asyncio
async def test_tui_facade_separates_process_text_from_final_answer(
    aiohttp_server, monkeypatch, tmp_path,
):
    """North 多阶段 assistant blocks 不得在 TUI 最终正文中 concat。"""

    async def composite_run(request):
        return web.json_response({
            "conversation_id": "conv-process",
            "workspace_id": "home-default",
            "invocation_id": "inv-process",
            "status": "running",
        }, status=202)

    polls = 0

    async def invocation_result(request):
        nonlocal polls
        polls += 1
        blocks = [
            {
                "position": 1,
                "role": "assistant",
                "block_type": "text",
                "content": "先检查仓库和发布入口。",
            },
            {
                "position": 2,
                "role": "assistant",
                "block_type": "tool_use",
                "content": json.dumps({
                    "id": "call-check",
                    "name": "run_shell_command",
                    "input": {"command": "git status --short"},
                }),
            },
            {
                "position": 3,
                "role": "assistant",
                "block_type": "tool_result",
                "content": json.dumps({
                    "toolUseId": "call-check",
                    "content": "clean",
                    "isError": False,
                }),
            },
        ]
        if polls >= 2:
            blocks.append({
                "position": 4,
                "role": "assistant",
                "block_type": "text",
                "content": "发布检查完成。",
            })
        return web.json_response({
            "status": "completed" if polls >= 2 else "running",
            "blocks": blocks,
        })

    async def websocket(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "run_finished"})
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_post("/api/run", composite_run)
    app.router.add_get("/api/invocations/{inv_id}/result", invocation_result)
    app.router.add_get("/ws/conversation/{conv_id}", websocket)
    server = await aiohttp_server(app)

    monkeypatch.setattr("agent.runtime_cwd.resolve_agent_cwd", lambda: tmp_path)
    runtime = NorthCoderRuntime(
        NorthCoderRuntimeConfig(
            base_url=f"http://{server.host}:{server.port}",
            timeout_seconds=10.0,
            seed_token_budget=2000,
            workspace_id="home-default",
            state_file=str(tmp_path / "bindings.json"),
            tui_variant=True,
            workspace_switch_supported=True,
        ),
        tmp_path,
    )
    agent = NorthCoderTUIAgent(runtime, "process-session")
    deltas: list[str] = []
    reasoning: list[str] = []
    tool_starts: list[tuple[str, str]] = []
    tool_completes: list[tuple[str, str]] = []

    result = await asyncio.to_thread(
        agent.run_conversation,
        "执行发布检查",
        stream_callback=deltas.append,
        reasoning_callback=reasoning.append,
        tool_start_callback=lambda call_id, name, _args: tool_starts.append((call_id, name)),
        tool_complete_callback=lambda call_id, name, _args, _result: tool_completes.append((call_id, name)),
        tool_progress_callback=lambda *_args, **_kwargs: None,
    )

    assert "".join(deltas) == "发布检查完成。"
    assert result["final_response"] == "发布检查完成。"
    assert reasoning == ["先检查仓库和发布入口。"]
    assert tool_starts == [("call-check", "run_shell_command")]
    assert tool_completes == [("call-check", "run_shell_command")]


# ======================================================================
# Seam B: stop-tool project_switch — REST tool_use blocks


def test_extract_workspace_switch_from_rest_tool_use():
    """extract_workspace_switch 支持 REST tool_use blocks (stop-tool 格式).

    真实 North 0.4: stop tool project_switch 在 REST blocks 中表现为
    type=tool_use, content='{"id":"...","input":{"project":"..."},"name":"project_switch"}'
    无 tool_call_start/tool_call_result。
    """
    runtime = NorthCoderRuntime(
        NorthCoderRuntimeConfig(base_url="http://0.0.0.0:1", timeout_seconds=0.1),
        Path("/tmp"),
    )

    # 模拟 REST result blocks 中的 tool_use (stop-tool 格式)
    terminal = {
        "status": "completed",
        "result": {
            "blocks": [
                {
                    "block_type": "tool_use",
                    "content": (
                        '{"id":"toolu_abc123","input":{"project":"ai-intelligence-cockpit"},'
                        '"name":"project_switch"}'
                    ),
                },
            ],
        },
        "tools": [],
    }

    ws = runtime.extract_workspace_switch(terminal)
    assert ws is not None, (
        "extract_workspace_switch 应从 REST tool_use blocks 提取 project_switch"
    )
    assert ws.get("name") == "project_switch" or ws.get("tool_name") == "project_switch", (
        f"应提取 project_switch, got {ws}"
    )
    assert "ai-intelligence-cockpit" in str(ws.get("input", {})), (
        f"应包含 input project 信息, got {ws}"
    )


def test_extract_workspace_switch_non_project_ignored():
    """非 project_switch 的 tool_use block 被忽略。"""
    runtime = NorthCoderRuntime(
        NorthCoderRuntimeConfig(base_url="http://0.0.0.0:1", timeout_seconds=0.1),
        Path("/tmp"),
    )
    terminal = {
        "status": "completed",
        "result": {
            "blocks": [
                {
                    "block_type": "tool_use",
                    "content": (
                        '{"id":"toolu_xyz","input":{"text":"hello"},'
                        '"name":"read_file"}'
                    ),
                },
            ],
        },
        "tools": [],
    }
    assert runtime.extract_workspace_switch(terminal) is None, (
        "非 project_switch 应返回 None"
    )


@pytest.mark.asyncio
async def test_tui_facade_real_runtime_project_switch_via_tool_use(
    aiohttp_server, monkeypatch, tmp_path,
):
    """真实 NorthCoderRuntime + NorthCoderTUIAgent + aiohttp 公共seam集成测试。

    Seam B 生产路径验证：
    - 真实 runtime + facade + HTTP/WS 两轮交换
    - 第一轮 /api/run 接收旧cwd → conv-old/inv-switch
      /result completed 含真实 0.4 assistant text + tool_use project_switch
      生产 extract_workspace_switch 触发 callback
    - callback 记录实际 dict 并 monkeypatch resolve_agent_cwd → 新目录
      detach_session 清除 state file 旧 binding
    - 第二轮同一个 agent/session_key 调用 follow-up
      /api/run 必须收到新 cwd + register_workdir=true
      state file 绑定 conv-new/new cwd
    - agent.history 保留两轮，不创建新 Hermes session
    - WS 静默连接，REST polling 真实执行
    """
    # run_conversation is synchronous (it owns an asyncio.run); execute it in a
    # worker thread so this test's event loop can keep serving the aiohttp seam.
    old_cwd = str(tmp_path / "old_workspace")
    new_cwd = str(tmp_path / "new_workspace")
    os.makedirs(old_cwd, exist_ok=True)
    os.makedirs(new_cwd, exist_ok=True)

    _cwd_container: list[str] = [old_cwd]

    def _fake_resolve_agent_cwd() -> Path:
        return Path(_cwd_container[0])

    monkeypatch.setattr(
        "agent.runtime_cwd.resolve_agent_cwd", _fake_resolve_agent_cwd,
    )

    # ── Server state machine ──────────────────────────────────────────
    run_payloads: list[dict[str, Any]] = []
    inv_switch_poll = [0]

    async def composite_run(request):
        payload = await request.json()
        run_payloads.append(payload)
        idx = len(run_payloads)
        if idx == 1:
            return web.json_response({
                "conversation_id": "conv-old",
                "workspace_id": "ws-old",
                "invocation_id": "inv-switch",
                "status": "running",
            }, status=202)
        else:
            return web.json_response({
                "conversation_id": "conv-new",
                "workspace_id": "ws-new",
                "invocation_id": "inv-follow",
                "status": "running",
            }, status=202)

    async def invocation_result(request):
        inv_id = request.match_info["inv_id"]
        if inv_id == "inv-switch":
            inv_switch_poll[0] += 1
            if inv_switch_poll[0] <= 2:
                return web.json_response({
                    "status": "running",
                    "blocks": [
                        {"role": "assistant", "block_type": "text", "content": "Switching"},
                    ],
                })
            return web.json_response({
                "status": "completed",
                "blocks": [
                    {"role": "assistant", "block_type": "text", "content": "Switching to new project..."},
                    {"block_type": "tool_use", "content": json.dumps({
                        "id": "toolu_abc",
                        "input": {"project": "test-project"},
                        "name": "project_switch",
                    })},
                ],
            })
        else:
            return web.json_response({
                "status": "completed",
                "blocks": [
                    {"role": "assistant", "block_type": "text", "content": "Follow-up response"},
                ],
            })

    async def websocket(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        # Reproduce the real race: North's WS terminal event can arrive before
        # the authoritative REST result exposes its final tool-use blocks.
        await ws.send_json({"type": "run_finished"})
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_post("/api/run", composite_run)
    app.router.add_get("/api/invocations/{inv_id}/result", invocation_result)
    app.router.add_get("/ws/conversation/{conv_id}", websocket)
    server = await aiohttp_server(app)

    runtime = NorthCoderRuntime(
        NorthCoderRuntimeConfig(
            base_url=f"http://{server.host}:{server.port}",
            timeout_seconds=15.0,
            workspace_switch_supported=True,
            tui_variant=True,
            agent_yaml_path=None,
        ),
        tmp_path,
    )

    callback_calls: list[dict[str, Any]] = []

    def cb(ws: dict[str, Any]):
        callback_calls.append(dict(ws))
        _cwd_container[0] = new_cwd
        return None

    agent = NorthCoderTUIAgent(
        runtime,
        "sess-real-switch",
        workspace_switch_callback=cb,
    )

    # Round 1 exercises the public synchronous facade. The callback must be
    # invoked by production parsing.  A successful application keeps the old
    # provider binding until the next run's workdir mismatch triggers rebind.
    result1 = await asyncio.to_thread(
        agent.run_conversation,
        "switch to project test-project",
    )
    assert len(callback_calls) == 1
    assert callback_calls[0]["project_id"] == "test-project"
    assert agent.session_key == "sess-real-switch"
    conv_id1, _, _ = await runtime._conversation_binding("sess-real-switch")
    assert conv_id1 == "conv-old", "successful switch must retain old binding until next run"

    # Round 2 reuses the exact same facade/session. resolve_agent_cwd now points
    # at the callback-selected workspace, so North must create a fresh provider
    # conversation there rather than closing or replacing the Hermes session.
    result2 = await asyncio.to_thread(
        agent.run_conversation,
        "follow-up in the new workspace",
    )

    assert len(run_payloads) == 2
    assert run_payloads[0].get("workdir") == old_cwd
    assert run_payloads[0].get("register_workdir") is True
    assert run_payloads[1].get("workdir") == new_cwd
    assert run_payloads[1].get("register_workdir") is True

    conv_id2, workdir2, _ = await runtime._conversation_binding("sess-real-switch")
    assert conv_id2 == "conv-new"
    assert workdir2 == new_cwd
    assert result1.get("north_conversation_id") == "conv-old"
    assert result2.get("north_conversation_id") == "conv-new"
    assert result1.get("north_invocation_id") == "inv-switch"
    assert result2.get("north_invocation_id") == "inv-follow"

    # Canonical Hermes history survives the provider-conversation boundary.
    assert agent.session_key == "sess-real-switch"
    assert [m.get("content") for m in agent.history] == [
        "switch to project test-project",
        "Switching to new project...",
        (
            "已切换工作区\n\n"
            "项目：test-project\n"
            "路径：\n\n"
            "后续消息将在该路径中执行。"
        ),
        "follow-up in the new workspace",
        "Follow-up response",
    ]
    assert result2["messages"] == agent.history


@pytest.mark.asyncio
async def test_gateway_direct_unsupported_project_switch_tool_use(
    aiohttp_server, tmp_path,
):
    """Gateway direct path (workspace_switch_supported=False) 收到
    真实 tool_use project_switch blocks 时 fail closed。

    断言：
    - _run_turn_impl 返回 failed/unsupported
    - callback 不执行（未配置）
    - detach 不发生
    - state file 维持原 binding
    """
    async def composite_run(request):
        return web.json_response({
            "conversation_id": "conv-unsup",
            "workspace_id": "ws-unsup",
            "invocation_id": "inv-unsup",
            "status": "running",
        }, status=202)

    async def message_send(request):
        return web.json_response({
            "invocation_id": "inv-unsup",
            "status": "running",
        }, status=202)

    poll_count: list[int] = [0]

    async def invocation_result(request):
        poll_count[0] += 1
        if poll_count[0] <= 2:
            return web.json_response({
                "status": "running",
                "blocks": [
                    {"role": "assistant", "block_type": "text", "content": "Working"},
                ],
            })
        return web.json_response({
            "status": "completed",
            "blocks": [
                {"role": "assistant", "block_type": "text", "content": "Done"},
                {"block_type": "tool_use", "content": json.dumps({
                    "id": "toolu_xyz",
                    "input": {"project": "some-project"},
                    "name": "project_switch",
                })},
            ],
        })

    async def websocket(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        while True:
            try:
                msg = await ws.receive()
                if msg.type in (web.WSMsgType.CLOSED, web.WSMsgType.ERROR, web.WSMsgType.CLOSE):
                    break
            except Exception:
                break
        return ws

    app = web.Application()
    app.router.add_post("/api/run", composite_run)
    app.router.add_post("/api/conversations/{conv_id}/messages", message_send)
    app.router.add_get("/api/invocations/{inv_id}/result", invocation_result)
    app.router.add_get("/ws/conversation/{conv_id}", websocket)
    server = await aiohttp_server(app)

    runtime = NorthCoderRuntime(
        NorthCoderRuntimeConfig(
            base_url=f"http://{server.host}:{server.port}",
            timeout_seconds=15.0,
            workspace_switch_supported=False,
            tui_variant=False,
        ),
        tmp_path,
    )

    # 记录 binding 作为 baseline（使用与 run_turn 相同的 workspace_id）
    await runtime._record_conversation_binding(
        "sess-unsup", "conv-unsup-old", None, "ws-unsup",
    )

    detach_called: list[str] = []
    original_detach = runtime.detach_session
    def _detach_watch(sk: str):
        detach_called.append(sk)
        return original_detach(sk)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(runtime, "detach_session", _detach_watch)

    try:
        result = await runtime.run_turn(
            message="switch project please",
            session_key="sess-unsup",
            hermes_session_id="h-unsup",
            workspace_id="ws-unsup",
        )

        # 返回 fail closed
        assert result.get("failed") is True, (
            "workspace_switch_supported=False 应返回 failed"
        )
        assert result.get("status") in ("unsupported", "failed"), (
            f"status 应为 unsupported/failed, got {result.get('status')!r}"
        )
        assert result.get("completed") is False, "应为未完成"

        # detach 不应该被调用
        assert len(detach_called) == 0, (
            "Gateway direct path 不应 detach"
        )

        # state file 维持原 binding
        conv, wd, ws_id = await runtime._conversation_binding("sess-unsup")
        assert conv == "conv-unsup-old", (
            f"state file 不应被修改, got conv={conv!r}"
        )
    finally:
        monkeypatch.undo()


# ======================================================================
# Seam C: REST child lineage fallback — North 0.4 result block shape


def _north_04_child_lineage_blocks(*, second_parent: str | None = None) -> list[dict[str, Any]]:
    """Build the persisted 0.4 shape, including the root Agent control block."""
    blocks: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "block_type": "tool_use",
            "content": json.dumps({
                "id": "call-a",
                "input": {"message": "inspect", "sub_agent_name": "explore"},
                "name": "Agent",
            }),
        },
        {
            "role": "assistant",
            "block_type": "tool_use",
            "parent_agent_id": "explore",
            "parent_tool_call_id": "call-a",
            "content": json.dumps({"id": "child-tool", "name": "read_file", "input": {"path": "README"}}),
        },
        {
            "role": "tool",
            "block_type": "tool_result",
            "parent_agent_id": "explore",
            "parent_tool_call_id": "call-a",
            "content": json.dumps({"toolUseId": "child-tool", "content": "ok"}),
        },
        {
            "role": "assistant",
            "block_type": "text",
            "parent_agent_id": "explore",
            "parent_tool_call_id": "call-a",
            "content": "inspection complete",
            "metadata": json.dumps({"subagentStatus": "done"}),
        },
        {
            "role": "assistant",
            "block_type": "subagent_anchor",
            "parent_agent_id": "explore",
            "parent_tool_call_id": "call-a",
        },
    ]
    if second_parent is not None:
        blocks.extend([
            {
                "role": "assistant",
                "block_type": "tool_use",
                "parent_agent_id": "explore",
                "parent_tool_call_id": second_parent,
                "content": json.dumps({"id": f"child-{second_parent}", "name": "read_file", "input": {}}),
            },
            {
                "role": "assistant",
                "block_type": "text",
                "parent_agent_id": "explore",
                "parent_tool_call_id": second_parent,
                "content": f"summary-{second_parent}",
                "metadata": json.dumps({"subagentStatus": "done"}),
            },
        ])
    return blocks


def _runtime_for_rest_projection() -> NorthCoderRuntime:
    return NorthCoderRuntime(
        NorthCoderRuntimeConfig(base_url="http://0.0.0.0:1", timeout_seconds=0.1),
        Path("/tmp"),
    )


@pytest.mark.asyncio
async def test_rest_child_lineage_projects_exact_north_04_lifecycle_and_dedupes():
    runtime = _runtime_for_rest_projection()
    result = {"status": "completed", "blocks": _north_04_child_lineage_blocks()}
    callbacks: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    emitted = await runtime._emit_rest_subagent_events(result, callbacks.append, seen)
    assert emitted == [
        {"type": "subagent_start", "agentId": "rest-child:call-a", "agentName": "explore",
         "parentToolCallId": "call-a", "query": "inspect"},
        {"type": "subagent_progress", "agentId": "rest-child:call-a", "agentName": "explore",
         "parentToolCallId": "call-a", "lastToolName": "read_file", "completedToolCalls": 1, "activeToolCalls": 0},
        {"type": "subagent_end", "agentId": "rest-child:call-a", "agentName": "explore",
         "parentToolCallId": "call-a", "status": "completed", "result": "inspection complete"},
    ]
    assert callbacks == emitted
    assert await runtime._emit_rest_subagent_events(result, callbacks.append, seen) == emitted
    assert callbacks == emitted, "重复 REST result 不得重复 callback"


@pytest.mark.asyncio
async def test_rest_child_lineage_keeps_parallel_same_role_children_independent():
    runtime = _runtime_for_rest_projection()
    result = {"status": "completed", "blocks": _north_04_child_lineage_blocks(second_parent="call-b")}
    callbacks: list[dict[str, Any]] = []
    await runtime._emit_rest_subagent_events(result, callbacks.append, set())

    lifecycle = [event for event in callbacks if event["type"] in {"subagent_start", "subagent_end"}]
    assert [(event["type"], event["agentId"], event["parentToolCallId"]) for event in lifecycle] == [
        ("subagent_start", "rest-child:call-a", "call-a"),
        ("subagent_end", "rest-child:call-a", "call-a"),
        ("subagent_start", "rest-child:call-b", "call-b"),
        ("subagent_end", "rest-child:call-b", "call-b"),
    ]


@pytest.mark.asyncio
async def test_rest_cancelled_root_closes_observed_child_lineage_once():
    runtime = _runtime_for_rest_projection()
    blocks = _north_04_child_lineage_blocks()[:-1]
    blocks[-1].pop("metadata")
    blocks[-1]["content"] = "last observed progress"
    result = {"status": "cancelled", "blocks": blocks}
    callbacks: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    emitted = await runtime._emit_rest_subagent_events(result, callbacks.append, seen)

    assert [event["type"] for event in emitted] == [
        "subagent_start", "subagent_progress", "subagent_end",
    ]
    assert emitted[-1] == {
        "type": "subagent_end", "agentId": "rest-child:call-a", "agentName": "explore",
        "parentToolCallId": "call-a", "status": "cancelled", "result": "last observed progress",
    }
    assert emitted[1]["lastToolName"] == "read_file"
    assert emitted[1]["completedToolCalls"] == 1
    await runtime._emit_rest_subagent_events(result, callbacks.append, seen)
    assert callbacks == emitted, "cancelled child end must be emitted exactly once"


@pytest.mark.asyncio
async def test_rest_cancelled_root_keeps_completed_child_completed():
    runtime = _runtime_for_rest_projection()
    result = {"status": "cancelled", "blocks": _north_04_child_lineage_blocks()}

    emitted = await runtime._emit_rest_subagent_events(result, None, set())

    child_end = next(event for event in emitted if event["type"] == "subagent_end")
    assert child_end["status"] == "completed"


def test_rest_child_blocks_are_hidden_from_root_process_and_result_extractors():
    runtime = _runtime_for_rest_projection()
    blocks = [
        {"position": 1, "role": "assistant", "block_type": "text", "content": "root process"},
        *_north_04_child_lineage_blocks(),
        {"position": 10, "role": "assistant", "block_type": "tool_use",
         "content": json.dumps({"id": "root-tool", "name": "read_file", "input": {}})},
        {"position": 11, "role": "assistant", "block_type": "text", "content": "root final"},
    ]
    result = {"blocks": blocks}
    assert [json.loads(block["content"])["id"] for block in runtime._extract_result_tool_blocks(result)] == [
        "call-a", "root-tool",
    ]
    assert runtime._extract_result_text(result) == "root processroot final"
    assert runtime._extract_final_result_text(result) == "root final"


@pytest.mark.asyncio
async def test_rest_process_projection_keeps_root_agent_but_emits_no_child_tool_lifecycle():
    runtime = _runtime_for_rest_projection()
    callbacks: list[dict[str, Any]] = []
    await runtime._emit_rest_process_events(
        {"blocks": _north_04_child_lineage_blocks()}, callbacks.append, set()
    )
    assert [event["type"] for event in callbacks] == ["tool_call_start"]
    assert callbacks[0]["toolCallName"] == "Agent"


@pytest.mark.asyncio
async def test_rest_child_lifecycle_shares_boundary_dedupe_with_live_events():
    runtime = _runtime_for_rest_projection()
    callbacks: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    await runtime._emit_rest_subagent_events(
        {"blocks": _north_04_child_lineage_blocks()}, callbacks.append, seen
    )
    live_start = {"type": "subagent_start", "agentId": "live-child", "parentToolCallId": "call-a"}
    live_end = {"type": "subagent_end", "agentId": "live-child", "parentToolCallId": "call-a"}
    assert runtime._lifecycle_key(live_start) == runtime._lifecycle_key(callbacks[0])
    assert runtime._lifecycle_key(live_end) == runtime._lifecycle_key(callbacks[-1])
    callback_count = len(callbacks)
    for event in (live_start, live_end):
        if runtime._lifecycle_key(event) not in seen:
            callbacks.append(event)
    assert len(callbacks) == callback_count, "shared lifecycle seen must suppress live boundary duplicates"


@pytest.mark.asyncio
async def test_rest_child_lineage_without_child_blocks_emits_nothing():
    runtime = _runtime_for_rest_projection()
    callbacks: list[dict[str, Any]] = []
    assert await runtime._emit_rest_subagent_events(
        {"blocks": [{"role": "assistant", "block_type": "text", "content": "root only"}]},
        callbacks.append, set(),
    ) == []
    assert callbacks == []
