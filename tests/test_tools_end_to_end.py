"""The runner driving the real default tools, with a scripted model."""

from nanoclaw.agent.runner import AgentRunner, AgentRunSpec
from nanoclaw.agent.tools import build_default_tools
from nanoclaw.providers import FakeProvider, LLMResponse, ToolCallRequest


def tool_turn(name, args, call_id):
    return LLMResponse(content=None, tool_calls=[ToolCallRequest(call_id, name, args)], finish_reason="tool_calls")


async def test_model_reads_edits_and_verifies_a_file(tmp_path):
    (tmp_path / "config.ini").write_text("[server]\nport = 8080\nhost = localhost\n")
    fake = FakeProvider(
        [
            tool_turn("list_dir", {"path": "."}, "c1"),
            tool_turn("read_file", {"path": "config.ini"}, "c2"),
            tool_turn("edit_file", {"path": "config.ini", "old_text": "port = 8080", "new_text": "port = 9090"}, "c3"),
            tool_turn("exec", {"command": "grep port config.ini"}, "c4"),
            "Changed the port to 9090.",
        ]
    )
    result = await AgentRunner().run(
        AgentRunSpec(
            initial_messages=[{"role": "user", "content": "change the port to 9090"}],
            tools=build_default_tools(tmp_path),
            provider=fake,
        )
    )

    assert result.stop_reason == "completed"
    assert result.tools_used == ["list_dir", "read_file", "edit_file", "exec"]
    assert (tmp_path / "config.ini").read_text() == "[server]\nport = 9090\nhost = localhost\n"
    tool_outputs = {m["tool_call_id"]: m["content"] for m in result.messages if m["role"] == "tool"}
    assert tool_outputs["c1"] == "📄 config.ini"
    assert tool_outputs["c2"].startswith("1| [server]\n2| port = 8080")
    assert tool_outputs["c3"] == "Patch applied:\n- update config.ini (+1/-1)"
    assert tool_outputs["c4"] == "port = 9090\n\n\nExit code: 0"


async def test_bad_arguments_reach_the_model_which_then_corrects_itself(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n")
    fake = FakeProvider(
        [
            tool_turn("read_file", {"file": "a.txt", "limit": "ten"}, "c1"),
            tool_turn("read_file", {"path": "a.txt", "limit": "10"}, "c2"),
            "It says hello.",
        ]
    )
    result = await AgentRunner().run(
        AgentRunSpec(
            initial_messages=[{"role": "user", "content": "what's in a.txt?"}],
            tools=build_default_tools(tmp_path),
            provider=fake,
        )
    )
    outputs = [m["content"] for m in result.messages if m["role"] == "tool"]
    assert outputs[0].startswith(
        "Error: Invalid parameters for tool 'read_file': "
        "missing required path; unexpected parameter file; limit should be integer"
    )
    assert outputs[1].startswith("1| hello")  # "10" was cast to 10
    assert result.tools_used == ["read_file"]


async def test_blocked_command_is_reported_to_the_model(tmp_path):
    fake = FakeProvider([tool_turn("exec", {"command": "rm -rf ."}, "c1"), "I won't do that."])
    result = await AgentRunner().run(
        AgentRunSpec(
            initial_messages=[{"role": "user", "content": "delete everything"}],
            tools=build_default_tools(tmp_path),
            provider=fake,
        )
    )
    assert result.messages[2]["content"].startswith("Error: Command blocked by deny pattern filter")
    assert result.tools_used == []
