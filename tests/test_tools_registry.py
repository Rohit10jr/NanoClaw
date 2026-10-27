import pytest

from nanoclaw.agent.tools import RETRY_HINT, ToolRegistry, ToolResult, is_tool_error
from nanoclaw.agent.tools.registry import with_retry_hint
from toy_tools import AddTool, BoomTool, EchoTool, FailTool


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    for tool in (EchoTool(), AddTool(), BoomTool(), FailTool()):
        reg.register(tool)
    return reg


def test_tool_result_is_a_string_with_error_flag():
    ok, bad = ToolResult("fine"), ToolResult.error("nope")
    assert ok == "fine" and isinstance(ok, str)
    assert not is_tool_error(ok) and is_tool_error(bad)
    assert not is_tool_error("plain string")


def test_to_schema_is_an_openai_function_tool():
    assert EchoTool().to_schema() == {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Return the given text.",
            "parameters": EchoTool().parameters,
        },
    }


def test_definitions_are_sorted_by_name(registry):
    names = [d["function"]["name"] for d in registry.get_definitions()]
    assert names == ["add", "boom", "echo", "fail"]


def test_register_unregister_and_lookup(registry):
    assert "echo" in registry and registry.has("echo") and len(registry) == 4
    registry.unregister("echo")
    assert "echo" not in registry and registry.get("echo") is None
    registry.unregister("echo")  # unregistering twice is harmless


async def test_execute_success(registry):
    assert await registry.execute("echo", {"text": "hi"}) == "hi"
    assert await registry.execute("add", {"a": 2, "b": 3}) == "5"


async def test_unknown_tool(registry):
    result = await registry.execute("nope", {})
    assert is_tool_error(result)
    assert result.startswith("Error: Tool 'nope' not found. Available: echo, add, boom, fail")
    assert result.endswith(RETRY_HINT)


@pytest.mark.parametrize("name", ["Echo", "ECHO", "e-cho"])
async def test_unknown_tool_suggests_near_match_but_does_not_run_it(registry, name):
    result = await registry.execute(name, {"text": "hi"})
    assert is_tool_error(result)
    assert "Did you mean 'echo'?" in result


async def test_empty_tool_name_is_rejected(registry):
    result = await registry.execute("", {})
    assert is_tool_error(result) and "not found" in result


@pytest.mark.parametrize("args", ['{"text": "unterminated', ["a"], 42])
async def test_arguments_must_be_an_object(registry, args):
    result = await registry.execute("echo", args)
    assert is_tool_error(result)
    assert result.startswith(
        f"Error: Tool 'echo' parameters must be a JSON object, got {type(args).__name__}. "
        'Use named parameters like tool_name(param1="value1", param2="value2") matching the tool schema.'
    )


async def test_missing_required_parameter(registry):
    result = await registry.execute("add", {"a": 1})
    assert is_tool_error(result)
    assert "Invalid parameters for tool 'add': missing required b" in result


async def test_raising_tool_becomes_error(registry):
    result = await registry.execute("boom", {})
    assert is_tool_error(result)
    assert result == "Error executing boom: kaboom" + RETRY_HINT


async def test_error_result_gets_hint(registry):
    result = await registry.execute("fail", {})
    assert result == "Error: file not found" + RETRY_HINT


def test_retry_hint_is_added_once():
    once = with_retry_hint("Error: x")
    assert with_retry_hint(once) == once
    assert once.count(RETRY_HINT) == 1


def test_retry_hint_text_matches_nanobot():
    # nanobot 841d2b8 registry.py:14 _ERROR_HINT: neutral guidance, not "try a different approach".
    assert RETRY_HINT == (
        "\n\n[Consider this result in the context of the user's goal "
        "and decide whether and how to proceed.]"
    )


@pytest.mark.parametrize("already_hinted", [False, True])
async def test_error_result_is_hinted_exactly_once(already_hinted):
    # nanobot tests/tools/test_tool_registry.py test_registry_uses_structured_tool_result_for_errors
    error_hint = (
        "\n\n[Consider this result in the context of the user's goal "
        "and decide whether and how to proceed.]"
    )

    class _HintedFail(FailTool):
        async def execute(self, **kwargs):
            return ToolResult.error("Error: real failure" + (error_hint if already_hinted else ""))

    reg = ToolRegistry()
    reg.register(_HintedFail())
    result = await reg.execute("fail", {})

    assert is_tool_error(result)
    assert result == "Error: real failure" + error_hint
