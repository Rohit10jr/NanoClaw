import pytest

from nanoclaw.agent.tools import ToolRegistry, is_tool_error
from nanoclaw.agent.tools.base import Tool
from nanoclaw.agent.tools.schema import cast_params, object_schema, validate_json_schema_value, validate_params

SCHEMA = object_schema(
    path={"type": "string", "minLength": 1, "maxLength": 10},
    count={"type": "integer", "minimum": 1, "maximum": 5},
    ratio={"type": "number"},
    flag={"type": "boolean"},
    mode={"type": "string", "enum": ["a", "b"]},
    tags={"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 2},
    opt={"type": ["integer", "null"]},
    nested={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
    required=["path"],
)


def test_object_schema_shape():
    assert object_schema(a={"type": "string"}, required=["a"]) == {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": False,
    }


def test_valid_params_have_no_errors():
    params = {"path": "a.txt", "count": 3, "ratio": 0.5, "flag": True, "mode": "a", "tags": ["x"], "opt": None,
              "nested": {"x": 1}}
    assert validate_params(params, SCHEMA) == []


@pytest.mark.parametrize(
    ("params", "error"),
    [
        ({}, "missing required path"),
        ({"path": 5}, "path should be string"),
        ({"path": ""}, "path must be at least 1 chars"),
        ({"path": "x" * 11}, "path must be at most 10 chars"),
        ({"path": "a", "count": 0}, "count must be >= 1"),
        ({"path": "a", "count": 6}, "count must be <= 5"),
        ({"path": "a", "count": 1.5}, "count should be integer"),
        ({"path": "a", "count": True}, "count should be integer"),
        ({"path": "a", "ratio": "x"}, "ratio should be number"),
        ({"path": "a", "ratio": float("inf")}, "ratio must be finite"),
        ({"path": "a", "flag": "yes"}, "flag should be boolean"),
        ({"path": "a", "mode": "c"}, "mode must be one of ['a', 'b']"),
        ({"path": "a", "tags": []}, "tags must have at least 1 items"),
        ({"path": "a", "tags": ["1", "2", "3"]}, "tags must be at most 2 items"),
        ({"path": "a", "tags": [1]}, "tags[0] should be string"),
        ({"path": "a", "nested": {}}, "missing required nested.x"),
        ({"path": "a", "pth": "typo"}, "unexpected parameter pth"),
    ],
)
def test_validation_messages_match_nanobot(params, error):
    assert error in validate_params(params, SCHEMA)


def test_non_object_params():
    assert validate_params(["a"], SCHEMA) == ["parameters must be an object, got list"]


def test_boolean_schemas():
    assert validate_json_schema_value(1, True) == []
    assert validate_json_schema_value(1, False, "x") == ["x is not allowed by schema"]


@pytest.mark.parametrize(
    ("raw", "cast"),
    [
        ({"count": "3"}, {"count": 3}),
        ({"ratio": "0.5"}, {"ratio": 0.5}),
        ({"flag": "true"}, {"flag": True}),
        ({"flag": "No"}, {"flag": False}),
        ({"path": 42}, {"path": "42"}),
        ({"count": "three"}, {"count": "three"}),  # left for validation to reject
        ({"flag": "maybe"}, {"flag": "maybe"}),
        ({"tags": [1, 2]}, {"tags": ["1", "2"]}),
        ({"nested": {"x": "7"}}, {"nested": {"x": 7}}),
        ({"unknown": "5"}, {"unknown": "5"}),
    ],
)
def test_cast_params(raw, cast):
    assert cast_params(raw, SCHEMA) == cast


class _ArgsTool(Tool):
    """A tool that really has a parameter called 'arguments'."""

    name = "args_tool"  # type: ignore[assignment]
    description = "x"  # type: ignore[assignment]
    parameters = object_schema(arguments={"type": "string"}, required=["arguments"])  # type: ignore[assignment]

    async def execute(self, arguments: str) -> str:
        return f"got {arguments}"


class _CountTool(Tool):
    name = "count_tool"  # type: ignore[assignment]
    description = "x"  # type: ignore[assignment]
    parameters = object_schema(n={"type": "integer", "minimum": 1}, required=["n"])  # type: ignore[assignment]

    async def execute(self, n: int) -> str:
        return f"n={n!r}"


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(_ArgsTool())
    reg.register(_CountTool())
    return reg


async def test_registry_casts_then_validates(registry):
    assert await registry.execute("count_tool", {"n": "4"}) == "n=4"
    result = await registry.execute("count_tool", {"n": "0"})
    assert is_tool_error(result)
    assert result.startswith("Error: Invalid parameters for tool 'count_tool': n must be >= 1")


async def test_registry_reports_all_errors_joined(registry):
    result = await registry.execute("count_tool", {"x": 1})
    assert "missing required n; unexpected parameter x" in result


async def test_registry_unwraps_arguments_payload(registry):
    assert await registry.execute("count_tool", {"arguments": {"n": 2}}) == "n=2"
    assert await registry.execute("count_tool", {"arguments": '{"n": 3}'}) == "n=3"


async def test_registry_keeps_real_arguments_parameter(registry):
    assert await registry.execute("args_tool", {"arguments": "hello"}) == "got hello"


async def test_registry_parses_json_string_params(registry):
    assert await registry.execute("count_tool", '{"n": 5}') == "n=5"
