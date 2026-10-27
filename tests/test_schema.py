import json

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


# --- nanobot c4dac64 schema fixes ---------------------------------------------------------------


def _tool(schema):
    """A throwaway tool exposing ``schema`` (nanobot tests' ``CastTestTool``)."""

    class _SchemaTool(Tool):
        name = "sample"  # type: ignore[assignment]
        description = "x"  # type: ignore[assignment]
        parameters = schema  # type: ignore[assignment]

        def __init__(self):
            self.calls = []

        async def execute(self, **kwargs):
            self.calls.append(kwargs)
            return "ok"

    return _SchemaTool()


# 9fe061a: JSON Schema counts 3.0 as an integer; Python decodes it as a float.
@pytest.mark.parametrize("schema_type", ["integer", ["integer", "null"], ["integer", "string"]])
def test_integer_schema_accepts_integral_json_numbers(schema_type):
    assert validate_json_schema_value(3.0, {"type": schema_type}) == []


async def test_registry_casts_integral_json_number_to_integer(registry):
    assert await registry.execute("count_tool", {"n": 3.0}) == "n=3"


@pytest.mark.parametrize("value", [1.5, True, float("inf"), float("nan")])
def test_integer_schema_still_rejects_non_integral_numbers(value):
    assert validate_json_schema_value(value, {"type": "integer"}, "n") == ["n should be integer"]


# 23bf1b8: enum uses JSON equality, so True is not 1 (Python says it is).
@pytest.mark.parametrize(
    ("value", "allowed", "valid"),
    [
        (True, 1, False),
        (False, 0, False),
        (1, True, False),
        (0, False, False),
        ([True], [1], False),
        ({"enabled": True}, {"enabled": 1}, False),
        (1.0, 1, True),
        (True, True, True),
        ("x", "x", True),
        ({"a": 1, "b": 2}, {"b": 2.0, "a": 1.0}, True),
    ],
)
async def test_registry_enforces_json_enum_equality(value, allowed, valid):
    tool = _tool({"type": "object", "properties": {"value": {"enum": [allowed]}}, "required": ["value"]})
    reg = ToolRegistry()
    reg.register(tool)

    result = await reg.execute("sample", {"value": value})

    if valid:
        assert result == "ok"
        assert tool.calls == [{"value": value}]
    else:
        assert "value must be one of" in result
        assert tool.calls == []


# 9ea3268 / e3fc2a7: type "null" rejects other values; a nullable None is still checked against enum.
@pytest.mark.parametrize("raw_type", ["null", ["null"]])
@pytest.mark.parametrize("value", ["", 0, False, [], {}])
def test_null_only_schema_rejects_non_null_values(raw_type, value):
    schema = {"type": raw_type}
    assert validate_json_schema_value(value, schema, "value") == ["value should be null"]
    assert validate_json_schema_value(None, schema) == []


@pytest.mark.parametrize(
    "schema",
    [
        {"type": ["string", "null"], "enum": ["fast"]},
        {"type": "string", "nullable": True, "enum": ["fast"]},
        {"type": ["integer", "null"], "enum": [1]},
    ],
)
def test_nullable_schema_still_enforces_enum(schema):
    assert validate_json_schema_value(None, schema, "mode") == [f"mode must be one of {schema['enum']}"]
    assert validate_json_schema_value(None, {**schema, "enum": [None]}) == []


# 432421b: a multi-type union picks the member the value already matches; nothing is coerced.
@pytest.mark.parametrize("types", [["array", "object"], ["object", "array"]])
@pytest.mark.parametrize(("value", "expected"), [(["2"], [2]), ({"count": "2"}, {"count": 2})])
def test_cast_type_union_recurses_into_matching_container(types, value, expected):
    schema = {
        "type": "object",
        "properties": {"value": {
            "type": types,
            "properties": {"count": {"type": "integer", "minimum": 1}},
            "items": {"type": "integer", "minimum": 1},
        }},
    }

    result = cast_params({"value": value}, schema)

    assert result == {"value": expected}
    assert validate_params(result, schema) == []
    invalid = {"count": 0} if isinstance(value, dict) else [0]
    assert "must be >= 1" in "; ".join(validate_params({"value": invalid}, schema))


@pytest.mark.parametrize("types", [["integer", "string"], ["string", "integer"]])
@pytest.mark.parametrize(("value", "error"), [(0, "must be >= 1"), ("ab", "must be at least 3 chars")])
def test_type_union_enforces_matching_type_constraints(types, value, error):
    schema = {"type": "object", "properties": {"value": {"type": types, "minimum": 1, "minLength": 3}}}

    result = cast_params({"value": value}, schema)

    assert result["value"] is value
    assert error in "; ".join(validate_params(result, schema))


@pytest.mark.parametrize("types", [["integer", "string"], ["string", "integer"]])
@pytest.mark.parametrize("value", [True, 1.5, None])
def test_type_union_rejects_undeclared_types_without_coercion(types, value):
    schema = {"type": "object", "properties": {"value": {"type": types}}}

    result = cast_params({"value": value}, schema)

    assert result["value"] is value
    assert validate_params(result, schema) == [f"value should match one of the types {types}"]


@pytest.mark.parametrize(
    ("types", "value", "error"),
    [
        (["number", "boolean", "null"], True, None),
        (["number", "boolean", "null"], 1.5, None),
        (["number", "boolean", "null"], None, None),
        (["string", "number"], float("inf"), "value must be finite"),
        (["string", "number"], float("nan"), "value must be finite"),
    ],
)
def test_type_union_preserves_numeric_boolean_and_null_values(types, value, error):
    for order in (types, types[::-1]):
        schema = {"type": "object", "properties": {"value": {"type": order}}}

        result = cast_params({"value": value}, schema)

        assert result["value"] is value
        assert validate_params(result, schema) == ([error] if error else [])


@pytest.mark.parametrize(("value", "expected"), [("42", 42), (None, None)])
def test_nullable_integer_retains_safe_casts(value, expected):
    schema = {"type": "object", "properties": {"value": {"type": ["integer", "null"]}}}

    result = cast_params({"value": value}, schema)

    assert result == {"value": expected}
    assert validate_params(result, schema) == []


@pytest.mark.parametrize("types", [["integer", "string"], ["string", "integer"]])
@pytest.mark.parametrize("value", ["00123", "doc-A", 42])
async def test_registry_preserves_type_union_arguments(types, value):
    # nanobot tests/tools/test_mcp_tool.py test_registry_preserves_mcp_type_union_arguments
    tool = _tool({"type": "object", "properties": {"id": {"type": types}}, "required": ["id"]})
    reg = ToolRegistry()
    reg.register(tool)

    assert await reg.execute("sample", {"id": value}) == "ok"
    assert tool.calls == [{"id": value}]
    assert type(tool.calls[0]["id"]) is type(value)


# 6b93043: a JSON string is decoded only where the schema can hold nothing but a container.
_READ_NOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "target": {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {"type": {"type": "string"}, "path": {"type": "string"}},
                    "required": ["type", "path"],
                },
                {
                    "type": "object",
                    "properties": {"type": {"type": "string"}, "name": {"type": "string"}},
                    "required": ["type", "name"],
                },
            ],
        },
        "format": {"type": "string"},
    },
    "required": ["target", "format"],
}


def test_prepare_call_decodes_json_string_for_nested_object_schema():
    reg = ToolRegistry()
    reg.register(_tool(_READ_NOTE_SCHEMA))

    tool, params, error = reg.prepare_call(
        "sample", {"target": '{"type":"path","path":"Projects/nanobot.md"}', "format": "content"}
    )

    assert tool is not None
    assert error is None
    assert params == {"target": {"type": "path", "path": "Projects/nanobot.md"}, "format": "content"}


def test_prepare_call_preserves_json_like_text_for_string_schema():
    reg = ToolRegistry()
    reg.register(_tool({"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}))

    _, params, error = reg.prepare_call("sample", {"value": '{"type":"path"}'})

    assert error is None
    assert params == {"value": '{"type":"path"}'}


@pytest.mark.parametrize("union", ["oneOf", "anyOf"])
@pytest.mark.parametrize(
    "branches",
    [
        [{"type": "object"}, {"type": "string"}],
        [{"type": "string"}, {"type": "object"}],
        [{"type": "object"}, {}],
        [{"type": "object"}, True],
    ],
)
def test_nested_json_cast_preserves_unions_that_can_accept_strings(union, branches):
    reg = ToolRegistry()
    reg.register(_tool({"type": "object", "properties": {"value": {union: branches}}}))
    original = {"value": '{"mode":"text"}'}

    _, params, error = reg.prepare_call("sample", original)

    assert error is None
    assert params == original


@pytest.mark.parametrize("encoded", [True, False])
def test_nested_json_cast_does_not_select_the_first_object_branch(encoded):
    reg = ToolRegistry()
    reg.register(_tool({"type": "object", "properties": {"value": {"oneOf": [
        {"type": "object", "properties": {"count": {"type": "string"}}},
        {"type": "object", "properties": {"count": {"type": "integer"}}},
    ]}}}))
    value = {"count": 12}

    _, params, error = reg.prepare_call("sample", {"value": json.dumps(value) if encoded else value})

    assert error is None
    assert params == {"value": value}


@pytest.mark.parametrize(
    ("schema", "value", "expected"),
    [
        ({"type": "object"}, '{"count":12}', {"count": 12}),
        ({"type": "array"}, '[{"count":12}]', [{"count": 12}]),
        ({"type": ["object", "null"]}, '{"count":12}', {"count": 12}),
        ({"type": ["object", "array"]}, "[1,2]", [1, 2]),
        ({"anyOf": [{"type": "object"}, {"type": "array"}]}, "[1,2]", [1, 2]),
        ({"type": ["string", "object"]}, '{"count":12}', '{"count":12}'),
        ({"type": ["object", "string"]}, '{"count":12}', '{"count":12}'),
        ({"type": "object"}, "{count:12}", "{count:12}"),
        ({"type": "object"}, "[1,2]", "[1,2]"),
        ({"type": "array"}, '{"count":12}', '{"count":12}'),
        ({"$ref": "#/$defs/options"}, '{"count":12}', '{"count":12}'),
    ],
)
def test_nested_container_decoding_is_lossless_and_schema_directed(schema, value, expected):
    reg = ToolRegistry()
    reg.register(_tool({
        "type": "object", "properties": {"value": schema}, "$defs": {"options": {"type": "object"}},
    }))
    original = {"value": value}
    schemas_before = json.dumps(reg.get_definitions())
    arguments_before = json.dumps(original)

    _, params, _ = reg.prepare_call("sample", original)

    assert params == {"value": expected}
    assert json.dumps(original) == arguments_before
    assert json.dumps(reg.get_definitions()) == schemas_before


def test_decoded_json_container_is_then_cast():
    # nanobot tests/agent/test_runner_tool_execution.py: '["1","2"]' for an integer array reaches the tool as [1, 2]
    schema = {"type": "object", "properties": {"value": {"type": "array", "items": {"type": "integer"}}}}
    assert cast_params({"value": '["1","2"]'}, schema) == {"value": [1, 2]}
