"""JSON Schema checks and safe type casting for tool arguments.

Ports nanobot/agent/tools/base.py (``Schema.validate_json_schema_value``,
``Tool.cast_params``) and the ``tool_parameters_schema`` helper from
nanobot/agent/tools/schema.py. Error texts match nanobot's exactly, because
the model reads them and corrects its call.
"""

from __future__ import annotations

import math
from typing import Any

_JSON_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}

_BOOL_TRUE = frozenset(("true", "1", "yes"))
_BOOL_FALSE = frozenset(("false", "0", "no"))


def object_schema(
    *,
    required: list[str] | None = None,
    additional_properties: bool = False,
    **properties: dict[str, Any],
) -> dict[str, Any]:
    """Build a tool's root parameter schema.

    Like nanobot's built-in tools, unknown arguments are rejected by default
    (``additionalProperties: false``), so a misspelled argument is reported
    instead of silently ignored.
    """
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = list(required)
    schema["additionalProperties"] = additional_properties
    return schema


def _resolve_type(t: Any) -> str | None:
    """``["string", "null"]`` -> ``"string"``."""
    if isinstance(t, list):
        return next((x for x in t if x != "null"), None)
    return t


def _subpath(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def validate_json_schema_value(val: Any, schema: dict[str, Any] | bool, path: str = "") -> list[str]:
    """Check ``val`` against a JSON Schema fragment. Returns error messages; empty means valid."""
    label = path or "parameter"
    if isinstance(schema, bool):
        return [] if schema else [f"{label} is not allowed by schema"]

    raw_type = schema.get("type")
    nullable = (isinstance(raw_type, list) and "null" in raw_type) or schema.get("nullable", False)
    t = _resolve_type(raw_type)

    if nullable and val is None:
        return []
    if t == "integer" and (not isinstance(val, int) or isinstance(val, bool)):
        return [f"{label} should be integer"]
    if t == "number" and (not isinstance(val, (int, float)) or isinstance(val, bool)):
        return [f"{label} should be number"]
    if t in _JSON_TYPE_MAP and t not in ("integer", "number") and not isinstance(val, _JSON_TYPE_MAP[t]):
        return [f"{label} should be {t}"]
    if t == "number" and isinstance(val, float) and not math.isfinite(val):
        return [f"{label} must be finite"]

    errors: list[str] = []
    if "enum" in schema and val not in schema["enum"]:
        errors.append(f"{label} must be one of {schema['enum']}")
    if t in ("integer", "number"):
        if "minimum" in schema and val < schema["minimum"]:
            errors.append(f"{label} must be >= {schema['minimum']}")
        if "maximum" in schema and val > schema["maximum"]:
            errors.append(f"{label} must be <= {schema['maximum']}")
    if t == "string":
        if "minLength" in schema and len(val) < schema["minLength"]:
            errors.append(f"{label} must be at least {schema['minLength']} chars")
        if "maxLength" in schema and len(val) > schema["maxLength"]:
            errors.append(f"{label} must be at most {schema['maxLength']} chars")
    if t == "object":
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in val:
                errors.append(f"missing required {_subpath(path, key)}")
        additional = schema.get("additionalProperties", True)
        for key, value in val.items():
            if key in props:
                errors.extend(validate_json_schema_value(value, props[key], _subpath(path, key)))
            elif additional is False:
                errors.append(f"unexpected parameter {_subpath(path, key)}")
            elif isinstance(additional, dict):
                errors.extend(validate_json_schema_value(value, additional, _subpath(path, key)))
    if t == "array":
        if "minItems" in schema and len(val) < schema["minItems"]:
            errors.append(f"{label} must have at least {schema['minItems']} items")
        if "maxItems" in schema and len(val) > schema["maxItems"]:
            errors.append(f"{label} must be at most {schema['maxItems']} items")
        if "items" in schema:
            prefix = f"{path}[{{}}]" if path else "[{}]"
            for i, item in enumerate(val):
                errors.extend(validate_json_schema_value(item, schema["items"], prefix.format(i)))
    return errors


def validate_params(params: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Validate a tool call's arguments against the tool's root schema."""
    if not isinstance(params, dict):
        return [f"parameters must be an object, got {type(params).__name__}"]
    return validate_json_schema_value(params, {**schema, "type": "object"}, "")


def cast_params(params: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Apply safe casts before validation: ``"5"`` -> ``5``, ``"true"`` -> ``True``, ...

    Models sometimes send numbers and booleans as strings. Only unambiguous
    conversions are made; anything else is left for validation to reject.
    """
    if schema.get("type", "object") != "object":
        return params
    return _cast_object(params, schema)


def _cast_object(obj: Any, schema: dict[str, Any]) -> Any:
    if not isinstance(obj, dict):
        return obj
    props = schema.get("properties", {})
    additional = schema.get("additionalProperties")
    casted: dict[str, Any] = {}
    for key, value in obj.items():
        if key in props:
            casted[key] = _cast_value(value, props[key])
        elif isinstance(additional, dict):
            casted[key] = _cast_value(value, additional)
        else:
            casted[key] = value
    return casted


def _cast_value(val: Any, schema: dict[str, Any] | bool) -> Any:
    if isinstance(schema, bool):
        return val
    t = _resolve_type(schema.get("type"))
    if t == "boolean" and isinstance(val, bool):
        return val
    if t == "integer" and isinstance(val, int) and not isinstance(val, bool):
        return val
    if t in _JSON_TYPE_MAP and t not in ("boolean", "integer", "array", "object"):
        if isinstance(val, _JSON_TYPE_MAP[t]):
            return val
    if isinstance(val, str) and t in ("integer", "number"):
        try:
            return int(val) if t == "integer" else float(val)
        except ValueError:
            return val
    if t == "string":
        return val if val is None else str(val)
    if t == "boolean" and isinstance(val, str):
        low = val.lower()
        if low in _BOOL_TRUE:
            return True
        if low in _BOOL_FALSE:
            return False
        return val
    if t == "array" and isinstance(val, list):
        items = schema.get("items")
        return [_cast_value(x, items) for x in val] if items else val
    if t == "object" and isinstance(val, dict):
        return _cast_object(val, schema)
    return val
