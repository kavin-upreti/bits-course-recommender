"""Check the LLM's tool arguments against TOOL_SCHEMAS (todo.md 9.3). Never raises: a problem comes back as an
error string, which the agent returns to the LLM as the tool result so it can fix its call."""
from .tool_schemas import TOOL_SCHEMAS

SCHEMAS = {tool["name"]: tool["parameters"] for tool in TOOL_SCHEMAS}
EMPTY_TEXT = {"", "none", "null"}  # some models write "None" instead of leaving an argument out
TYPE_NAMES = {"string": "a string", "integer": "a whole number", "boolean": "true or false", "array": "a list", "object": "an object"}


def _check(name: str, value, schema: dict):
    """(clean value, error) for one value against its schema."""
    kind = schema["type"]
    if kind == "integer" and isinstance(value, float) and value.is_integer():
        value = int(value)  # why: some providers send every number as a float
    if kind == "array" and isinstance(value, str):
        value = [value]  # why: a model may send one topic as plain text
    if kind == "boolean" and isinstance(value, str) and value.lower() in ("true", "false"):
        value = value.lower() == "true"  # why: some models write booleans as text
    valid = {"string": isinstance(value, str), "boolean": isinstance(value, bool),
             "integer": isinstance(value, int) and not isinstance(value, bool),
             "array": isinstance(value, list), "object": isinstance(value, dict)}[kind]
    if not valid:
        return None, f"{name} must be {TYPE_NAMES[kind]}."
    if "enum" in schema and value not in schema["enum"]:
        return None, f"{name} must be one of: {', '.join(schema['enum'])}."
    if kind == "string" and len(value) > schema.get("maxLength", len(value)):
        return None, f"{name} is too long (at most {schema['maxLength']} characters)."
    if kind == "integer" and not schema.get("minimum", value) <= value <= schema.get("maximum", value):
        return None, f"{name} must be between {schema['minimum']} and {schema['maximum']}."
    if kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", len(value)):
            return None, f"{name} must have between {schema.get('minItems', 0)} and {schema['maxItems']} items."
        for item in value:
            error = _check(f"each item of {name}", item, schema["items"])[1]
            if error:
                return None, error
    if kind == "object":
        return _check_object(name, value, schema)
    return value, None


def _check_object(name: str, args: dict, schema: dict):
    """Known keys only; missing / None / "" optional values dropped; required ones present."""
    properties = schema.get("properties", {})
    clean = {}
    for key, value in args.items():
        if key not in properties:
            what = "filter" if name == "filters" else "argument"
            allowed = ", ".join(properties) or "none"
            return None, f"Unknown {what} {key}. Allowed: {allowed}."
        if value is None or (isinstance(value, str) and value.strip().lower() in EMPTY_TEXT):
            continue
        clean[key], error = _check(key, value, properties[key])
        if error:
            return None, error
    missing = [key for key in schema.get("required", []) if key not in clean]
    if missing:
        return None, f"Missing required argument: {', '.join(missing)}."
    return clean, None


def validate_call(name: str, args: dict | None) -> tuple[dict, str | None]:
    """(clean args, error or None)."""
    if name not in SCHEMAS:
        return {}, f"Unknown tool {name}. Available: {', '.join(SCHEMAS)}."
    clean, error = _check_object(name, args or {}, SCHEMAS[name])
    return (clean or {}), error
