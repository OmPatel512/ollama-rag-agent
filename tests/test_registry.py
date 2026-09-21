from agent.tools.registry import TOOL_SCHEMAS, TOOL_FUNCS, REQUIRES_CONFIRMATION, validate_args

def test_calculator_registered():
    assert "calculator" in TOOL_FUNCS
    assert TOOL_FUNCS["calculator"]("2 + 2") == 4

def test_schemas_have_required_shape():
    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert {"calculator", "rag_search", "web_search"} <= names
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]

def test_web_search_requires_confirmation():
    assert "web_search" in REQUIRES_CONFIRMATION
    assert "rag_search" not in REQUIRES_CONFIRMATION
    assert "calculator" not in REQUIRES_CONFIRMATION

def test_validate_args_missing_requires_field():
    errors = validate_args("calculator", {})
    assert errors == ["missing required argument: expression"]

def test_validate_args_wrong_type():
    errors = validate_args("calculator", {"expression": 123})
    assert errors == ["argument 'expression' must be a string"]

def test_validate_args_ok():
    assert validate_args("calculator", {"expression": "1+1"}) == []

def test_validate_args_unknown_tool():
    errors = validate_args("nonexistent_tool", {})
    assert errors == ["unknown tool: nonexistent_tool"]


