from agent.tools import calculator

  # rag_search and web_search callables are patched in by later tasks
  # via register_tool(); until then calling them raises.

def _not_implemented(*_args, **kwargs):
    raise NotImplementedError("This tool is not yet implemented.")

_SCHEMAS = {
    "calculator": {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression (+ - * / ** and paranthesis).",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic Expression, e.g. '47 * 2' ",
                    }
                },
                "required": ["expression"]
            }
        }
    },
    "rag_search": {
          "type": "function",
          "function": {
              "name": "rag_search",
              "description": "Search the user's local notes/docs for relevant passages.",
              "parameters": {
                  "type": "object",
                  "properties": {
                      "query": {"type": "string", "description": "Search query"}
                  },
                  "required": ["query"],
              },
          },
      },
    "web_search": {
          "type": "function",
          "function": {
              "name": "web_search",
              "description": "Search the public web for current information.",
              "parameters": {
                  "type": "object",
                  "properties": {
                      "query": {"type": "string", "description": "Search query"}
                  },
                  "required": ["query"],
              },
          },
      },
}

TOOL_SCHEMAS = list(_SCHEMAS.values())

TOOL_FUNCS = {
    "calculator": lambda expression: calculator.evaluate(expression),
    "rag_search": _not_implemented,
    "web_search": _not_implemented,
}

REQUIRES_CONFIRMATION = {"web_search"}

def register_tool(name: str, func) -> None:
    """Replace a placeholder tool implementation."""
    if name not in TOOL_FUNCS:
        raise KeyError(f"unknown tool: {name}")
    TOOL_FUNCS[name] = func


def validate_args(tool_name: str, args: dict) -> list[str]:
    if tool_name not in _SCHEMAS:
        return [f"unknown tool: {tool_name}"]
    params = _SCHEMAS[tool_name]["function"]["parameters"]
    required = params.get("required", [])
    properties = params.get("properties", {})
    errors = []
    for field in required:
        if field not in args:
            errors.append(f"missing required argument: {field}")
            continue
        expected_type = properties[field]["type"]
        if expected_type == "string" and not isinstance(args[field], str):
            errors.append(f"argument '{field}' must be a string")
    return errors

