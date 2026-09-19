# Demo Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a hand-rolled personal-knowledge-assistant agent (RAG + calculator + web search tools, human-in-the-loop, tracing, evals) using Ollama + Qdrant, to learn LLM/agent mechanics by implementing them directly instead of via a framework.

**Architecture:** A plain-Python plan-act-observe loop (`agent/loop.py`) drives one Ollama chat model (`qwen2.5:0.5b`) with tool-calling. Tools are plug-in modules registered in `agent/tools/__init__.py` with JSON schemas. Retrieval goes through Qdrant (embeddings via `nomic-embed-text`). Every LLM call and tool call is written to a JSONL trace log. A CLI REPL (`cli.py`) is the only interface.

**Tech Stack:** Python 3.11, `ollama` (python client), `qdrant-client`, `duckduckgo-search`, `pyyaml` (evals), `pytest`, Docker (Qdrant only).

**Spec:** `docs/superpowers/specs/2026-09-19-demo-agent-design.md`

## Global Constraints

- Python 3.11.
- LLM model: `qwen2.5:0.5b` (Ollama). Embedding model: `nomic-embed-text` (Ollama).
- Vector DB: Qdrant, run via `docker-compose.yml`.
- Web search: `duckduckgo-search` package, no API key.
- No agent framework (no LangChain/LangGraph) — hand-rolled loop only.
- No `eval()` anywhere — calculator uses `ast`.
- In-session conversation history only; no cross-session persistence.
- CLI only; no web UI, no streaming to a UI.

---

### Task 1: Project scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `docker-compose.yml`
- Create: `.gitignore`
- Create: `agent/__init__.py`
- Create: `agent/tools/__init__.py` (empty for now, filled in Task 4)
- Create: `logs/.gitkeep`

**Interfaces:**
- Produces: a `logs/` directory (git-tracked via `.gitkeep`) that `agent/tracing.py` (Task 5) writes into; a running Qdrant instance on `localhost:6333` that Task 8 connects to.

- [ ] **Step 1: Write `requirements.txt`**

```
ollama==0.4.7
qdrant-client==1.12.1
duckduckgo-search==6.3.7
pyyaml==6.0.2
pytest==8.3.4
```

- [ ] **Step 2: Write `docker-compose.yml`**

```yaml
services:
  qdrant:
    image: qdrant/qdrant:v1.12.4
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  qdrant_data:
```

- [ ] **Step 3: Write `.gitignore`**

```
__pycache__/
*.pyc
.venv/
logs/*.jsonl
!logs/.gitkeep
```

- [ ] **Step 4: Create package/dir stubs**

```bash
mkdir -p agent/tools logs tests evals
touch agent/__init__.py agent/tools/__init__.py logs/.gitkeep
```

- [ ] **Step 5: Install dependencies and bring up Qdrant**

Run: `pip install -r requirements.txt`
Run: `docker compose up -d`
Expected: `docker compose ps` shows the `qdrant` service `running`, and `curl -s localhost:6333/collections` returns `{"result":{"collections":[]},"status":"ok",...}`.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt docker-compose.yml .gitignore agent/ logs/.gitkeep tests evals
git commit -m "chore: project scaffolding, requirements, qdrant compose"
```

---

### Task 2: Calculator tool

**Files:**
- Create: `agent/tools/calculator.py`
- Test: `tests/test_calculator.py`

**Interfaces:**
- Produces: `calculator.evaluate(expression: str) -> float` — raises `ValueError` with a human-readable message on any disallowed or malformed expression. Consumed by the tool registry in Task 4.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_calculator.py
import pytest
from agent.tools.calculator import evaluate

def test_addition():
    assert evaluate("2 + 3") == 5

def test_precedence_and_parens():
    assert evaluate("(2 + 3) * 4") == 20

def test_power_and_division():
    assert evaluate("2 ** 10 / 4") == 256

def test_rejects_names():
    with pytest.raises(ValueError):
        evaluate("__import__('os').system('ls')")

def test_rejects_function_calls():
    with pytest.raises(ValueError):
        evaluate("abs(-5)")

def test_rejects_garbage():
    with pytest.raises(ValueError):
        evaluate("not a math expression")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_calculator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.tools.calculator'`

- [ ] **Step 3: Write minimal implementation**

```python
# agent/tools/calculator.py
import ast
import operator

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def evaluate(expression: str) -> float:
    """Evaluate a basic arithmetic expression. Only numeric literals and
    + - * / ** ( ) are allowed. Raises ValueError on anything else."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"invalid expression: {expression}") from e
    return _eval_node(tree.body)


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _ALLOWED_BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"disallowed expression element: {ast.dump(node)}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_calculator.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add agent/tools/calculator.py tests/test_calculator.py
git commit -m "feat: safe calculator tool using ast, no eval()"
```

---

### Task 3: Conversation memory & truncation

**Files:**
- Create: `agent/memory.py`
- Test: `tests/test_memory.py`

**Interfaces:**
- Produces: `estimate_tokens(text: str) -> int` and `truncate_history(messages: list[dict], max_tokens: int) -> list[dict]`. Message dicts have shape `{"role": "system"|"user"|"assistant"|"tool", "content": str, ...}`. Consumed by `agent/loop.py` (Task 10).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_memory.py
from agent.memory import estimate_tokens, truncate_history

def test_estimate_tokens_roughly_chars_over_four():
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 100) == 25

def test_truncate_keeps_system_message():
    messages = [
        {"role": "system", "content": "s" * 400},
        {"role": "user", "content": "u" * 400},
        {"role": "assistant", "content": "a" * 400},
    ]
    result = truncate_history(messages, max_tokens=100)
    assert result[0]["role"] == "system"

def test_truncate_drops_oldest_non_system_first():
    messages = [
        {"role": "system", "content": "s" * 40},
        {"role": "user", "content": "oldest"},
        {"role": "assistant", "content": "middle"},
        {"role": "user", "content": "newest"},
    ]
    result = truncate_history(messages, max_tokens=15)
    contents = [m["content"] for m in result]
    assert "newest" in contents
    assert "oldest" not in contents

def test_truncate_no_op_when_under_budget():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]
    result = truncate_history(messages, max_tokens=1000)
    assert result == messages
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_memory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.memory'`

- [ ] **Step 3: Write minimal implementation**

```python
# agent/memory.py

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token, minimum 1."""
    return max(1, len(text) // 4)


def _total_tokens(messages: list[dict]) -> int:
    return sum(estimate_tokens(m["content"]) for m in messages)


def truncate_history(messages: list[dict], max_tokens: int) -> list[dict]:
    """Keep the leading system message (if any) plus the most recent
    messages such that total estimated tokens <= max_tokens. Drops the
    oldest non-system messages first."""
    if _total_tokens(messages) <= max_tokens:
        return messages

    has_system = bool(messages) and messages[0]["role"] == "system"
    system_msg = [messages[0]] if has_system else []
    rest = messages[1:] if has_system else messages[:]

    budget = max_tokens - _total_tokens(system_msg)
    kept: list[dict] = []
    for msg in reversed(rest):
        cost = estimate_tokens(msg["content"])
        if cost > budget and kept:
            break
        kept.insert(0, msg)
        budget -= cost

    return system_msg + kept
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_memory.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add agent/memory.py tests/test_memory.py
git commit -m "feat: conversation history truncation by token budget"
```

---

### Task 4: Tool registry & argument validation

**Files:**
- Create: `agent/tools/registry.py`
- Modify: `agent/tools/__init__.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: `calculator.evaluate(expression: str) -> float` (Task 2).
- Produces:
  - `TOOL_SCHEMAS: list[dict]` — Ollama-format tool schemas, passed to `llm.py` (Task 7).
  - `TOOL_FUNCS: dict[str, callable]` — name -> callable, used by `loop.py` (Task 10) to dispatch.
  - `REQUIRES_CONFIRMATION: set[str]` — tool names needing the human-in-the-loop gate.
  - `validate_args(tool_name: str, args: dict) -> list[str]` — list of error strings, empty if valid.
  (`rag_search` and `web_search` are registered here as placeholders raising `NotImplementedError` until Tasks 8-9 replace them.)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_registry.py
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
    assert "calculator" not in REQUIRES_CONFIRMATION
    assert "rag_search" not in REQUIRES_CONFIRMATION

def test_validate_args_missing_required_field():
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.tools.registry'`

- [ ] **Step 3: Write minimal implementation**

```python
# agent/tools/registry.py
from agent.tools import calculator

# rag_search and web_search callables are patched in by Task 8 and Task 9
# respectively via register_tool(); until then calling them raises.


def _not_implemented(*_args, **_kwargs):
    raise NotImplementedError("tool not wired up yet")


_SCHEMAS = {
    "calculator": {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression (+ - * / ** and parentheses).",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression, e.g. '47 * 89'",
                    }
                },
                "required": ["expression"],
            },
        },
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
    """Replace a placeholder tool implementation. Used by ingest/tool
    modules that have dependencies (Qdrant, network) we don't want
    imported by every consumer of the registry."""
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_registry.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Update `agent/tools/__init__.py` to re-export the registry**

```python
# agent/tools/__init__.py
from agent.tools.registry import (
    TOOL_SCHEMAS,
    TOOL_FUNCS,
    REQUIRES_CONFIRMATION,
    validate_args,
    register_tool,
)

__all__ = [
    "TOOL_SCHEMAS",
    "TOOL_FUNCS",
    "REQUIRES_CONFIRMATION",
    "validate_args",
    "register_tool",
]
```

- [ ] **Step 6: Commit**

```bash
git add agent/tools/registry.py agent/tools/__init__.py tests/test_registry.py
git commit -m "feat: tool registry with schemas and argument validation"
```

---

### Task 5: Tracing / structured logging

**Files:**
- Create: `agent/tracing.py`
- Test: `tests/test_tracing.py`

**Interfaces:**
- Produces: `Tracer(log_dir="logs", session_id=None)` with `.log(event: str, **fields)` writing one JSON line per call to `logs/<session_id>.jsonl`, each line including `ts` (float epoch seconds) and `event`. `session_id` defaults to a timestamp-based string if not given. Consumed by `agent/loop.py` (Task 10) and `agent/llm.py` (Task 7).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tracing.py
import json
from agent.tracing import Tracer

def test_log_writes_one_json_line(tmp_path):
    tracer = Tracer(log_dir=tmp_path, session_id="test-session")
    tracer.log("llm_call", duration_ms=42, tokens=10)

    log_file = tmp_path / "test-session.jsonl"
    assert log_file.exists()
    lines = log_file.read_text().strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["event"] == "llm_call"
    assert record["duration_ms"] == 42
    assert record["tokens"] == 10
    assert "ts" in record

def test_log_appends_multiple_events(tmp_path):
    tracer = Tracer(log_dir=tmp_path, session_id="test-session")
    tracer.log("llm_call", duration_ms=1)
    tracer.log("tool_call", tool="calculator")

    lines = (tmp_path / "test-session.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["tool"] == "calculator"

def test_default_session_id_is_generated():
    tracer = Tracer(log_dir="logs")
    assert tracer.session_id
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tracing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.tracing'`

- [ ] **Step 3: Write minimal implementation**

```python
# agent/tracing.py
import json
import time
import uuid
from pathlib import Path


class Tracer:
    def __init__(self, log_dir="logs", session_id: str | None = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
        self.path = self.log_dir / f"{self.session_id}.jsonl"

    def log(self, event: str, **fields) -> None:
        record = {"ts": time.time(), "event": event, **fields}
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tracing.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add agent/tracing.py tests/test_tracing.py
git commit -m "feat: JSONL trace logger for llm/tool events"
```

---

### Task 6: Human-in-the-loop confirmation gate

**Files:**
- Create: `agent/confirm.py`
- Test: `tests/test_confirm.py`

**Interfaces:**
- Produces: `confirm_tool_call(tool_name: str, args: dict, input_func=input, print_func=print) -> bool`. Consumed by `agent/loop.py` (Task 10) for any tool in `REQUIRES_CONFIRMATION`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_confirm.py
from agent.confirm import confirm_tool_call

def test_yes_confirms():
    result = confirm_tool_call("web_search", {"query": "x"}, input_func=lambda _: "y")
    assert result is True

def test_no_declines():
    result = confirm_tool_call("web_search", {"query": "x"}, input_func=lambda _: "n")
    assert result is False

def test_empty_input_declines():
    result = confirm_tool_call("web_search", {"query": "x"}, input_func=lambda _: "")
    assert result is False

def test_prints_the_proposed_call():
    printed = []
    confirm_tool_call(
        "web_search",
        {"query": "capital of France"},
        input_func=lambda _: "n",
        print_func=printed.append,
    )
    assert any("web_search" in line and "capital of France" in line for line in printed)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_confirm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.confirm'`

- [ ] **Step 3: Write minimal implementation**

```python
# agent/confirm.py

def confirm_tool_call(tool_name: str, args: dict, input_func=input, print_func=print) -> bool:
    """Prompt the user to approve a tool call. Returns True only on an
    explicit 'y' answer; any other input (including empty) declines."""
    print_func(f"Agent wants to call `{tool_name}` with args: {args}")
    answer = input_func("Allow this call? [y/N]: ").strip().lower()
    return answer == "y"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_confirm.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add agent/confirm.py tests/test_confirm.py
git commit -m "feat: human-in-the-loop confirmation gate for risky tool calls"
```

---

### Task 7: Ollama LLM client wrapper

**Files:**
- Create: `agent/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (standalone wrapper around the `ollama` package).
- Produces:
  - `class LLMConnectionError(Exception)`, `class LLMTimeoutError(Exception)`.
  - `class OllamaClient` with `__init__(self, model="qwen2.5:0.5b", host="http://localhost:11434", timeout=30, max_tokens=512, client=None)` (the `client` param lets tests inject a fake) and `.chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict` returning the assistant message dict (`{"role": "assistant", "content": str, "tool_calls": [...] | None}`).
  Consumed by `agent/loop.py` (Task 10).

- [ ] **Step 1: Write the failing tests (using a fake client, no live Ollama needed)**

```python
# tests/test_llm.py
import pytest
from agent.llm import OllamaClient, LLMConnectionError, LLMTimeoutError


class FakeOllamaClient:
    def __init__(self, response=None, raise_exc=None):
        self.response = response
        self.raise_exc = raise_exc
        self.last_call_kwargs = None

    def chat(self, **kwargs):
        self.last_call_kwargs = kwargs
        if self.raise_exc:
            raise self.raise_exc
        return self.response


def test_chat_returns_assistant_message():
    fake = FakeOllamaClient(response={"message": {"role": "assistant", "content": "hi", "tool_calls": None}})
    client = OllamaClient(client=fake)
    result = client.chat(messages=[{"role": "user", "content": "hello"}])
    assert result == {"role": "assistant", "content": "hi", "tool_calls": None}

def test_chat_passes_tools_and_max_tokens():
    fake = FakeOllamaClient(response={"message": {"role": "assistant", "content": "hi", "tool_calls": None}})
    client = OllamaClient(client=fake, max_tokens=256, model="qwen2.5:0.5b")
    tools = [{"type": "function", "function": {"name": "calculator"}}]
    client.chat(messages=[{"role": "user", "content": "2+2"}], tools=tools)
    assert fake.last_call_kwargs["model"] == "qwen2.5:0.5b"
    assert fake.last_call_kwargs["tools"] == tools
    assert fake.last_call_kwargs["options"]["num_predict"] == 256

def test_chat_wraps_connection_errors():
    fake = FakeOllamaClient(raise_exc=ConnectionError("refused"))
    client = OllamaClient(client=fake)
    with pytest.raises(LLMConnectionError):
        client.chat(messages=[{"role": "user", "content": "hi"}])

def test_chat_wraps_timeout_errors():
    fake = FakeOllamaClient(raise_exc=TimeoutError("slow"))
    client = OllamaClient(client=fake)
    with pytest.raises(LLMTimeoutError):
        client.chat(messages=[{"role": "user", "content": "hi"}])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.llm'`

- [ ] **Step 3: Write minimal implementation**

```python
# agent/llm.py
import ollama


class LLMConnectionError(Exception):
    pass


class LLMTimeoutError(Exception):
    pass


class OllamaClient:
    def __init__(
        self,
        model: str = "qwen2.5:0.5b",
        host: str = "http://localhost:11434",
        timeout: int = 30,
        max_tokens: int = 512,
        client=None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.client = client or ollama.Client(host=host, timeout=timeout)

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                tools=tools or [],
                options={"num_predict": self.max_tokens},
            )
        except TimeoutError as e:
            raise LLMTimeoutError(str(e)) from e
        except (ConnectionError, OSError) as e:
            raise LLMConnectionError(str(e)) from e
        return response["message"]

    def check_available(self) -> None:
        """Raise LLMConnectionError with a helpful message if Ollama or
        the configured model isn't available. Call once at startup."""
        try:
            models = self.client.list()
        except Exception as e:
            raise LLMConnectionError(
                f"Cannot reach Ollama. Is the daemon running? ({e})"
            ) from e
        names = [m["model"] for m in models.get("models", [])]
        if not any(self.model in n for n in names):
            raise LLMConnectionError(
                f"Model '{self.model}' not found locally. Run: ollama pull {self.model}"
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Manual smoke test against a real Ollama (requires `ollama pull qwen2.5:0.5b` done)**

```bash
python -c "
from agent.llm import OllamaClient
c = OllamaClient()
c.check_available()
print(c.chat(messages=[{'role': 'user', 'content': 'Say hi in 3 words.'}]))
"
```
Expected: prints an assistant message dict with non-empty `content`.

- [ ] **Step 6: Commit**

```bash
git add agent/llm.py tests/test_llm.py
git commit -m "feat: Ollama chat client wrapper with error handling"
```

---

### Task 8: RAG search tool + document ingestion

**Files:**
- Create: `agent/tools/rag_search.py`
- Create: `ingest.py`
- Modify: `cli.py` (not yet created — this task creates the ingestion path only; wiring into the registry happens in Step 5 below)
- Test: `tests/test_rag_search.py`
- Test fixtures: `tests/fixtures/notes/note1.md`, `tests/fixtures/notes/note2.md`

**Interfaces:**
- Consumes: `agent.tools.registry.register_tool` (Task 4).
- Produces:
  - `agent/tools/rag_search.py`: `embed_text(text: str, ollama_client=None) -> list[float]`, `search(query: str, collection_name="notes", limit=3, ollama_client=None, qdrant_client=None) -> str` (returns a formatted string of the top matching chunks; returns `"error: vector store unavailable"` on Qdrant connection failure rather than raising).
  - `ingest.py`: `chunk_text(text: str, chunk_size=500, overlap=50) -> list[str]`, `ingest_folder(folder: str, collection_name="notes", ollama_client=None, qdrant_client=None) -> int` (returns number of chunks upserted).

- [ ] **Step 1: Write the failing tests (mocking Qdrant and Ollama embeddings)**

```python
# tests/test_rag_search.py
from unittest.mock import MagicMock
from agent.tools.rag_search import embed_text, search
from ingest import chunk_text, ingest_folder


def test_chunk_text_splits_on_size_with_overlap():
    text = "word " * 300  # ~1500 chars
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)


def test_chunk_text_short_text_single_chunk():
    assert chunk_text("short text", chunk_size=500, overlap=50) == ["short text"]


def test_embed_text_calls_ollama_embeddings():
    fake_ollama = MagicMock()
    fake_ollama.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}
    result = embed_text("hello", ollama_client=fake_ollama)
    assert result == [0.1, 0.2, 0.3]
    fake_ollama.embeddings.assert_called_once_with(model="nomic-embed-text", prompt="hello")


def test_search_formats_results():
    fake_ollama = MagicMock()
    fake_ollama.embeddings.return_value = {"embedding": [0.1, 0.2]}
    fake_qdrant = MagicMock()
    hit = MagicMock(payload={"text": "Qdrant is a vector database."}, score=0.9)
    fake_qdrant.search.return_value = [hit]

    result = search("what is qdrant", ollama_client=fake_ollama, qdrant_client=fake_qdrant)
    assert "Qdrant is a vector database." in result


def test_search_returns_error_string_on_qdrant_failure():
    fake_ollama = MagicMock()
    fake_ollama.embeddings.return_value = {"embedding": [0.1, 0.2]}
    fake_qdrant = MagicMock()
    fake_qdrant.search.side_effect = ConnectionError("refused")

    result = search("anything", ollama_client=fake_ollama, qdrant_client=fake_qdrant)
    assert result == "error: vector store unavailable"


def test_ingest_folder_upserts_expected_chunk_count(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("word " * 300)
    (notes / "b.md").write_text("short note")

    fake_ollama = MagicMock()
    fake_ollama.embeddings.return_value = {"embedding": [0.1, 0.2]}
    fake_qdrant = MagicMock()

    count = ingest_folder(str(notes), ollama_client=fake_ollama, qdrant_client=fake_qdrant)

    assert count >= 2  # a.md produces multiple chunks + b.md produces one
    fake_qdrant.recreate_collection.assert_called_once()
    assert fake_qdrant.upsert.call_count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rag_search.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.tools.rag_search'`

- [ ] **Step 3: Write minimal implementation**

```python
# agent/tools/rag_search.py
import ollama as ollama_module
from qdrant_client import QdrantClient

EMBED_MODEL = "nomic-embed-text"


def embed_text(text: str, ollama_client=None) -> list[float]:
    client = ollama_client or ollama_module
    response = client.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


def search(
    query: str,
    collection_name: str = "notes",
    limit: int = 3,
    ollama_client=None,
    qdrant_client=None,
) -> str:
    vector = embed_text(query, ollama_client=ollama_client)
    qclient = qdrant_client or QdrantClient(host="localhost", port=6333)
    try:
        hits = qclient.search(collection_name=collection_name, query_vector=vector, limit=limit)
    except (ConnectionError, OSError) as e:
        return "error: vector store unavailable"

    if not hits:
        return "no relevant notes found"

    return "\n---\n".join(hit.payload["text"] for hit in hits)
```

```python
# ingest.py
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from agent.tools.rag_search import embed_text, EMBED_MODEL

VECTOR_SIZE = 768  # nomic-embed-text output dimension


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def ingest_folder(
    folder: str,
    collection_name: str = "notes",
    ollama_client=None,
    qdrant_client=None,
) -> int:
    qclient = qdrant_client or QdrantClient(host="localhost", port=6333)
    qclient.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    points = []
    point_id = 0
    for path in sorted(Path(folder).glob("*.md")):
        text = path.read_text()
        for chunk in chunk_text(text):
            vector = embed_text(chunk, ollama_client=ollama_client)
            points.append(
                PointStruct(id=point_id, vector=vector, payload={"text": chunk, "source": str(path)})
            )
            point_id += 1

    if points:
        qclient.upsert(collection_name=collection_name, points=points)
    return len(points)


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "notes"
    count = ingest_folder(folder)
    print(f"Ingested {count} chunks from {folder}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rag_search.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Wire `rag_search` into the tool registry**

```python
# add to agent/tools/__init__.py, after the existing imports
from agent.tools.rag_search import search as _rag_search
from agent.tools.registry import register_tool as _register_tool

_register_tool("rag_search", lambda query: _rag_search(query))
```

- [ ] **Step 6: Manual ingest smoke test against live Qdrant + Ollama**

```bash
mkdir -p notes
echo "Qdrant is an open-source vector database used for similarity search." > notes/qdrant.md
python ingest.py notes
```
Expected: prints `Ingested N chunks from notes` with N >= 1, and `curl -s localhost:6333/collections/notes` shows `points_count` >= 1.

- [ ] **Step 7: Commit**

```bash
git add agent/tools/rag_search.py ingest.py agent/tools/__init__.py tests/test_rag_search.py
git commit -m "feat: RAG search tool and document ingestion into Qdrant"
```

---

### Task 9: Web search tool

**Files:**
- Create: `agent/tools/web_search.py`
- Modify: `agent/tools/__init__.py`
- Test: `tests/test_web_search.py`

**Interfaces:**
- Consumes: `agent.tools.registry.register_tool` (Task 4).
- Produces: `search(query: str, max_results=3, ddgs_client=None) -> str` — formatted string of result titles + snippets; returns `"error: web search unavailable"` on any exception from the search backend.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_web_search.py
from unittest.mock import MagicMock
from agent.tools.web_search import search


def test_search_formats_results():
    fake_ddgs = MagicMock()
    fake_ddgs.text.return_value = [
        {"title": "Paris", "body": "Paris is the capital of France."},
    ]
    result = search("capital of France", ddgs_client=fake_ddgs)
    assert "Paris" in result
    assert "capital of France" in result


def test_search_returns_error_string_on_failure():
    fake_ddgs = MagicMock()
    fake_ddgs.text.side_effect = Exception("network error")
    result = search("anything", ddgs_client=fake_ddgs)
    assert result == "error: web search unavailable"


def test_search_no_results():
    fake_ddgs = MagicMock()
    fake_ddgs.text.return_value = []
    result = search("obscure query", ddgs_client=fake_ddgs)
    assert result == "no results found"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_web_search.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.tools.web_search'`

- [ ] **Step 3: Write minimal implementation**

```python
# agent/tools/web_search.py
from duckduckgo_search import DDGS


def search(query: str, max_results: int = 3, ddgs_client=None) -> str:
    client = ddgs_client or DDGS()
    try:
        results = list(client.text(query, max_results=max_results))
    except Exception:
        return "error: web search unavailable"

    if not results:
        return "no results found"

    return "\n---\n".join(f"{r['title']}: {r['body']}" for r in results)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_web_search.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Wire `web_search` into the tool registry**

```python
# add to agent/tools/__init__.py
from agent.tools.web_search import search as _web_search
_register_tool("web_search", lambda query: _web_search(query))
```

- [ ] **Step 6: Commit**

```bash
git add agent/tools/web_search.py agent/tools/__init__.py tests/test_web_search.py
git commit -m "feat: DuckDuckGo web search tool"
```

---

### Task 10: Agent loop (plan-act-observe)

**Files:**
- Create: `agent/loop.py`
- Test: `tests/test_loop.py`

**Interfaces:**
- Consumes:
  - `OllamaClient.chat(messages, tools=None) -> dict` (Task 7).
  - `TOOL_FUNCS: dict[str, callable]`, `REQUIRES_CONFIRMATION: set[str]`, `validate_args(name, args) -> list[str]`, `TOOL_SCHEMAS: list[dict]` (Task 4).
  - `truncate_history(messages, max_tokens) -> list[dict]` (Task 3).
  - `confirm_tool_call(tool_name, args, input_func=input) -> bool` (Task 6).
  - `Tracer.log(event, **fields)` (Task 5).
- Produces: `class AgentLoop` with
  - `__init__(self, llm_client, tool_funcs, requires_confirmation, validate_args_fn, tracer, confirm_fn=confirm_tool_call, max_tokens_budget=4000, max_iterations=6)`
  - `run_turn(self, user_input: str, history: list[dict]) -> str` — mutates `history` in place (appends user/assistant/tool messages), returns the final assistant text. Consumed by `cli.py` (Task 11).

- [ ] **Step 1: Write the failing tests (fully mocked: fake LLM client, fake tools)**

```python
# tests/test_loop.py
from agent.loop import AgentLoop
from agent.tracing import Tracer


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append({"messages": [m.copy() for m in messages], "tools": tools})
        return self.responses.pop(0)


def make_tool_call(id_, name, arguments):
    return {"function": {"name": name, "arguments": arguments}, "id": id_}


def test_direct_answer_no_tool_calls(tmp_path):
    llm = FakeLLM([{"role": "assistant", "content": "Paris.", "tool_calls": None}])
    loop = AgentLoop(
        llm_client=llm,
        tool_funcs={},
        requires_confirmation=set(),
        validate_args_fn=lambda name, args: [],
        tracer=Tracer(log_dir=tmp_path),
    )
    history = [{"role": "system", "content": "You are helpful."}]
    result = loop.run_turn("What is the capital of France?", history)
    assert result == "Paris."
    assert history[-1] == {"role": "assistant", "content": "Paris."}


def test_tool_call_then_final_answer(tmp_path):
    llm = FakeLLM([
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [make_tool_call("1", "calculator", {"expression": "2+2"})],
        },
        {"role": "assistant", "content": "The answer is 4.", "tool_calls": None},
    ])
    loop = AgentLoop(
        llm_client=llm,
        tool_funcs={"calculator": lambda expression: 4},
        requires_confirmation=set(),
        validate_args_fn=lambda name, args: [],
        tracer=Tracer(log_dir=tmp_path),
    )
    history = [{"role": "system", "content": "sys"}]
    result = loop.run_turn("what's 2+2", history)
    assert result == "The answer is 4."
    tool_messages = [m for m in history if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert "4" in tool_messages[0]["content"]


def test_confirmation_declined_feeds_synthetic_result(tmp_path):
    llm = FakeLLM([
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [make_tool_call("1", "web_search", {"query": "x"})],
        },
        {"role": "assistant", "content": "Okay, skipping that.", "tool_calls": None},
    ])
    loop = AgentLoop(
        llm_client=llm,
        tool_funcs={"web_search": lambda query: "should not be called"},
        requires_confirmation={"web_search"},
        validate_args_fn=lambda name, args: [],
        tracer=Tracer(log_dir=tmp_path),
        confirm_fn=lambda name, args: False,
    )
    history = [{"role": "system", "content": "sys"}]
    loop.run_turn("search the web for x", history)
    tool_messages = [m for m in history if m["role"] == "tool"]
    assert "declined" in tool_messages[0]["content"]


def test_invalid_args_feeds_error_back_to_model(tmp_path):
    llm = FakeLLM([
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [make_tool_call("1", "calculator", {})],
        },
        {"role": "assistant", "content": "Sorry, let me retry.", "tool_calls": None},
    ])
    loop = AgentLoop(
        llm_client=llm,
        tool_funcs={"calculator": lambda expression: 4},
        requires_confirmation=set(),
        validate_args_fn=lambda name, args: ["missing required argument: expression"],
        tracer=Tracer(log_dir=tmp_path),
    )
    history = [{"role": "system", "content": "sys"}]
    loop.run_turn("what's 2+2", history)
    tool_messages = [m for m in history if m["role"] == "tool"]
    assert "invalid arguments" in tool_messages[0]["content"]


def test_max_iterations_guard(tmp_path):
    always_calls_tool = {
        "role": "assistant",
        "content": "",
        "tool_calls": [make_tool_call("1", "calculator", {"expression": "1+1"})],
    }
    llm = FakeLLM([always_calls_tool] * 10)
    loop = AgentLoop(
        llm_client=llm,
        tool_funcs={"calculator": lambda expression: 2},
        requires_confirmation=set(),
        validate_args_fn=lambda name, args: [],
        tracer=Tracer(log_dir=tmp_path),
        max_iterations=3,
    )
    history = [{"role": "system", "content": "sys"}]
    result = loop.run_turn("loop forever", history)
    assert "gave up" in result.lower()
    assert len(llm.calls) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_loop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.loop'`

- [ ] **Step 3: Write minimal implementation**

```python
# agent/loop.py
import json
import time

from agent.confirm import confirm_tool_call
from agent.memory import truncate_history


class AgentLoop:
    def __init__(
        self,
        llm_client,
        tool_funcs: dict,
        requires_confirmation: set,
        validate_args_fn,
        tracer,
        confirm_fn=confirm_tool_call,
        tool_schemas=None,
        max_tokens_budget: int = 4000,
        max_iterations: int = 6,
    ):
        self.llm_client = llm_client
        self.tool_funcs = tool_funcs
        self.requires_confirmation = requires_confirmation
        self.validate_args_fn = validate_args_fn
        self.tracer = tracer
        self.confirm_fn = confirm_fn
        self.tool_schemas = tool_schemas
        self.max_tokens_budget = max_tokens_budget
        self.max_iterations = max_iterations

    def run_turn(self, user_input: str, history: list[dict]) -> str:
        history.append({"role": "user", "content": user_input})

        for _ in range(self.max_iterations):
            budgeted = truncate_history(history, self.max_tokens_budget)

            start = time.monotonic()
            message = self.llm_client.chat(budgeted, tools=self.tool_schemas)
            duration_ms = (time.monotonic() - start) * 1000
            self.tracer.log("llm_call", duration_ms=duration_ms)

            tool_calls = message.get("tool_calls")
            if not tool_calls:
                history.append({"role": "assistant", "content": message["content"]})
                return message["content"]

            history.append({"role": "assistant", "content": message.get("content", ""), "tool_calls": tool_calls})

            for call in tool_calls:
                name = call["function"]["name"]
                args = call["function"]["arguments"]
                result_text = self._dispatch(name, args)
                history.append({"role": "tool", "content": result_text})

        give_up_msg = "I gave up after reaching the maximum number of tool-call iterations."
        history.append({"role": "assistant", "content": give_up_msg})
        self.tracer.log("max_iterations_reached")
        return give_up_msg

    def _dispatch(self, name: str, args: dict) -> str:
        errors = self.validate_args_fn(name, args)
        if errors:
            self.tracer.log("tool_call", tool=name, args=args, status="invalid_args", errors=errors)
            return f"invalid arguments: {'; '.join(errors)}"

        if name in self.requires_confirmation:
            approved = self.confirm_fn(name, args)
            self.tracer.log("user_confirm", tool=name, approved=approved)
            if not approved:
                return "user declined this tool call"

        start = time.monotonic()
        try:
            result = self.tool_funcs[name](**args)
            status = "ok"
        except Exception as e:
            result = f"error: {e}"
            status = "error"
        duration_ms = (time.monotonic() - start) * 1000
        self.tracer.log("tool_call", tool=name, args=args, status=status, duration_ms=duration_ms)
        return str(result)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_loop.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add agent/loop.py tests/test_loop.py
git commit -m "feat: plan-act-observe agent loop with confirmation and tracing"
```

---

### Task 11: CLI entrypoint

**Files:**
- Create: `cli.py`

**Interfaces:**
- Consumes: `OllamaClient` (Task 7), `AgentLoop` (Task 10), `TOOL_FUNCS`, `REQUIRES_CONFIRMATION`, `validate_args`, `TOOL_SCHEMAS` (Task 4), `Tracer` (Task 5).
- Produces: a runnable REPL. No new interfaces consumed by later tasks.

- [ ] **Step 1: Write `cli.py`**

```python
# cli.py
import sys

from agent.llm import OllamaClient, LLMConnectionError
from agent.loop import AgentLoop
from agent.tools import TOOL_FUNCS, REQUIRES_CONFIRMATION, TOOL_SCHEMAS, validate_args
from agent.tracing import Tracer

SYSTEM_PROMPT = (
    "You are a helpful personal knowledge assistant. Use the rag_search tool "
    "to answer questions about the user's notes, calculator for arithmetic, "
    "and web_search for current events or facts not in the notes. Only call "
    "a tool when you need it."
)


def main():
    llm_client = OllamaClient()
    try:
        llm_client.check_available()
    except LLMConnectionError as e:
        print(f"Startup error: {e}")
        sys.exit(1)

    tracer = Tracer()
    loop = AgentLoop(
        llm_client=llm_client,
        tool_funcs=TOOL_FUNCS,
        requires_confirmation=REQUIRES_CONFIRMATION,
        validate_args_fn=validate_args,
        tracer=tracer,
        tool_schemas=TOOL_SCHEMAS,
    )
    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Demo agent ready. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue
        answer = loop.run_turn(user_input, history)
        print(answer)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Manual golden-path test (requires Ollama + Qdrant running, notes ingested per Task 8 Step 6)**

```bash
python cli.py
```
Try, in order:
1. `what does my note about qdrant say?` — expect a `rag_search` tool_call trace entry and an answer referencing the ingested note.
2. `what's 47 * 89` — expect a `calculator` tool_call trace entry and the correct answer (4183).
3. `search the web for the current version of python` — expect a confirmation prompt; answer `n` and confirm the agent responds sensibly to the decline; run again and answer `y` to see a real result.

Expected: `logs/<session>.jsonl` contains `llm_call`, `tool_call`, and (for the web search case) `user_confirm` events with plausible timings.

- [ ] **Step 3: Commit**

```bash
git add cli.py
git commit -m "feat: CLI REPL entrypoint wiring loop, tools, and tracing"
```

---

### Task 12: Evals

**Files:**
- Create: `evals/cases.yaml`
- Create: `evals/run_evals.py`

**Interfaces:**
- Consumes: `OllamaClient`, `AgentLoop`, `TOOL_FUNCS`, `REQUIRES_CONFIRMATION`, `validate_args`, `TOOL_SCHEMAS`, `Tracer` (same as `cli.py`, Task 11).
- Produces: a manually-run script printing a pass/fail table. No interfaces consumed elsewhere.

- [ ] **Step 1: Write `evals/cases.yaml`**

```yaml
# evals/cases.yaml
- name: rag_qdrant_fact
  input: "According to my notes, what is Qdrant?"
  check: contains_keyword
  keyword: "vector database"

- name: calculator_multiplication
  input: "What is 47 * 89?"
  check: tool_called
  tool: calculator

- name: calculator_correct_result
  input: "What is 12 + 8?"
  check: contains_keyword
  keyword: "20"

- name: rag_triggers_rag_search
  input: "Search my notes for anything about qdrant."
  check: tool_called
  tool: rag_search

- name: web_search_triggers_confirmation
  input: "Search the web for today's weather in Paris."
  check: tool_called
  tool: web_search
  auto_confirm: "n"

- name: web_search_decline_is_respected
  input: "Search the web for today's weather in Paris."
  check: contains_keyword
  keyword: "declin"
  auto_confirm: "n"

- name: division
  input: "What is 100 / 4?"
  check: contains_keyword
  keyword: "25"

- name: power
  input: "What is 2 to the power of 10?"
  check: contains_keyword
  keyword: "1024"

- name: unrelated_small_talk_no_tool
  input: "Say hello in one short sentence."
  check: no_tool_called

- name: rag_search_no_match_still_answers
  input: "According to my notes, what is the airspeed velocity of an unladen swallow?"
  check: contains_keyword
  keyword: "not"

- name: calculator_negative_numbers
  input: "What is -5 + 10?"
  check: contains_keyword
  keyword: "5"

- name: calculator_parentheses
  input: "What is (3 + 2) * 4?"
  check: contains_keyword
  keyword: "20"
```

- [ ] **Step 2: Write `evals/run_evals.py`**

```python
# evals/run_evals.py
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.llm import OllamaClient
from agent.loop import AgentLoop
from agent.tools import TOOL_FUNCS, REQUIRES_CONFIRMATION, TOOL_SCHEMAS, validate_args
from agent.tracing import Tracer

SYSTEM_PROMPT = (
    "You are a helpful personal knowledge assistant. Use the rag_search tool "
    "to answer questions about the user's notes, calculator for arithmetic, "
    "and web_search for current events or facts not in the notes."
)


def load_cases(path: str) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)


def run_case(case: dict, loop: AgentLoop, tracer: Tracer) -> tuple[bool, str]:
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    answer = loop.run_turn(case["input"], history)

    check = case["check"]
    if check == "contains_keyword":
        passed = case["keyword"].lower() in answer.lower()
        return passed, answer
    if check in ("tool_called", "no_tool_called"):
        tool_messages = [m for m in history if m["role"] == "tool"]
        called_any = len(tool_messages) > 0
        if check == "no_tool_called":
            return not called_any, answer
        return called_any, answer
    raise ValueError(f"unknown check type: {check}")


def main():
    cases = load_cases(str(Path(__file__).parent / "cases.yaml"))
    llm_client = OllamaClient()
    llm_client.check_available()

    results = []
    for case in cases:
        confirm_fn = None
        if "auto_confirm" in case:
            answer = case["auto_confirm"]
            confirm_fn = lambda name, args, _a=answer: _a.lower() == "y"

        tracer = Tracer(session_id=f"eval-{case['name']}")
        loop = AgentLoop(
            llm_client=llm_client,
            tool_funcs=TOOL_FUNCS,
            requires_confirmation=REQUIRES_CONFIRMATION,
            validate_args_fn=validate_args,
            tracer=tracer,
            tool_schemas=TOOL_SCHEMAS,
            **({"confirm_fn": confirm_fn} if confirm_fn else {}),
        )
        try:
            passed, answer = run_case(case, loop, tracer)
        except Exception as e:
            passed, answer = False, f"ERROR: {e}"
        results.append((case["name"], passed, answer))

    print(f"{'CASE':<35} {'RESULT':<6} ANSWER")
    for name, passed, answer in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name:<35} {status:<6} {answer[:80]}")

    total = len(results)
    passed_count = sum(1 for _, p, _ in results if p)
    print(f"\n{passed_count}/{total} passed")
    sys.exit(0 if passed_count == total else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Manual run against the live stack (requires Ollama + Qdrant up, notes ingested)**

```bash
python evals/run_evals.py
```
Expected: a pass/fail table printed, with most cases passing. Since `qwen2.5:0.5b` is a very small model, some tool-triggering cases may be flaky — that flakiness is itself a useful observation about small-model function-calling reliability; note which cases fail consistently vs. intermittently.

- [ ] **Step 4: Commit**

```bash
git add evals/cases.yaml evals/run_evals.py
git commit -m "feat: hand-written eval suite for agent behavior"
```

---

## Self-Review Notes

- **Spec coverage:** tokens/context (Task 3), embeddings (Task 8), prompting/structured output/function calling (Tasks 4, 7, 10), vector search/RAG (Task 8), APIs as tools (Tasks 8, 9), plan-act-observe loop (Task 10), memory/state (Task 3), human-in-the-loop (Task 6), evals (Task 12), logging/tracing (Task 5), cost/latency control (Tasks 3, 7, 10 — token budget, timeouts, max_tokens, max_iterations) are each covered by a task.
- **Type consistency checked:** `TOOL_FUNCS` (Task 4) is consumed with the same name in Tasks 10 and 11; `truncate_history` signature matches between Task 3 and its use in Task 10; `Tracer.log` signature matches across Tasks 5, 10, 11, 12; `AgentLoop.run_turn` signature matches between Task 10 and its callers in Tasks 11-12.
- **No placeholders remain** — every step has runnable code or exact shell commands.
