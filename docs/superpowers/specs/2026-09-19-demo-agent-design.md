# Demo Agent — Design Spec

Date: 2026-09-19

## Purpose

A small, hand-built agent used as a learning vehicle to build hands-on understanding of:

1. LLM mechanics: tokens, context window, embeddings, prompting, structured output / function calling.
2. Retrieval + tools: vector search, RAG, APIs as tools, streaming/tracing.
3. Agent patterns: plan-act-observe loop, memory/state, human-in-the-loop, evals, logging/tracing, cost/latency control.

The agent is a **personal knowledge assistant**: it answers questions using RAG over a local folder of notes/docs, and can also use a calculator and web search as tools. It is deliberately hand-rolled (no LangChain/LangGraph) so every mechanic is visible in plain Python rather than hidden behind framework abstractions.

## Stack

- Language: Python 3.11
- LLM: Ollama, model `qwen2.5:0.5b` (must support native tool-calling)
- Embeddings: Ollama, model `nomic-embed-text`
- Vector DB: Qdrant (via `docker-compose.yml`)
- Web search tool: DuckDuckGo (`duckduckgo-search` package, no API key)
- Interface: CLI REPL
- No framework for orchestration — hand-rolled agent loop

## Project layout

```
demo-agent/
├── agent/
│   ├── llm.py           # Ollama chat client wrapper, token/context budgeting, timeouts
│   ├── loop.py          # plan-act-observe loop: send messages, parse tool_calls, dispatch, append observations
│   ├── tools/
│   │   ├── __init__.py  # tool registry + JSON schemas for function-calling
│   │   ├── rag_search.py    # embed query (nomic-embed-text) -> Qdrant search -> return chunks
│   │   ├── web_search.py    # DuckDuckGo search
│   │   └── calculator.py    # safe arithmetic via ast, no eval()
│   ├── memory.py         # in-memory conversation history + truncation policy
│   ├── confirm.py        # human-in-the-loop: CLI y/n gate before risky tool calls
│   └── tracing.py        # structured JSONL logger: every LLM call, tool call, timing, tokens
├── ingest.py              # chunk + embed local docs -> upsert into Qdrant
├── cli.py                 # REPL entrypoint
├── evals/
│   ├── cases.yaml         # ~12 hand-written question -> expected tool/answer cases
│   └── run_evals.py       # runs cases through the live agent, checks pass/fail
├── tests/                 # pytest unit tests for deterministic pieces
├── docker-compose.yml     # Qdrant container
└── requirements.txt
```

Each module has one job: `llm.py` only talks to Ollama, `loop.py` only orchestrates, `tools/*` only execute, `tracing.py` only records. `loop.py` depends on `llm.py`, `tools/`, `memory.py`, `confirm.py`, `tracing.py`; nothing depends back on `loop.py`, so each piece is testable in isolation.

## Data flow: plan-act-observe loop

1. **User input** appended to the in-memory message list (`role: user`).
2. **Budget check** (`memory.py`): if the message list exceeds a token budget (char/4 heuristic, no extra tokenizer dependency), drop the oldest non-system turns before sending.
3. **LLM call** (`llm.py`): send messages + tool schemas to Ollama's `/api/chat` with `tools=[...]`, a max_tokens cap, and a wall-clock timeout (default 30s, configurable). This is the "plan" step — model answers directly or emits `tool_calls`.
4. **No tool_calls** → print the answer, log it, append to history, loop back to step 1.
5. **tool_calls present** ("act" step), for each call:
   - `rag_search` / `calculator` run immediately (safe, deterministic).
   - `web_search` goes through `confirm.py` first: prints the proposed call, waits for y/n; on "n", a synthetic tool result ("user declined") is fed back instead of running it.
   - Every dispatch (attempted or declined) is written to the trace log with timing.
6. **Observe**: tool outputs appended as `role: tool` messages.
7. Loop back to step 3, until the model responds with no tool_calls or a max-iterations guard (default 6) trips, to prevent infinite loops.

Trace log (`tracing.py`): one JSONL file per session under `logs/`, one line per event (`llm_call`, `tool_call`, `tool_result`, `user_confirm`), each with timestamp and duration.

## Error handling

- **Ollama unreachable / model not pulled**: caught at startup in `llm.py`, prints a clear message (`ollama pull qwen2.5:0.5b`, `ollama pull nomic-embed-text`, start the daemon) and exits — no silent retries.
- **Qdrant unreachable**: caught per-call inside `rag_search`, returns a tool error observation (`"error: vector store unavailable"`) back to the LLM instead of crashing the loop, so other tools still work.
- **Malformed/empty tool_call arguments** (small models like `qwen2.5:0.5b` are prone to this): validate args against the tool's JSON schema before dispatch; on failure, feed a `role: tool` error message back to the model instead of crashing, giving it a chance to retry with corrected args.
- **Calculator**: no `eval()`; parse with Python's `ast` module, only allow numeric literals and `+ - * / ** ( )`, reject anything else.
- **Max loop iterations** (default 6): if hit, stop and tell the user the agent gave up; logged as a distinct trace event.
- **LLM timeout**: per-call timeout; on timeout, treat as a failed call, log it, surface to the user rather than hanging the REPL.

## Testing & evals

- **Unit tests** (pytest), no live services required: `calculator.py` safe-eval edge cases (valid and rejected expressions), `memory.py` truncation logic at budget boundaries, tool-arg schema validation.
- **Ingest smoke test**: run `ingest.py` against a small fixture folder of 2-3 markdown files, assert the expected chunk count lands in a test Qdrant collection.
- **Evals** (`evals/cases.yaml` + `run_evals.py`), run manually against the live stack (Ollama + Qdrant up): ~12 hand-written cases —
  - RAG questions with an expected keyword/fact that must appear in the final answer.
  - Cases that should trigger a specific tool (e.g. "what's 47 * 89" → expects a `calculator` call), checked via the trace log rather than exact text.
  - A case designed to trigger the `web_search` confirm gate, run with auto-answer "n" piped in, asserting the agent respects the decline.
  `run_evals.py` prints a pass/fail table. Not wired into CI — run manually since it needs local services.
- **Manual REPL testing**: golden-path pass after implementation (a RAG question, a math question, a question needing web search), confirming trace log entries look sane.

## Out of scope (v1)

- Cross-session persistent memory/fact store (in-session history only for now).
- Streaming tokens to a UI (CLI only; deferred if a web UI is wanted later).
- API-key-based tool auth (DuckDuckGo needs none; deferred if a keyed API is added later).
- Any framework (LangChain/LangGraph) — intentionally hand-rolled for learning.
