import json
from agent.tracing import Tracer

def test_log_write_one_json_line(tmp_path):
    tracer = Tracer(log_dir=tmp_path, session_id="test_session")
    tracer.log("llm_call", duration_ms=42, tokens=10)

    log_file = tmp_path / "test_session.jsonl"
    assert log_file.exists()
    lines = log_file.read_text().strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["event"] == "llm_call"
    assert record["duration_ms"] == 42
    assert record["tokens"] == 10
    assert 'ts' in record


def test_log_append_multiple_events(tmp_path):
    tracer = Tracer(log_dir=tmp_path, session_id="test_session")
    tracer.log("llm_call", duration_ms=42, tokens=10)
    tracer.log("tool_call", tool="calculator", duration_ms=5)

    log_file = tmp_path / "test_session.jsonl"
    assert log_file.exists()
    lines = log_file.read_text().strip().splitlines()
    assert len(lines) == 2

    record1 = json.loads(lines[0])
    assert record1["event"] == "llm_call"
    assert record1["duration_ms"] == 42
    assert record1["tokens"] == 10

    record2 = json.loads(lines[1])
    assert record2["event"] == "tool_call"
    assert record2["tool"] == "calculator"
    assert record2["duration_ms"] == 5

def test_default_session_id_is_generated():
    tracer = Tracer(log_dir="logs")
    assert tracer.session_id

