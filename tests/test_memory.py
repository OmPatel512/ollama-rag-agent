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
        {"role": "system", "content": "s" * 60},
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
