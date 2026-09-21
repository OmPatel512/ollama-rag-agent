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


