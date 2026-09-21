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
            client = None
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.client = client or ollama.Client(host=host, timeout=timeout)

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        try:
            response = self.client.chat(
                model = self.model,
                messages = messages,
                tools = tools,
                options = {"num_predict": self.max_tokens},
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

