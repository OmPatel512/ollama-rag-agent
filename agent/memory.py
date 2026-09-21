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
