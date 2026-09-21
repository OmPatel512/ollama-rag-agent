def confirm_tool_call(tool_name: str, args: dict, input_func=input, print_func=print) -> bool:
    """Prompt the user to approve a tool call. Returns True only on an
    explicit 'y' answer; any other input (including empty) declines."""
    print_func(f"Agent wants to call `{tool_name}` with args: {args}")
    answer = input_func("Allow this call? [y/N]: ").strip().lower()
    return answer == "y"
