from agent.confirm import confirm_tool_call

def test_yes_confirms():
    result = confirm_tool_call("web_search", {"query": "x"}, input_func= lambda _:"y")
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

