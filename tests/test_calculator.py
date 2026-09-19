import pytest
from agent.tools.calculator import evaluate

def test_addition():
    assert evaluate("2 + 3") == 5

def test_subtraction():
    assert evaluate("5 - 2") == 3

def test_multiplication():
    assert evaluate("4 * 3") == 12

def test_division():
    assert evaluate("10 / 2") == 5

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
