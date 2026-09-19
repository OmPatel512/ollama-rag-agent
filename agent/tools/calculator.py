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

