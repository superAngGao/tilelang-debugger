"""Normalize supported source syntax without evaluating user expressions."""
import ast
from ..instrument import Unsupported


def normalized(source):
    """Normalize explicit T.If/Then/Else without evaluating its condition twice."""
    class Normalize(ast.NodeTransformer):
        def visit_With(self, node):
            node = self.generic_visit(node)
            if len(node.items) == 1 and isinstance(node.items[0].context_expr, ast.Call):
                call = node.items[0].context_expr
                if ast.unparse(call.func) == "T.If":
                    arms = {}
                    for child in node.body:
                        if not isinstance(child, ast.With) or len(child.items) != 1:
                            raise Unsupported("T.If requires explicit Then/Else bodies")
                        arm = child.items[0].context_expr
                        name = ast.unparse(arm.func) if isinstance(arm, ast.Call) else ""
                        if name not in {"T.Then", "T.Else"} or name in arms:
                            raise Unsupported("invalid T.If arm")
                        arms[name] = child.body
                    if len(call.args) != 1 or call.keywords or "T.Then" not in arms:
                        raise Unsupported("invalid T.If condition")
                    return ast.copy_location(ast.If(test=call.args[0], body=arms["T.Then"], orelse=arms.get("T.Else", [])), node)
            return node
    return Normalize().visit(ast.parse(source))
