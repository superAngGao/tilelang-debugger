"""Replay classification shared by indices, regions and path guards."""
import ast

PURE_CALLS = {'T.cast', 'T.min', 'T.max', 'T.And', 'T.Or', 'T.Not'} | {
    'T.' + dtype for dtype in ('bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32')}


def pure(node):
    if isinstance(node, ast.Call):
        return ast.unparse(node.func) in PURE_CALLS and not node.keywords and all(pure(a) for a in node.args)
    if isinstance(node, (ast.Name, ast.Constant)):
        return True
    if isinstance(node, ast.Attribute):
        return isinstance(node.value, ast.Name) and node.value.id == 'T' and 'T.' + node.attr in PURE_CALLS
    if isinstance(node, ast.BinOp):
        return isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.BitAnd, ast.BitOr,
                                    ast.BitXor, ast.LShift, ast.RShift)) and pure(node.left) and pure(node.right)
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, (ast.Not, ast.USub, ast.UAdd, ast.Invert)) and pure(node.operand)
    if isinstance(node, ast.BoolOp):
        return all(pure(v) for v in node.values)
    if isinstance(node, ast.Compare):
        return all(isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)) for op in node.ops) and all(pure(v) for v in [node.left, *node.comparators])
    if isinstance(node, ast.IfExp):
        return all(pure(v) for v in (node.test, node.body, node.orelse))
    return False


def operand(node):
    if isinstance(node, ast.Name):
        return True
    if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
        return False
    axes = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
    return all(all(v is None or pure(v) for v in (a.lower, a.upper, a.step)) if isinstance(a, ast.Slice) else pure(a) for a in axes)


def domain_safe(node):
    """Only statically safe denominators/counts may be speculated."""
    for child in ast.walk(node):
        if isinstance(child, ast.BinOp) and isinstance(child.op, (ast.FloorDiv, ast.Mod, ast.LShift, ast.RShift)):
            try:
                constant = ast.literal_eval(child.right)
            except (ValueError, TypeError):
                return False
            if type(constant) is not int:
                return False
            if isinstance(child.op, (ast.FloorDiv, ast.Mod)) and constant in (0, -1):
                return False
            if isinstance(child.op, (ast.LShift, ast.RShift)) and not 0 <= constant < 64:
                return False
    return True
