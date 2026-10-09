"""Insert existing whole-tile calls from a prepared source plan."""
import ast


def inject(source, points):
    tree = ast.parse(source)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    helper = "__tldbg_source_capture"
    while helper in names:
        helper += "_"
    by_line = {}
    for p in points:
        by_line.setdefault(p["line"], []).append(p)

    def call(p):
        loops = "(" + ",".join(p["loop_vars"]) + ("," if p["loop_vars"] else "") + ")"
        bounds = "[" + ",".join("(" + ",".join(b) + ")" for b in p["loop_bounds"]) + "]"
        return ast.parse(f"{helper}({p['buffer']}, {p['id']!r}, {loops}, {bounds})").body[0]

    class Insert(ast.NodeTransformer):
        def visit(self, node):
            out = super().visit(node)
            if isinstance(node, ast.stmt) and node.lineno in by_line:
                ps = by_line[node.lineno]
                return [*[call(p) for p in ps if p["when"] == "before"], out,
                        *[call(p) for p in ps if p["when"] == "after"]]
            return out
    tree = Insert().visit(tree)
    pos = 1 if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str) else 0
    while pos < len(tree.body) and isinstance(tree.body[pos], ast.ImportFrom) and tree.body[pos].module == "__future__":
        pos += 1
    tree.body.insert(pos, ast.ImportFrom(module="tilelang_debugger.monitor", names=[ast.alias(name="capture_source", asname=helper)], level=0))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"
