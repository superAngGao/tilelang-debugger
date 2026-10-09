"""Apply an already prepared sample observation plan to source code."""
import ast
from ..source_analysis.normalize import normalized


def inject(source, points):
    tree = normalized(source)
    prefix = points[0]["prefix"]
    def call(name, args):
        return ast.parse(f"{prefix}_{name}({args})").body[0]
    def coords(p, depth=None):
        names = [a for s in p["scopes"] if s["kind"] != "branch" for a in s["aliases"]]
        names = names if depth is None else names[:depth]
        return "(" + ",".join(names) + ("," if names else "") + ")"

    class Insert(ast.NodeTransformer):
        def visit(self, node):
            out = super().visit(node)
            if isinstance(node, ast.With):
                roots = [p for p in points if p["kernel_line"] == node.lineno]
                for p in reversed(roots):
                    bounds = ["(" + ",".join(b) + ")" for s in p["scopes"] if s["kind"] != "branch" for b in s["bounds"]]
                    out.body.insert(0, call("root", f"{p['id']!r}, [{','.join(bounds)}]"))
            if isinstance(node, ast.For):
                matching = [s for p in points for s in p["scopes"] if s["kind"] != "branch" and s["line"] == node.lineno]
                if matching:
                    out.body[:0] = [ast.parse(f"{a} = {v}").body[0] for a, v in zip(matching[0]["aliases"], matching[0]["names"])]
            if isinstance(node, ast.If):
                for p in points:
                    for i, s in enumerate(p["scopes"]):
                        if s["kind"] == "branch" and s["line"] == node.lineno:
                            for arm, body in ((1, out.body), (0, out.orelse)):
                                body.insert(0, call("branch", f"{p['id']!r}, {i}, {arm}, {coords(p, s['depth'])}"))
            if isinstance(node, ast.stmt):
                ps = [p for p in points if p["line"] == node.lineno]
                if ps:
                    def data(p):
                        return call("value", f"{p['buffer']}, {p['id']!r}, {coords(p)}")
                    return [*[data(p) for p in ps if p["when"] == "before"], out, *[data(p) for p in ps if p["when"] == "after"]]
            return out
    tree = Insert().visit(tree)
    pos = int(bool(tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str)))
    while pos < len(tree.body) and isinstance(tree.body[pos], ast.ImportFrom) and tree.body[pos].module == "__future__":
        pos += 1
    tree.body.insert(pos, ast.ImportFrom(module="tilelang_debugger.sample_monitor", names=[ast.alias(name=f"observe_{n}", asname=f"{prefix}_{n}") for n in ("root", "branch", "value")], level=0))
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"
