"""Source-only recursive scope paths for version-two observations."""
import ast
import copy
import re

from .instrument import Unsupported, digest


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


def prepare(source, config):
    if set(config) != {"schema", "source", "points"} or config["schema"] != 2:
        raise Unsupported("samples config requires schema=2, source, points")
    if not isinstance(config["points"], list) or not 1 <= len(config["points"]) <= 8:
        raise Unsupported("select 1..8 observation points")
    tree = normalized(source)
    parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    prefix = "__tldbg2"
    while any(n.startswith(prefix) for n in used):
        prefix += "_"
    result, ids = [], set()
    for original in config["points"]:
        fields = {"id", "line", "when", "buffer", "block", "loops"}
        if not isinstance(original, dict) or not fields <= original.keys() or original.keys() - fields - {"thread"}:
            raise Unsupported("invalid sample point fields")
        p = copy.deepcopy(original)
        if not isinstance(p["id"], str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", p["id"]) or p["id"] in ids:
            raise Unsupported("point ids must be unique ASCII identifiers, length <=32")
        ids.add(p["id"])
        if type(p["line"]) is not int or p["when"] not in {"before", "after"}:
            raise Unsupported("invalid source position")
        if not isinstance(p["buffer"], str):
            raise Unsupported("observation requires a variable or scalar element")
        try:
            expression = ast.parse(p["buffer"], mode="eval").body
        except SyntaxError as exc:
            raise Unsupported("invalid observation expression") from exc
        allowed = (ast.Name, ast.Subscript, ast.Tuple, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Load,
                   ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.USub, ast.UAdd)
        if not isinstance(expression, (ast.Name, ast.Subscript)) or any(not isinstance(n, allowed) for n in ast.walk(expression)):
            raise Unsupported("observe a variable or scalar element with pure indices")
        if isinstance(expression, ast.Subscript) and (not isinstance(expression.value, ast.Name) or any(isinstance(n, ast.Subscript) for n in ast.walk(expression.slice))):
            raise Unsupported("element indices cannot read other buffers")
        if not isinstance(p["block"], list) or len(p["block"]) != 3 or any(type(v) is not int or v < 0 for v in p["block"]):
            raise Unsupported("block requires three nonnegative integers")
        if p.get("thread") is not None and (type(p["thread"]) is not int or p["thread"] < 0):
            raise Unsupported("thread must be null or a nonnegative integer")
        candidates = [n for n in ast.walk(tree) if isinstance(n, ast.stmt) and n.lineno == p["line"]]
        if len(candidates) != 1:
            raise Unsupported("select one unambiguous statement start line")
        target = candidates[0]
        if not isinstance(target, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr)):
            raise Unsupported("select a computation statement, not a control header")
        if isinstance(expression, ast.Subscript):
            destinations = target.targets if isinstance(target, ast.Assign) else [target.target] if isinstance(target, (ast.AnnAssign, ast.AugAssign)) else []
            if p["when"] != "after" or not any(ast.unparse(n) == ast.unparse(expression) for n in destinations):
                raise Unsupported("element observation must follow the same element's source write")
        chain, child = [], target
        while child in parents:
            parent = parents[child]
            if isinstance(parent, ast.FunctionDef):
                break
            chain.append((parent, child))
            child = parent
        chain.reverse()
        kernels = [n for n, _ in chain if isinstance(n, ast.With) and len(n.items) == 1 and isinstance(n.items[0].context_expr, ast.Call) and ast.unparse(n.items[0].context_expr.func) == "T.Kernel"]
        if len(kernels) != 1:
            raise Unsupported("point needs one enclosing T.Kernel")
        kernel = kernels[0]
        kernel_writes = {n.id for n in ast.walk(kernel) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        if any(isinstance(n, (ast.Break, ast.Continue, ast.Return)) for n in ast.walk(kernel)):
            raise Unsupported("early exits need a visit/exit protocol")
        scopes, depth = [], 0
        for node, child in chain[chain.index(next(pair for pair in chain if pair[0] is kernel)) + 1:]:
            if isinstance(node, ast.For):
                call = node.iter
                name = ast.unparse(call.func) if isinstance(call, ast.Call) else ""
                kind = "parallel" if name == "T.Parallel" else "serial"
                if name not in {"T.Parallel", "T.serial", "T.Serial", "T.unroll", "T.Unroll", "range"}:
                    raise Unsupported(f"loop strategy not validated: {name}")
                pure_bound = (ast.Name, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Load,
                              ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.USub, ast.UAdd)
                for arg in call.args:
                    if any(not isinstance(n, pure_bound) or isinstance(n, ast.Name) and n.id in kernel_writes for n in ast.walk(arg)):
                        raise Unsupported("loop bounds require pure static integers/closure constants")
                names = [node.target.id] if isinstance(node.target, ast.Name) else [n.id for n in node.target.elts if isinstance(n, ast.Name)] if isinstance(node.target, ast.Tuple) else []
                if not names or node.orelse:
                    raise Unsupported("unsupported loop target/else")
                if kind == "parallel":
                    if len(names) != len(call.args):
                        raise Unsupported("Parallel coordinate arity differs")
                    bounds = [["0", ast.unparse(a), "1"] for a in call.args]
                else:
                    if len(names) != 1 or not 1 <= len(call.args) <= 3:
                        raise Unsupported("counted loop needs 1..3 arguments")
                    args = [ast.unparse(a) for a in call.args]
                    bounds = [["0", args[0], "1"] if len(args) == 1 else [args[0], args[1], args[2] if len(args) == 3 else "1"]]
                aliases = [f"{prefix}_c{node.lineno}_{i}" for i in range(len(names))]
                scopes.append(dict(kind=kind, line=node.lineno, names=names, aliases=aliases, bounds=bounds, depth=depth))
                depth += len(names)
            elif isinstance(node, ast.If):
                scopes.append(dict(kind="branch", line=node.lineno, arm=1 if child in node.body else 0, depth=depth))
            elif isinstance(node, ast.stmt):
                raise Unsupported(f"unsupported enclosing scope: {type(node).__name__}")
        if depth + 7 > 32:
            raise Unsupported("coordinate arity exceeds CUDA printf argument budget")
        selections = p["loops"]
        if not isinstance(selections, list):
            raise Unsupported("loops must be a list")
        selected = set()
        for selection in selections:
            if not isinstance(selection, dict) or type(selection.get("line")) is not int or selection["line"] in selected:
                raise Unsupported("invalid/duplicate loop selection")
            selected.add(selection["line"])
            matches = [s for s in scopes if s["kind"] != "branch" and s["line"] == selection["line"]]
            if len(matches) != 1:
                raise Unsupported("selected loop must enclose the point")
            scope = matches[0]
            key = "coordinates" if scope["kind"] == "parallel" else "iteration"
            if set(selection) != {"line", key}:
                raise Unsupported(f"loop selection requires {key}")
            value = selection[key]
            if key == "iteration" and (type(value) is not int or value < 1):
                raise Unsupported("iteration is 1-based")
            if key == "coordinates" and (not isinstance(value, list) or len(value) != len(scope["names"]) or any(type(v) is not int for v in value)):
                raise Unsupported("invalid Parallel coordinates")
            scope["selection"] = value
        p.update(schema=2, engine="samples", scopes=scopes, kernel_line=kernel.lineno, prefix=prefix,
                 source_path=config["source"], source_sha256=digest(source.encode()), thread=p.get("thread"))
        result.append(p)
    return result


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
