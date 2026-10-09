"""Source positions, lexical bindings and recursive observation scope paths."""
import ast
import copy
import re
from ..instrument import Unsupported, digest
from .normalize import normalized


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
