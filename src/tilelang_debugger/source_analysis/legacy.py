"""Compatibility analysis for the existing whole-tile source path."""
import ast
import re
from ..instrument import Unsupported, digest, locate


def readonly_syntax(function, buffer):
    """Conservative global-input domain, not a general Python alias analysis."""
    calls = {"T.Kernel", "T.Parallel", "T.serial", "T.Serial", "range", "T.ceildiv",
             "T.if_then_else", "T.cast", "T.Tensor", "T.exp", "T.sqrt", "T.rsqrt",
             "T.sin", "T.cos", "T.abs", "T.min", "T.max", "T.And", "T.Or"}
    parents = {c: n for n in ast.walk(function) for c in ast.iter_child_nodes(n)}
    for n in ast.walk(function):
        if isinstance(n, ast.Call) and ast.unparse(n.func) not in calls:
            return False
        if isinstance(n, ast.Name) and n.id == buffer:
            p = parents.get(n)
            if not (isinstance(p, ast.Subscript) and p.value is n and isinstance(p.ctx, ast.Load)):
                return False
            if any(isinstance(x, ast.Slice) for x in ast.walk(p.slice)):
                return False
    return True


def prepare(source, config):
    if set(config) != {"source", "points"} or not isinstance(config["points"], list) or not 1 <= len(config["points"]) <= 8:
        raise Unsupported("config requires source and 1..8 points")
    result, seen = [], set()
    for point in config["points"]:
        if not isinstance(point, dict) or set(point) != {"id", "line", "when", "buffer", "block", "loops"}:
            raise Unsupported("point fields must be id,line,when,buffer,block,loops")
        p = dict(point)
        if not isinstance(p["id"], str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", p["id"]) or p["id"] in seen:
            raise Unsupported("point ids must be unique ASCII identifiers, length <=32")
        seen.add(p["id"])
        if type(p["line"]) is not int or p["when"] not in ("before", "after"):
            raise Unsupported("invalid source line/placement")
        if not isinstance(p["buffer"], str) or not p["buffer"].isidentifier():
            raise Unsupported("buffer must be an identifier")
        if not isinstance(p["block"], list) or len(p["block"]) != 3 or any(type(x) is not int or x < 0 for x in p["block"]):
            raise Unsupported("block must be three nonnegative integers")
        if not isinstance(p["loops"], list) or any(not isinstance(x, dict) or set(x) != {"line", "iteration"} for x in p["loops"]):
            raise Unsupported("invalid loop selections")
        tree, node, loops, values = locate(source, p)
        parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
        chain, cur = [], node
        while cur in parents:
            cur = parents[cur]
            chain.append(cur)
            if isinstance(cur, ast.FunctionDef):
                break
        if not chain or not isinstance(chain[-1], ast.FunctionDef):
            raise Unsupported("observation must be inside a kernel function")
        function = chain[-1]
        kernels = [n for n in chain if isinstance(n, ast.With) and any(
            isinstance(i.context_expr, ast.Call) and ast.unparse(i.context_expr.func) == "T.Kernel" for i in n.items)]
        if len(kernels) != 1:
            raise Unsupported("observation requires one enclosing T.Kernel")
        for n in chain[:-1]:
            if isinstance(n, (ast.If, ast.Try, ast.While, ast.Match)):
                raise Unsupported("observation must be in a full-CTA source scope, outside branches")
            if isinstance(n, ast.With) and n is not kernels[0]:
                raise Unsupported("nested context/conditional/warp scope is unsupported")
        for loop in loops:
            if any(isinstance(n, (ast.Break, ast.Continue, ast.Return)) for n in ast.walk(loop)):
                raise Unsupported("selected serial loops cannot exit early")
        for n in ast.walk(function):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                name = n.func.attr.lower()
                if name in {"ws", "pipelined", "gemm", "wgmma", "mma", "async_copy", "cp_async"} or "wgmma" in name:
                    raise Unsupported("asynchronous matrix/warp-specialized kernels require the reviewed engine")
        bounds = []
        for loop in loops:
            args = loop.iter.args
            bounds.append([ast.unparse(args[0]) if len(args) > 1 else "0",
                           ast.unparse(args[0] if len(args) == 1 else args[1]),
                           ast.unparse(args[2]) if len(args) == 3 else "1"])
        p.update(loop_vars=[l.target.id for l in loops], loop_values=values, loop_bounds=bounds,
                 global_readonly_syntax=readonly_syntax(function, p["buffer"]),
                 source_sha256=digest(source.encode()), source_path=config["source"], engine="source")
        result.append(p)
    return result
