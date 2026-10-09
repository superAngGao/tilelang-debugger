"""Source-only observation insertion and normal Python package loading."""
import ast
from contextlib import contextmanager
import importlib.abc
import importlib.machinery
import linecache
from pathlib import Path
import re
import sys

from .instrument import Unsupported, digest, locate


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


@contextmanager
def source_loader(path, source):
    """Keep the real module origin/package; bypass pyc for the selected file only."""
    path = Path(path).resolve()
    previous_cache = dict(linecache.cache)
    previous_meta = list(sys.meta_path)
    state = {"loads": 0}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).resolve() == path:
            raise Unsupported("selected source was imported before the source hook")

    class Loader(importlib.machinery.SourceFileLoader):
        def get_source(self, fullname):
            return source

        def get_code(self, fullname):
            state["loads"] += 1
            return self.source_to_code(source, str(path))

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, package_path=None, target=None):
            spec = importlib.machinery.PathFinder.find_spec(fullname, package_path)
            if spec is not None and spec.origin and Path(spec.origin).resolve() == path:
                if not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
                    raise Unsupported("selected module needs a normal Python source loader")
                spec.loader = Loader(fullname, str(path))
                return spec
            return None

    linecache.cache[str(path)] = (len(source), None, source.splitlines(True), str(path))
    sys.meta_path.insert(0, Finder())
    try:
        yield state
    finally:
        sys.meta_path[:] = previous_meta
        linecache.cache.clear()
        linecache.cache.update(previous_cache)
        state["restored"] = sys.meta_path == previous_meta and linecache.cache == previous_cache
