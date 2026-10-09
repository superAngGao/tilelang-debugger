"""Locate original statements; never infer source positions from lowered code."""
import ast
import hashlib
import json
import re
from pathlib import Path


class Unsupported(ValueError):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def contracts():
    return json.loads(Path(__file__).with_name("contracts.json").read_text())


def locate(source, point):
    tree = ast.parse(source)
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.stmt)
               and n.lineno == point["line"]]
    if len(matches) != 1:
        raise Unsupported("line must identify exactly one statement start")
    node = matches[0]
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    chain, current = [], node
    while current in parents:
        current = parents[current]
        chain.append(current)
    loops = []
    for ancestor in reversed(chain):
        if isinstance(ancestor, (ast.While, ast.AsyncFor)):
            raise Unsupported("while/async loop is unsupported")
        if isinstance(ancestor, ast.For):
            call = ancestor.iter
            if not isinstance(call, ast.Call) or ast.unparse(call.func) not in ("range", "T.serial", "T.Serial"):
                raise Unsupported("only enclosing serial loops are supported")
            if not isinstance(ancestor.target, ast.Name) or call.keywords or not 1 <= len(call.args) <= 3:
                raise Unsupported("unsupported serial loop form")
            loops.append(ancestor)
    selected = point["loops"]
    if len(loops) > 2 or [x.lineno for x in loops] != [x["line"] for x in selected]:
        raise Unsupported("select every enclosing serial loop, outermost first (maximum 2)")
    values = []
    for loop, selection in zip(loops, selected):
        args = loop.iter.args
        start = ast.Constant(0) if len(args) == 1 else args[0]
        step = ast.Constant(1) if len(args) < 3 else args[2]
        if not isinstance(start, ast.Constant) or type(start.value) is not int or not isinstance(step, ast.Constant) or type(step.value) is not int or step.value == 0:
            raise Unsupported("initial release requires constant serial start/step")
        ordinal = selection["iteration"]
        if type(ordinal) is not int or ordinal < 1:
            raise Unsupported("iteration is a positive, one-based ordinal")
        values.append(start.value + (ordinal - 1) * step.value)
    return tree, node, loops, values


def prepare(source, config, driver_bytes=None):
    if set(config) != {"source", "points"} or not isinstance(config["source"], str) or not config["source"].strip():
        raise Unsupported("config requires a nonempty source path and points")
    if not isinstance(config["points"], list) or not 1 <= len(config["points"]) <= 8:
        raise Unsupported("select 1..8 observation points")
    source_hash = digest(source.encode("utf-8"))
    candidates = [c for c in contracts().values() if c["source_sha256"] == source_hash]
    contract = next((c for c in candidates if driver_bytes is None or digest(driver_bytes) in c["driver_sha256"]), None)
    if contract is None:
        raise Unsupported("source/driver has no reviewed safety contract; modified kernels require review")
    if driver_bytes is not None and digest(driver_bytes) not in contract["driver_sha256"]:
        raise Unsupported("driver parameters are outside the reviewed contract")
    result, seen = [], set()
    for p in config["points"]:
        if set(p) != {"id", "line", "when", "buffer", "block", "loops"}:
            raise Unsupported("point fields must be id,line,when,buffer,block,loops")
        if not isinstance(p["id"], str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", p["id"]) or p["id"] in seen:
            raise Unsupported("point ids must be unique ASCII identifiers, length <=32")
        seen.add(p["id"])
        if type(p["line"]) is not int or p["when"] not in ("before", "after"):
            raise Unsupported("invalid source line or placement")
        if not isinstance(p["buffer"], str) or not p["buffer"].isidentifier():
            raise Unsupported("buffer must be an identifier")
        if not isinstance(p["block"], list) or len(p["block"]) != 3 or any(type(x) is not int or x < 0 for x in p["block"]):
            raise Unsupported("block must contain three nonnegative integers")
        if not isinstance(p["loops"], list) or any(not isinstance(x, dict) or set(x) != {"line", "iteration"} for x in p["loops"]):
            raise Unsupported("invalid loop selections")
        rule = next((r for r in contract["points"] if all(r[k] == p[k] for k in ("line", "when", "buffer"))), None)
        if rule is None:
            raise Unsupported("statement/buffer is not a reviewed complete, ready observation point")
        _, _, loops, values = locate(source, p)
        if any(b >= size for b, size in zip(p["block"], contract["grid"])):
            raise Unsupported("block lies outside the reviewed launch grid")
        if any(x["iteration"] > extent for x, extent in zip(p["loops"], rule.get("loop_extents", []))):
            raise Unsupported("iteration is outside the reviewed loop")
        merged = dict(p, **{k: v for k, v in rule.items() if k not in p})
        merged.update(loop_vars=[l.target.id for l in loops], loop_values=values,
                      source_sha256=source_hash)
        result.append(merged)
    if sum(__import__("math").prod(p["shape"]) for p in result) > 65536:
        raise Unsupported("capture exceeds 65536-element budget")
    return contract, result


def inject(source, points):
    tree = ast.parse(source)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    helper = "__tldbg_capture"
    while helper in used:
        helper += "_"
    # Lowering clones Var objects. Give selected allocations unique names in
    # their actual TileLang lexical scopes, so the two GQA acc_s buffers cannot
    # be confused after cloning. All uses in the same scope are renamed.
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    scopes = {}
    for p in points:
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.stmt) and n.lineno == p["line"])
        current = node
        while True:
            current = parents[current]
            if isinstance(current, ast.FunctionDef) or (isinstance(current, ast.With) and any(ast.unparse(x.context_expr).startswith("T.ws(") for x in current.items)):
                break
        key = (current.lineno, p["buffer"])
        if key not in scopes:
            name = f"tldbg_buffer_{len(scopes)}"
            while name in used:
                name += "_"
            used.add(name)
            scopes[key] = (current, name)
        p["instrumented_buffer"] = scopes[key][1]
    for (_, old), (scope, new) in scopes.items():
        allocations = [n for n in ast.walk(scope) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == old for t in n.targets)
                       and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "T.alloc_fragment"]
        if len(allocations) != 1:
            raise Unsupported("selected buffer allocation is not unique in its reviewed lexical scope")
        for node in ast.walk(scope):
            if isinstance(node, ast.Name) and node.id == old:
                node.id = new
    by_line = {}
    for p in points:
        by_line.setdefault(p["line"], []).append(p)

    def inserted(p):
        block = ", ".join(p["block_vars"])
        loops = ", ".join(p["loop_vars"])
        return ast.parse(f"{helper}({p['instrumented_buffer']}, {p['id']!r}, ({block},), ({loops}{',' if loops else ''}))").body[0]

    class Injector(ast.NodeTransformer):
        def visit(self, node):
            # Recurse first, preserving original line numbers during all matches.
            out = super().visit(node)
            if isinstance(node, ast.stmt) and node.lineno in by_line:
                selected = by_line[node.lineno]
                return [*[inserted(p) for p in selected if p["when"] == "before"], out,
                        *[inserted(p) for p in selected if p["when"] == "after"]]
            return out
    tree = Injector().visit(tree)
    imp = ast.ImportFrom(module="tilelang_debugger.monitor", names=[ast.alias(name="capture", asname=helper)], level=0)
    pos = 1 if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) else 0
    while pos < len(tree.body) and isinstance(tree.body[pos], ast.ImportFrom) and tree.body[pos].module == "__future__":
        pos += 1
    tree.body.insert(pos, imp)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"
