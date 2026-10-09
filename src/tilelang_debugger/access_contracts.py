"""Reviewed, finite source/access domains. No automatic contract learning."""
import ast
import json
import re
from pathlib import Path
from .instrument import contracts, digest, Unsupported


def rule(line, operation, buffer, data, width, loops=(), elected=None, offset=None):
    return dict(line=line, operation=operation, buffer=buffer, data=data, width=width,
                source_loops=list(loops), elected=elected, shared_offset=offset)


RULES = {
    "gelu": [rule(87, "read", "x", "x", 8), rule(90, "write", "y", "y", 8)],
    "sum": [rule(35, "read", "x", "x", 1, [(33, 2)])],
    "sum_unpadded": [rule(42, "read", "x", "x", 4), rule(42, "write", "shared_buf", "shared_buf", 4),
                     rule(47, "read", "shared_buf", "shared_buf", 2, [(45, 2)])],
    "gemm": [rule(93, "transfer", "a", "a_desc", 1, [(70, 4), (77, 3)], [0, 32]),
             rule(98, "transfer", "b", "b_desc", 1, [(70, 4), (77, 3)], [0, 32])],
    "gqa": [rule(114, "transfer", "Q", "Q_desc", 1, elected=[256, 288], offset=0),
            rule(115, "transfer", "Q", "Q_desc", 1, elected=[256, 288], offset=4096),
            rule(120, "transfer", "K", "K_desc", 1, [(117, 3)], [256, 288]),
            rule(128, "transfer", "V", "V_desc", 1, [(117, 3)], [256, 288]),
            rule(293, "transfer", "Os", "O_desc", 1, elected=[0, 32], offset=0),
            rule(455, "transfer", "Os", "O_desc", 1, elected=[128, 160], offset=4096)],
}


def prepare(source, driver, config):
    if set(config) != {"source", "accesses"} or not isinstance(config["source"], str) or not config["source"].strip():
        raise Unsupported("access config requires a nonempty source path and accesses")
    matches = [(name, c) for name, c in contracts().items()
               if digest(source.encode()) == c["source_sha256"] and digest(driver) in c["driver_sha256"]]
    if len(matches) != 1:
        raise Unsupported("source/driver has no reviewed access contract")
    name, contract = matches[0]
    points = config["accesses"]
    if not isinstance(points, list) or not 1 <= len(points) <= 8:
        raise Unsupported("select 1..8 accesses")
    tree = ast.parse(source)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    seen, result = set(), []
    for p in points:
        if not isinstance(p, dict) or set(p) != {"id", "line", "operation", "buffer", "block", "loops"}:
            raise Unsupported("access fields: id,line,operation,buffer,block,loops")
        if not isinstance(p["id"], str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", p["id"]) or p["id"] in seen:
            raise Unsupported("unique ASCII access id required (max 32 characters)")
        seen.add(p["id"])
        if type(p["line"]) is not int:
            raise Unsupported("line must be an integer")
        rules = [r for r in RULES[name] if all(p[k] == r[k] for k in ("line", "operation", "buffer"))]
        if len(rules) != 1:
            raise Unsupported("source access not in reviewed mapping")
        r = rules[0]
        if not isinstance(p["block"], list) or len(p["block"]) != 3 or any(type(b) is not int or not 0 <= b < n for b, n in zip(p["block"], contract["grid"])):
            raise Unsupported("block outside reviewed grid")
        selections = p["loops"]
        if not isinstance(selections, list) or len(selections) != len(r["source_loops"]):
            raise Unsupported("select every enclosing serial loop")
        for s, (line, extent) in zip(selections, r["source_loops"]):
            if not isinstance(s, dict) or set(s) != {"line", "iteration"} or type(s["line"]) is not int or s["line"] != line or type(s["iteration"]) is not int or not 1 <= s["iteration"] <= extent:
                raise Unsupported("invalid serial loop selection")
        nodes = [n for n in ast.walk(tree) if isinstance(n, ast.stmt) and n.lineno == p["line"]]
        if len(nodes) != 1:
            raise Unsupported("source statement is ambiguous")
        current, serial = nodes[0], []
        while current in parents:
            current = parents[current]
            if isinstance(current, ast.For):
                f = ast.unparse(current.iter.func) if isinstance(current.iter, ast.Call) else ""
                if f in ("T.Parallel",):
                    continue
                if f not in ("range", "T.serial", "T.Serial"):
                    raise Unsupported("unreviewed enclosing loop")
                serial.append(current.lineno)
        if serial[::-1] != [x[0] for x in r["source_loops"]]:
            raise Unsupported("source scope differs from mapping")
        values = [s["iteration"] - 1 for s in selections]
        if name == "gemm" and values[1] != values[0] % 3:
            raise Unsupported("selected ring slot is inactive in this iteration")
        result.append(dict(r, **p, case=name, loop_values=values,
                           site=f"{name}_L{p['line']}_{p['operation']}_{p['buffer']}",
                           source_sha256=contract["source_sha256"]))
    sites = [p["site"] for p in result]
    if len(sites) != len(set(sites)):
        raise Unsupported("select each static site once per trace")
    return name, contract, result


def pins():
    return json.loads(Path(__file__).with_name("access-pins.json").read_text())
