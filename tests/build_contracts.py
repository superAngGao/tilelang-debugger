"""Maintainer-only generator. Updating contracts requires independent review."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def generate():
    contracts = {}
    for name in ("gelu", "sum", "gemm", "gqa"):
        folder = ROOT / "examples" / name
        source = (folder / "kernel.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        points = []
        def add(node, buffer, shape, dtype="float32", when="after", leader=0, loops=None, fence=0):
            points.append(dict(line=node.lineno, when=when, buffer=buffer, shape=shape, dtype=dtype,
                               leader=leader, threads=128, block_vars={"gelu": ["bx", "0", "0"],
                               "sum": ["pid_m", "0", "0"], "gemm": ["bx", "by", "0"], "gqa": ["bx", "by", "bz"]}[name],
                               loop_extents=loops or [], fence_regs=fence))
        stmts = sorted((n for n in ast.walk(tree) if isinstance(n, ast.stmt)), key=lambda n: n.lineno)
        if name == "gelu":
            add(next(n for n in stmts if isinstance(n, ast.Expr) and ast.unparse(n).startswith("T.copy(x[")), "x_reg", [2048], "bfloat16")
            add(next(n for n in stmts if isinstance(n, ast.For) and ast.unparse(n.iter).startswith("T.Parallel")), "y_reg", [2048], "bfloat16")
        elif name == "sum":
            reduce = next(n for n in stmts if isinstance(n, ast.Expr) and ast.unparse(n).startswith("T.reduce_sum("))
            add(reduce, "x_f32", [2, 512], when="before")
            add(reduce, "acc", [2])
        elif name == "gemm":
            rings = [n for n in stmts if isinstance(n, ast.For) and ast.unparse(n.iter) == "range(num_stages)"]
            add(rings[-1], "c_local", [128, 128], leader=128, loops=[4], fence=128)
        else:
            groups = [n for n in stmts if isinstance(n, ast.With) and any(ast.unparse(x.context_expr).startswith("T.ws(") for x in n.items)]
            for i, group in enumerate(groups):
                wait = next(n for n in group.body if isinstance(n, ast.Expr) and ast.unparse(n) == "T.wait_wgmma(0)")
                add(wait, "acc_s", [64, 128], leader=i*128, fence=64)
        contract = dict(source_sha256=sha(source.encode()), driver_sha256=[sha((folder / "run.py").read_bytes())], allow_auto_cta_change=name in ("gelu", "sum"),
                        cta_threads={"gelu": 128, "sum": 128, "gemm": 256, "gqa": 384}[name],
                        grid={"gelu": [2,1,1], "sum": [2,1,1], "gemm": [2,1,1], "gqa": [2,2,1]}[name], points=points)
        contracts[name] = contract
        selected = []
        for i, p in enumerate(points):
            loops = []
            if name == "gemm":
                outer = next(n for n in stmts if isinstance(n, ast.For) and ast.unparse(n.iter) == "T.serial(k_iters)" and n.lineno < p["line"] < n.end_lineno)
                loops = [dict(line=outer.lineno, iteration=2)]
            selected.append(dict(id=f"{name}_{i}", line=p["line"], when=p["when"], buffer=p["buffer"], block=[1,0,0], loops=loops))
        (folder / "monitor.json").write_text(json.dumps(dict(source="kernel.py", points=selected), indent=2)+"\n")
        if name == "sum":
            driver = (folder / "run.py").read_text().replace("257", "256")
            (folder / "run_unpadded.py").write_text(driver, encoding="utf-8", newline="\n")
            unpadded = copy.deepcopy(contract)
            unpadded["driver_sha256"] = [sha((folder / "run_unpadded.py").read_bytes())]
            unpadded["points"][0]["shape"] = [2, 256]
            contracts["sum_unpadded"] = unpadded
    layout_file = ROOT / "tests/layout-contracts.json"
    if layout_file.exists():
        layouts = json.loads(layout_file.read_text())
        for name, contract in contracts.items():
            prefix = "sum" if name == "sum_unpadded" else name
            for i, point in enumerate(contract["points"]):
                point["layout_sha256"] = [layouts[name]["layouts"][f"{prefix}_{i}"]]
    (ROOT / "src/tilelang_debugger/contracts.json").write_text(json.dumps(contracts, indent=2)+"\n")


if __name__ == "__main__":
    generate()
