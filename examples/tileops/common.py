"""External test fixtures, not a product kernel adapter or admission whitelist."""
import argparse
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest():
    return json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))


def cases():
    return [dict(item, dtype=dtype, id=f"{item['id']}-{dtype}")
            for item in manifest()["cases"] for dtype in manifest()["dtypes"]]


def locate(source, point):
    """Resolve example anchors only. No lowering interpretation or code changes."""
    tree = ast.parse(source)
    scopes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == point["factory"]]
    if len(scopes) != 1:
        raise ValueError("factory is missing or ambiguous")
    scope = scopes[0]
    if point.get("branch"):
        branch = ast.dump(ast.parse(point["branch"], mode="eval").body)
        matches = [n for n in ast.walk(scope) if isinstance(n, ast.If) and ast.dump(n.test) == branch]
        if len(matches) != 1:
            raise ValueError("factory branch is missing or ambiguous")
        scope = ast.Module(body=matches[0].body, type_ignores=[])
    anchor = ast.parse(point["statement"]).body
    if len(anchor) != 1:
        raise ValueError("anchor must be one statement")
    matches = [n for n in ast.walk(scope) if isinstance(n, ast.stmt) and ast.dump(n) == ast.dump(anchor[0])]
    if len(matches) != 1:
        raise ValueError(f"anchor must uniquely identify a statement: {point['id']} ({len(matches)} matches)")
    node = matches[0]
    parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
    loops, cur = [], node
    while cur in parents:
        cur = parents[cur]
        if isinstance(cur, ast.For):
            loops.append(dict(line=cur.lineno, target=ast.unparse(cur.target), iterator=ast.unparse(cur.iter)))
    return dict(point, line=node.lineno, end_line=node.end_lineno, enclosing_loops=loops[::-1])


def observations(case, checkout):
    result = {}
    for kind in ("monitor", "access"):
        config = json.loads((HERE / case["example"] / f"{kind}.json").read_text())
        path = checkout / config["source"]
        source = path.read_text(encoding="utf-8")
        points = [locate(source, p) for p in config["points"] if p["factory"] == case["factory"]]
        result[kind] = dict(schema=config["schema"], source=str(path), sha256=sha(path), points=points,
                            status="planned_not_captured", note=config.get("note"))
    return result


def load_upstream(checkout, module_name):
    checkout = Path(checkout).resolve()
    source = checkout / "src"
    if not (source / "tileops" / "__init__.py").is_file():
        raise ValueError("--tileops must name a TileOPs checkout with src/tileops")
    for name, module in tuple(sys.modules.items()):
        if name == "tileops" or name.startswith("tileops."):
            path = getattr(module, "__file__", None)
            if path is None or not Path(path).resolve().is_relative_to(source):
                raise ValueError(f"another TileOPs package is already imported: {name}: {path}")
    sys.path.insert(0, str(source))
    module = importlib.import_module(module_name)
    # Also check transitive imports; the actual source, not pip metadata, is authoritative.
    imported = {}
    for name, obj in tuple(sys.modules.items()):
        if name == "tileops" or name.startswith("tileops."):
            path = Path(obj.__file__).resolve()
            if not path.is_relative_to(source):
                raise ValueError(f"import escaped specified checkout: {name}: {path}")
            imported[str(path.relative_to(checkout))] = sha(path)
    def git(*args):
        return subprocess.run(["git", "-C", str(checkout), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    provenance = dict(checkout=str(checkout), commit=git("rev-parse", "HEAD"),
                      status=git("status", "--porcelain"), module=module_name,
                      source=str(Path(module.__file__).resolve()), imported_sources=imported,
                      reference_commit=manifest()["tileops_commit"])
    provenance["matches_reference_commit"] = provenance["commit"] == provenance["reference_commit"]
    return module, provenance


def export_kernel(kernel, folder):
    import tvm
    artifact = kernel.artifact
    if artifact is None or artifact.device_mod is None:
        raise ValueError("compiled IR unavailable; run with TILELANG_DISABLE_CACHE=1")
    for name, obj in (("frontend", kernel.prim_func), ("device", artifact.device_mod)):
        (folder / f"{name}.py").write_text(obj.script(), encoding="utf-8")
        (folder / f"{name}.json").write_text(tvm.ir.save_json(obj), encoding="utf-8")
    (folder / "kernel.cu").write_text(kernel.get_kernel_source(), encoding="utf-8")
    save(folder / "compile.json", dict(target=str(kernel.target), backend=kernel.execution_backend,
         pass_configs={str(k): str(v) for k, v in kernel.pass_configs.items()},
         compile_flags=list(kernel.compile_flags or [])))


def run_example(example, build, reference):
    parser = argparse.ArgumentParser(description="Run an unmodified upstream TileOPs numerical baseline")
    parser.add_argument("--tileops", required=True)
    parser.add_argument("--case", default=next(c["id"] for c in cases() if c["example"] == example))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    candidates = [c for c in cases() if c["id"] == args.case and c["example"] == example]
    if len(candidates) != 1:
        parser.error("unknown case for this example")
    case = candidates[0]
    capture_folder = Path(os.environ["TLDBG_OUTPUT"]) if os.environ.get("TLDBG_OUTPUT") else None
    folder = capture_folder / "example" if capture_folder else Path(args.output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    save(folder / "result.json", dict(status="running", case=case))
    try:
        os.environ["TILELANG_DISABLE_CACHE"] = "1"
        import torch
        import tilelang
        from tilelang_debugger.numerics import compare, tensor_data
        if not torch.cuda.is_available() or "H200" not in torch.cuda.get_device_name(0):
            raise RuntimeError("these validation fixtures require NVIDIA H200")
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        torch.backends.cuda.matmul.allow_tf32 = False
        upstream, origin = load_upstream(args.tileops, case["module"])
        origin["driver_sha256"] = sha(HERE / example / "run.py")
        origin["reference_sha256"] = sha(HERE / example / "reference.py")
        save(folder / "provenance.json", origin)
        save(folder / "observations.json", observations(case, Path(args.tileops).resolve()))
        save(folder / "environment.json", dict(python=sys.version, torch=torch.__version__,
             tilelang=tilelang.__version__, tilelang_path=tilelang.__file__, cuda=torch.version.cuda,
             gpu=torch.cuda.get_device_name(0), cache_disabled=True, seed=1234))
        kernel, inputs = build(upstream, case)
        export_kernel(kernel, folder)
        host_inputs = tuple(x.detach().cpu().clone() for x in inputs)
        expected = reference(host_inputs, case)
        if expected.device.type != "cpu":
            raise ValueError("reference must run on CPU")
        actual = kernel(*inputs).detach().cpu()
        torch.cuda.synchronize()
        torch.save(dict(inputs=host_inputs, output=actual, expected=expected, case=case), folder / "tensors.pt")
        atol, rtol = manifest()["tolerances"][case["dtype"]]
        comparison, rows = compare(tensor_data(actual), tensor_data(expected), atol, rtol)
        # Check padding separately with zero tolerance; approximate comparisons cannot hide nonzero padding.
        padding_ok = True
        if example in ("softmax", "rms_norm"):
            padding_ok = bool(torch.all(actual[:, case["N"]:] == 0))
        save(folder / "comparison.json", comparison)
        (folder / "elements.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
        worker_mode = json.loads((capture_folder / "request.json").read_text())["mode"] if capture_folder else None
        result = dict(status="passed" if comparison["passed"] and padding_ok else "failed", case=case,
                      numerical_status="passed" if comparison["passed"] else "failed", padding_zero=padding_ok,
                      debugger_status="worker_completed_pending_capture_validation" if capture_folder else "not_run",
                      instrumented=worker_mode == "instrumented", worker_mode=worker_mode)
        save(folder / "result.json", result)
        if capture_folder:
            save(capture_folder / "reference.json", dict(passed=result["status"] == "passed", comparison=comparison, padding_zero=padding_ok))
        save(folder / "files.json", {p.name: sha(p) for p in folder.iterdir() if p.is_file() and p.name != "files.json"})
        print(json.dumps(result), flush=True)
        return 0 if capture_folder or result["status"] == "passed" else 1
    except Exception as exc:
        save(folder / "result.json", dict(status="failed", case=case, error=f"{type(exc).__name__}: {exc}"))
        raise
