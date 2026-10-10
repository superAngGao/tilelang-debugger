"""Resolve upstream source points, run real debugger capture, then CPU analysis."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys

from common import cases, observations, save

HERE = Path(__file__).resolve().parent


def configuration(case, checkout):
    resolved = observations(case, checkout)["monitor"]
    points = []
    for p in resolved["points"]:
        loops = [dict(line=l["line"], iteration=p.get("iteration_by_variable", {}).get(l["target"], 1))
                 for l in p["enclosing_loops"]]
        points.append(dict(id=p["id"], line=p["line"], when=p["when"], buffer=p["buffer"], block=p["block"], loops=loops))
    if case["example"] == "rope":
        tree = ast.parse(Path(resolved["source"]).read_text())
        factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == case["factory"])
        loop = next(n for n in ast.walk(factory) if isinstance(n, ast.For) and isinstance(n.iter, ast.Call) and ast.unparse(n.iter.func) == "T.Parallel")
        points = [dict(id="input_x", line=loop.lineno, when="before", buffer="x", block=[1, 0, 0], loops=[])]
    return dict(source=resolved["source"], points=points)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tileops", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sanitizer", choices=("racecheck", "synccheck"))
    args = parser.parse_args()
    case = next(c for c in cases() if c["id"] == args.case)
    checkout = Path(args.tileops).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = configuration(case, checkout)
    save(output / "monitor.json", cfg)
    command = [sys.executable, "-m", "tilelang_debugger", "run", str(HERE / case["example"] / "run.py"),
               "--source", cfg["source"], "--monitor", str(output / "monitor.json"), "--output", str(output / "capture")]
    if args.sanitizer:
        command += ["--sanitizer", args.sanitizer]
    command += ["--", "--tileops", str(checkout), "--case", args.case, "--output", str(output / "driver-result")]
    save(output / "command.json", command)
    capture_process = subprocess.run(command)
    if capture_process.returncode not in (0, 2):
        capture_process.check_returncode()
    from tilelang_debugger.evidence import verify_capture
    verify_capture(output / "capture", args.sanitizer)
    provider = "import sys\nsys.path.insert(0, " + repr(str(HERE)) + ")\nfrom capture_reference import reference_case\n\ndef reference(inputs, points):\n    return reference_case(inputs, points, " + repr(case) + ")\n"
    (output / "reference.py").write_text(provider, encoding="utf-8")
    analysis_process = subprocess.run([sys.executable, "-m", "tilelang_debugger", "analyze", str(output / "capture"),
                                      "--reference", str(output / "reference.py"), "--output", str(output / "analysis")])
    if analysis_process.returncode not in (0, 2):
        analysis_process.check_returncode()
    analysis = json.loads((output / "analysis/analysis.json").read_text())
    if analysis.get("status") != "completed":
        raise ValueError("offline analysis did not complete")
    passed = capture_process.returncode == analysis_process.returncode == 0 and analysis.get("matched") is True
    save(output / "result.json", dict(status="passed" if passed else "numerical_mismatch", case=case, sanitizer=args.sanitizer,
                                      scope="source values; RoPE input only"))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
