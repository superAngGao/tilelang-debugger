"""Negative controls against real H200 evidence and a deliberately wrong kernel."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

from tilelang_debugger.sample_records import parse

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    matrix, output = Path(args.matrix).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = []
    capture = matrix / "nested/capture"
    points = json.loads((capture / "points.json").read_text())
    lines = (capture / "instrumented/stdout.log").read_text().splitlines()
    samples = [line for line in lines if line.startswith("TLDBG2|value|")]
    root = next(line for line in samples if "|R|" in line)
    branch = next(line for line in samples if "|B|" in line)
    data = next(line for line in samples if "|D|" in line)
    variants = {"missing-root": [r for r in lines if r != root],
                "missing-branch": [r for r in lines if r != branch],
                "missing-instance": [r for r in lines if r != data],
                "duplicate": [*lines, data],
                "truncated-last-word": [data.rsplit("|", 2)[0][:-1] if r == data else r for r in lines],
                "missing-all-point-events": [r for r in lines if not r.startswith("TLDBG2|value|")]}
    for name, altered in variants.items():
        try:
            parse("\n".join(altered), points)
        except ValueError as exc:
            results.append(dict(case=name, passed=True, error=str(exc)))
        else:
            raise AssertionError(f"mutation unexpectedly accepted: {name}")
    baseline = parse("\n".join(lines), points)
    assert parse("\n".join(reversed(lines)), points) == baseline
    results.append(dict(case="unordered-real-events", passed=True))

    def command(name, argv, expected):
        with (output / (name + ".log")).open("w") as stream:
            process = subprocess.run([sys.executable, "-m", "tilelang_debugger", *argv], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=600)
        if process.returncode != expected:
            raise AssertionError(f"{name}: expected exit {expected}, got {process.returncode}")
        results.append(dict(case=name, passed=True, returncode=process.returncode))

    reference = ROOT / "examples/nested_scopes/reference.py"
    wrong_ref = output / "wrong_int64_reference.py"
    wrong_ref.write_text(reference.read_text().replace("9007199254740993 + tx", "9007199254740994 + tx"))
    command("int64-one-bit-reference", ["analyze", str(matrix / "local/capture"), "--reference", str(wrong_ref), "--output", str(output / "wrong-reference-analysis")], 2)
    report = json.loads((output / "wrong-reference-analysis/analysis.json").read_text())
    assert report["status"] == "completed" and report["matched"] is False
    assert next(p for p in report["comparisons"] if p["point"] == "wide")["mismatches"] == 32

    fixture = output / "wrong-kernel"
    shutil.copytree(ROOT / "examples/nested_scopes", fixture, ignore=shutil.ignore_patterns("__pycache__"))
    source = fixture / "kernel.py"
    source.write_text(source.read_text().replace("value = tx * 1000 + j * 100 + i", "value = tx * 1000 + j * 100 + i + 1"))
    spec = importlib.util.spec_from_file_location("negative_config", fixture / "configure.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = output / "wrong-monitor.json"
    config.write_text(json.dumps(module.config()))
    command("wrong-kernel-capture", ["run", str(fixture / "run.py"), "--monitor", str(config), "--output", str(output / "wrong-capture")], 2)
    command("wrong-kernel-analysis", ["analyze", str(output / "wrong-capture"), "--reference", str(reference), "--output", str(output / "wrong-analysis")], 2)
    result = json.loads((output / "wrong-analysis/analysis.json").read_text())
    assert result["status"] == "completed" and result["matched"] is False
    (output / "summary.json").write_text(json.dumps(dict(passed=True, results=results), indent=2) + "\n")
    print(json.dumps(dict(passed=True, tests=len(results))))


if __name__ == "__main__":
    main()
