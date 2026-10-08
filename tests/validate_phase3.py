"""Analyze genuine existing captures through the public CLI, including bad references."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from tilelang_debugger.analysis import save

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captures", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", nargs="+", default=["gelu", "sum", "sum_unpadded", "gemm", "gqa"])
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases = {}
    summary = dict(status="running", captures=str(Path(args.captures).resolve()), cases=cases)
    save(output / "summary.json", summary)
    try:
        for name in args.cases:
            example = "sum" if name == "sum_unpadded" else name
            reference = ROOT / "examples" / example / "reference.py"
            destination = output / name
            command = [sys.executable, "-m", "tilelang_debugger", "analyze", str(Path(args.captures).resolve() / name),
                       "--reference", str(reference), "--output", str(destination)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            (output / f"{name}.stdout.log").write_text(result.stdout)
            (output / f"{name}.stderr.log").write_text(result.stderr)
            if result.returncode != 0:
                raise AssertionError(f"analysis failed: {name}: {result.stderr}")
            report = json.loads((destination / "analysis.json").read_text())
            assert report["status"] == "completed" and report["matched"] is True
            cases[name] = dict(passed=True, comparisons=[{k: c[k] for k in ("label", "elements", "mismatches", "max_abs_error")} for c in report["comparisons"]])
            print(json.dumps({name: cases[name]}), flush=True)
            save(output / "summary.json", summary)
        # Wrong mathematical reference, NOT tampered capture evidence.
        name = args.cases[0]
        example = "sum" if name == "sum_unpadded" else name
        original = (ROOT / "examples" / example / "reference.py").read_text()
        wrong = output / "wrong_reference.py"
        wrong.write_text(original + '\n_original = reference\ndef reference(inputs, points):\n    bundle = _original(inputs, points)\n    for spec in bundle["points"].values():\n        spec["tensor"] = spec["tensor"].float() + 100\n    return bundle\n')
        result = subprocess.run([sys.executable, "-m", "tilelang_debugger", "analyze", str(Path(args.captures).resolve() / name),
                                 "--reference", str(wrong), "--output", str(output / "wrong-reference")], capture_output=True, text=True, timeout=120)
        (output / "wrong-reference.stdout.log").write_text(result.stdout)
        (output / "wrong-reference.stderr.log").write_text(result.stderr)
        assert result.returncode == 2, result.stderr
        report = json.loads((output / "wrong-reference/analysis.json").read_text())
        assert report["status"] == "completed" and report["matched"] is False
        for c in report["comparisons"]:
            assert c["mismatches"] == (c["elements"] if c["kind"] == "point" else 0)
        summary.update(status="passed", negative_reference=dict(exit_code=2, status="completed", matched=False))
        save(output / "summary.json", summary)
    except Exception as exc:
        summary.update(status="failed", error=str(exc))
        save(output / "summary.json", summary)
        raise


if __name__ == "__main__":
    main()
