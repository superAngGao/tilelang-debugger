"""H200 source-print matrix; each case must pass capture and independent analysis."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "tileops"))
from common import cases, save
from tilelang_debugger.evidence import verify_success


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tileops", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", action="append")
    parser.add_argument("--sanitizers", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    selected = [c for c in cases() if not args.case or c["id"] in args.case]
    if not selected or args.case and set(args.case) != {c["id"] for c in selected}:
        parser.error("unknown case")
    jobs = [(c, None) for c in selected]
    if args.sanitizers:
        representatives = ("softmax-tiled-float32", "rms-n257-float16", "rope-32x64-bfloat16")
        jobs += [(c, tool) for c in selected if c["id"] in representatives for tool in ("racecheck", "synccheck")]
    results = []
    for case, sanitizer in jobs:
        label = case["id"] + ("-" + sanitizer if sanitizer else "")
        folder = output / label
        command = [sys.executable, str(ROOT / "examples/tileops/capture.py"), "--tileops", args.tileops,
                   "--case", case["id"], "--output", str(folder)]
        if sanitizer: command += ["--sanitizer", sanitizer]
        with (output / (label + ".log")).open("w") as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        result = dict(case=case["id"], sanitizer=sanitizer, returncode=process.returncode, passed=False)
        if process.returncode == 0:
            try:
                points = verify_success(folder / "capture", sanitizer)
                analysis = json.loads((folder / "analysis/analysis.json").read_text())
                if analysis["status"] != "completed" or analysis["matched"] is not True:
                    raise ValueError("independent analysis did not match")
                result.update(passed=True, points=[p["id"] for p in points],
                              records=json.loads((folder / "capture/run.json").read_text())["records"])
            except Exception as exc:
                result["error"] = str(exc)
        results.append(result)
        save(output / "summary.json", dict(passed=all(r["passed"] for r in results) and len(results) == len(jobs), expected=len(jobs), results=results))
        print(json.dumps(result), flush=True)
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
