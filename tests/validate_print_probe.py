"""Assert the expected positive/negative controls of the original print probe."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
output = Path(args.output).resolve()
if output.exists():
    raise FileExistsError(output)
subprocess.run([sys.executable, str(root / "experiments/print_sync_probe.py"), "matrix", "--output", str(output), "--sanitizer", "compute-sanitizer"], check=True, timeout=600)
summary = json.loads((output / "summary.json").read_text())
assert len(summary) == 8
for case in summary:
    assert case["print_matches"] and case["printed_records"] == 768, case
    clean = case["mode"] in ("auto", "patched")
    assert case["returncode"] == (0 if clean else 86), case
    zero = any("0 hazards displayed (0 errors, 0 warnings)" in s for s in case["racecheck_summary"])
    assert zero == clean, case
(output / "validation.json").write_text(json.dumps(dict(passed=True, cases=8, positive_controls=4, negative_controls=4), indent=2))
