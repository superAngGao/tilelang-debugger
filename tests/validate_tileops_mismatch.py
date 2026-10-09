"""Changed upstream computation must capture and complete mismatch analysis."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from tilelang_debugger.capture import save_json
from tilelang_debugger.evidence import verify_capture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tileops", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    checkout = output / "upstream-copy"
    subprocess.run(["git", "clone", "--shared", str(Path(args.tileops).resolve()), str(checkout)], check=True)
    source = checkout / "src/tileops/kernels/norm/rms_norm.py"
    original = source.read_text()
    expression = 'T.cast(x_local[i, j], "float32") * T.cast(x_local[i, j], "float32")'
    if original.count(expression) != 1:
        raise ValueError("negative fixture expression changed upstream")
    source.write_text(original.replace(expression, expression + " + 1.0"))
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "examples/tileops/capture.py"), "--tileops", str(checkout),
               "--case", "rms-n257-float16", "--output", str(output / "case")]
    save_json(output / "command.json", command)
    with (output / "stdout.log").open("w") as out, (output / "stderr.log").open("w") as err:
        process = subprocess.run(command, stdout=out, stderr=err)
    if process.returncode != 2:
        raise ValueError(f"expected numerical mismatch exit 2, got {process.returncode}")
    verify_capture(output / "case/capture")
    run = json.loads((output / "case/capture/run.json").read_text())
    analysis = json.loads((output / "case/analysis/analysis.json").read_text())
    example = json.loads((output / "case/capture/instrumented/example/result.json").read_text())
    if run["numerical_status"] != "failed" or analysis["status"] != "completed" or analysis["matched"] is not False or example["instrumented"] is not True:
        raise ValueError("mismatch or instrumented status misclassified")
    save_json(output / "summary.json", dict(passed=True, returncode=2, capture_status=run["status"],
              records=run["records"], numerical_status=run["numerical_status"], analysis_status=analysis["status"], matched=False))


if __name__ == "__main__":
    main()
