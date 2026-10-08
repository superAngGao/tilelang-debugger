"""Maintainer diagnostic: derive candidate layouts only from identical, validated CUDA.

Does not modify contracts. A reviewer must approve the resulting candidate file.
Instrumented workers stop before launch because layouts are not yet authorized.
"""
import argparse
import json
from pathlib import Path

from tilelang_debugger.capture import run
from tilelang_debugger.evidence import verify_success
from tilelang_debugger.instrument import digest, prepare

parser = argparse.ArgumentParser()
parser.add_argument("--verified", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
output = Path(args.output).resolve()
output.mkdir(parents=True, exist_ok=False)
candidates = {}
for name in ("gelu", "sum", "sum_unpadded", "gemm", "gqa"):
    verified = Path(args.verified).resolve() / name
    verify_success(verified, "racecheck")
    example = root / "examples" / ("sum" if name == "sum_unpadded" else name)
    driver = example / ("run_unpadded.py" if name == "sum_unpadded" else "run.py")
    _, points = prepare((example / "kernel.py").read_text(), json.loads((example / "monitor.json").read_text()), driver.read_bytes())
    if any(p.get("layout_sha256") for p in points):
        raise RuntimeError("layout is already authorized; diagnostic will not bypass or replace a reviewed contract")
    folder = output / name
    try:
        run(example / ("run_unpadded.py" if name == "sum_unpadded" else "run.py"), example / "monitor.json", folder)
    except RuntimeError:
        log = (folder / "instrumented/stderr.log").read_text()
        if "no reviewed layout contract" not in log:
            raise
    else:
        raise AssertionError("diagnostic was expected to refuse launch without a layout contract")
    if (folder / "instrumented/inputs.json").exists():
        raise AssertionError("instrumented launch must not have begun")
    current = (folder / "instrumented/kernel.cu").read_bytes()
    accepted = (verified / "instrumented/kernel.cu").read_bytes()
    if current != accepted:
        raise AssertionError(f"{name}: compiled CUDA is not identical to the verified artifact")
    candidates[name] = dict(verified_cuda_sha256=digest(accepted),
                            layouts=json.loads((folder / "instrumented/observed-layouts.json").read_text())["observed_layouts"])
    print(json.dumps({name: candidates[name]}), flush=True)
    (output / "candidates.json").write_text(json.dumps(candidates, indent=2))
