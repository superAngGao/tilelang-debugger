"""Verify source induction identities survive ordinary loop unrolling."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixture = output / "fixture"
    shutil.copytree(ROOT / "examples/nested_scopes", fixture, ignore=shutil.ignore_patterns("__pycache__"))
    source = fixture / "kernel.py"
    source.write_text(source.read_text().replace("for i in T.serial(3, 5):", "for i in T.unroll(3, 5):"))
    spec = importlib.util.spec_from_file_location("unroll_config", fixture / "configure.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = output / "monitor.json"
    config.write_text(json.dumps(module.config()))
    commands = [["run", str(fixture / "run.py"), "--monitor", str(config), "--output", str(output / "capture"), "--sanitizer", "synccheck"],
                ["analyze", str(output / "capture"), "--reference", str(ROOT / "examples/nested_scopes/reference.py"), "--output", str(output / "analysis")]]
    for i, command in enumerate(commands):
        with (output / f"command-{i}.log").open("w") as stream:
            subprocess.run([sys.executable, "-m", "tilelang_debugger", *command], stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=600)
    summary = json.loads((output / "analysis/analysis.json").read_text())
    assert summary["matched"] is True
    (output / "summary.json").write_text(json.dumps(dict(passed=True, case="unroll-original-coordinates")) + "\n")
