"""Isolate a supplied GQA epilogue ordering issue without altering its source."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    original = (root / "examples/gqa/kernel.original.py").read_text()
    modified = original
    for group in range(2):
        anchor = f"T.copy(acc_o, Os[{group}, :, :])"
        assert modified.count(anchor) == 1
        modified = modified.replace(anchor, anchor + f"\n                    T.fence_proxy_async()\n                    T.sync_threads({group+3}, 128)")
    summaries = []
    for mode, source in (("original", original), ("ordered", modified)):
        folder = output / mode
        folder.mkdir()
        (folder / "kernel.py").write_text(source)
        driver = (root / "examples/gqa/run.py").read_text()
        # Export the actual compiler source in this standalone diagnostic.
        driver = driver.replace("out = kernel(q, k, v)", "Path(os.environ['TLDBG_OUTPUT'], 'kernel.cu').write_text(kernel.get_kernel_source())\nout = kernel(q, k, v)")
        (folder / "run.py").write_text(driver)
        for i in range(args.repeats):
            trial = folder / str(i)
            trial.mkdir()
            env = dict(os.environ, TLDBG_OUTPUT=str(trial), TILELANG_DISABLE_CACHE="1")
            cmd = ["compute-sanitizer", "--tool", "racecheck", "--error-exitcode", "86", "--log-file", str(trial / "racecheck.log"), sys.executable, str(folder / "run.py")]
            with (trial / "stdout.log").open("w") as out, (trial / "stderr.log").open("w") as err:
                done = subprocess.run(cmd, env=env, stdout=out, stderr=err, timeout=240)
            result = dict(mode=mode, trial=i, returncode=done.returncode,
                          reference=json.loads((trial / "reference.json").read_text()) if (trial / "reference.json").exists() else None)
            summaries.append(result)
            print(json.dumps(result), flush=True)
            (output / "summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
