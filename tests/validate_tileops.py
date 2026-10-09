"""H200 upstream baselines and honest public-CLI readiness probes."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "tileops"
sys.path.insert(0, str(EXAMPLES))
from common import cases, save, sha


def execute(command, folder, env, timeout):
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    with (folder / "stdout.log").open("w") as stdout, (folder / "stderr.log").open("w") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env, cwd=ROOT,
                                   start_new_session=sys.platform == "linux")
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            if sys.platform == "linux":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
    result = dict(command=command, returncode=process.returncode, timed_out=timed_out,
                  seconds=time.monotonic() - started)
    save(folder / "process.json", result)
    return result


def debugger_status(process, stderr, driver, command):
    if process["timed_out"] or process["returncode"] < 0:
        return "failed"
    if process["returncode"] == 0:
        return "unverified"  # Never infer complete captures from a zero exit alone.
    expected = driver.resolve().with_name("kernel.py")
    module = "capture.py" if command == "run" else "access.py"
    source_read = ('source = source_file.read_text(encoding="utf-8")' if command == "run"
                   else 'source = driver.with_name("kernel.py").read_text(encoding="utf-8")')
    lines = stderr.rstrip().splitlines()
    missing = f"FileNotFoundError: [Errno 2] No such file or directory: {str(expected)!r}"
    # Match the exact known public-entry failure, not an arbitrary missing dependency/config.
    if (lines and lines[-1] == missing and "Traceback (most recent call last):" in stderr
            and f'{module}", line ' in stderr
            and source_read in stderr
            and not expected.exists()):
        return "unsupported"
    return "failed"


def baseline_status(folder, process, sanitizer):
    if process["returncode"] != 0 or process["timed_out"]:
        return False
    try:
        files = json.loads((folder / "result" / "files.json").read_text())
        required = {"result.json", "provenance.json", "environment.json", "compile.json", "frontend.py",
                    "frontend.json", "device.py", "device.json", "kernel.cu", "tensors.pt",
                    "comparison.json", "elements.jsonl", "observations.json"}
        if not required.issubset(files) or any(Path(name).name != name for name in files):
            return False
        if any(sha(folder / "result" / name) != value for name, value in files.items()):
            return False
        result = json.loads((folder / "result" / "result.json").read_text())
        if result["status"] != "passed" or not result["padding_zero"] or result["instrumented"]:
            return False
        if sanitizer:
            text = (folder / "stdout.log").read_text() + (folder / "stderr.log").read_text()
            summaries = re.findall(r"(?:ERROR|RACECHECK) SUMMARY: (\d+) (?:errors|hazards)", text)
            if not summaries or any(int(x) for x in summaries):
                return False
        return True
    except (OSError, KeyError, ValueError):
        return False


def suite_status(baselines, probes, require_debugger):
    baseline_ok = bool(baselines) and all(r["passed"] for r in baselines)
    debugger_ok = bool(probes) and all(r["status"] == "passed" for r in probes)
    probe_errors = not probes or any(r["status"] not in ("passed", "unsupported") for r in probes)
    return dict(baseline_passed=baseline_ok,
                debugger_status="passed" if debugger_ok else "failed" if probe_errors else "unsupported",
                delivery_passed=baseline_ok and debugger_ok,
                exitcode=0 if baseline_ok and not probe_errors and (not require_debugger or debugger_ok) else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tileops", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", action="append", help="Limit baseline cases (repeatable)")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--sanitizers", action="store_true", help="Add all three sanitizers to one case per example")
    parser.add_argument("--require-debugger", action="store_true", help="Fail if public run/trace cannot capture these kernels")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("H200 validation runs on Linux")
    selected = [c for c in cases() if not args.case or c["id"] in args.case]
    if not selected or (args.case and set(args.case) != {c["id"] for c in selected}):
        parser.error("unknown case")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), TILELANG_DISABLE_CACHE="1", PYTHONDONTWRITEBYTECODE="1")
    baseline, probes = [], []
    jobs = [(c, None) for c in selected]
    representatives = {name: next(c for c in selected if c["example"] == name) for name in dict.fromkeys(c["example"] for c in selected)}
    if args.sanitizers:
        jobs += [(c, tool) for c in representatives.values() for tool in ("memcheck", "racecheck", "synccheck")]
    for case, sanitizer in jobs:
        label = case["id"] + ("-" + sanitizer if sanitizer else "")
        folder = output / label
        command = [sys.executable, str(EXAMPLES / case["example"] / "run.py"), "--tileops", str(Path(args.tileops).resolve()),
                   "--case", case["id"], "--output", str(folder / "result")]
        if sanitizer:
            command = ["compute-sanitizer", "--tool", sanitizer, "--error-exitcode", "86", "--target-processes", "all", *command]
        try:
            process = execute(command, folder, env, args.timeout)
            passed = baseline_status(folder, process, sanitizer)
            result = dict(case=case["id"], sanitizer=sanitizer, passed=passed, process=process)
        except OSError as exc:
            result = dict(case=case["id"], sanitizer=sanitizer, passed=False, error=str(exc))
        baseline.append(result)
        save(output / "baselines.json", baseline)
        print(json.dumps(dict(case=label, passed=result["passed"])), flush=True)
    for example in representatives:
        driver = EXAMPLES / example / "run.py"
        for command in ("run", "trace"):
            kind = "monitor" if command == "run" else "access"
            folder = output / f"{example}-{command}-probe"
            cmd = [sys.executable, "-m", "tilelang_debugger", command, str(driver), f"--{kind}",
                   str(driver.with_name(kind + ".json")), "--output", str(folder / "capture")]
            process = execute(cmd, folder, env, args.timeout)
            status = debugger_status(process, (folder / "stderr.log").read_text(), driver, command)
            probes.append(dict(example=example, command=command, status=status, process=process,
                               reason="See stderr.log; configuration compatibility has not been reached"))
            save(output / "debugger-probes.json", probes)
    result = dict(suite_status(baseline, probes, args.require_debugger), baselines=baseline, probes=probes,
                  scope="Upstream numerical baselines and public-entry readiness, not debug capture acceptance")
    save(output / "summary.json", result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("baselines", "probes")}), flush=True)
    return result["exitcode"]


if __name__ == "__main__":
    raise SystemExit(main())
