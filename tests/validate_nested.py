"""H200 validation of source samples, independent references and sanitizers."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    spec = importlib.util.spec_from_file_location("fixture_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tileops", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", action="append")
    parser.add_argument("--sanitizers", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    small = load(ROOT / "examples/nested_scopes/configure.py")
    external = load(ROOT / "examples/tileops/nested/configure.py")
    cases = []
    for case in ("nested", "local", "selected", "not-executed"):
        cfg = small.config(local=case == "local")
        if case in {"selected", "not-executed"}:
            cfg["points"] = [p for p in cfg["points"] if p["id"] == "value"]
            p = cfg["points"][0]
            p["thread"] = 0 if case == "selected" else 1
            if case == "selected":
                lines = Path(cfg["source"]).read_text().splitlines()
                p["loops"] = [dict(line=i, iteration=2) for i, text in enumerate(lines, 1) if "for i in T.serial" in text]
        cases.append((case, cfg, ROOT / "examples/nested_scopes/run.py", ROOT / "examples/nested_scopes/reference.py", ["--local"] if case == "local" else []))
    for case in ("pool", "indices", "rope", "softmax"):
        cases.append((case, external.config(args.tileops, case), ROOT / "examples/tileops/nested/run.py",
                      ROOT / "examples/tileops/nested/reference.py", ["--tileops", args.tileops, "--case", case]))
    jobs = [(case, None) for case in cases if not args.case or case[0] in args.case]
    if args.sanitizers:
        jobs += [(case, tool) for case, _ in list(jobs) if case[0] in {"nested", "local", "pool", "softmax"} for tool in ("racecheck", "synccheck")]
    results = []
    for (name, cfg, driver, reference, driver_args), sanitizer in jobs:
        label = name + ("-" + sanitizer if sanitizer else "")
        folder = output / label
        folder.mkdir()
        config = folder / "monitor.json"
        config.write_text(json.dumps(cfg, indent=2) + "\n")
        commands = [[sys.executable, "-m", "tilelang_debugger", "run", str(driver), "--monitor", str(config), "--output", str(folder / "capture")]
                    + (["--sanitizer", sanitizer] if sanitizer else []) + ["--", *driver_args],
                    [sys.executable, "-m", "tilelang_debugger", "analyze", str(folder / "capture"), "--reference", str(reference), "--output", str(folder / "analysis")]]
        codes = []
        for i, command in enumerate(commands):
            with (folder / f"command-{i}.log").open("w") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT, timeout=600)
            codes.append(result.returncode)
            if result.returncode:
                break
        passed = codes == [0, 0]
        item = dict(case=label, passed=passed, returncodes=codes)
        if passed:
            run = json.loads((folder / "capture/run.json").read_text())
            item.update(records=run["records"], coverage=run["coverage"])
        results.append(item)
        (output / "summary.json").write_text(json.dumps(dict(passed=all(r["passed"] for r in results), results=results), indent=2) + "\n")
        print(json.dumps(dict(case=label, passed=passed, returncodes=codes)), flush=True)
    return 0 if results and all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
