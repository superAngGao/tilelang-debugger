"""Compare source plans and inserted text with a saved pre-refactor baseline."""
import argparse
import importlib.util
import json
from pathlib import Path

from tilelang_debugger import instrument, source_instrument

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot(checkout):
    result = {}
    small = load(ROOT / "examples/nested_scopes/configure.py", "small_configuration")
    external = load(ROOT / "examples/tileops/nested/configure.py", "external_configuration")
    configs = [("nested", small.config()), ("local", small.config(local=True))]
    configs += [(case, external.config(checkout, case)) for case in ("pool", "indices", "rope", "softmax")]
    old = load(ROOT / "tests/test_source_engine.py", "source_fixture")
    for case, source, config in [("legacy-source", old.SOURCE, old.config()),
                                 *[(name, Path(cfg["source"]).read_text(encoding="utf-8"), cfg) for name, cfg in configs]]:
        points = source_instrument.prepare(source, config)
        result[case] = dict(points=points, source=source_instrument.inject(source, points))
    for name in ("gelu", "sum", "gemm", "gqa"):
        folder = ROOT / "examples" / name
        source = (folder / "kernel.py").read_text(encoding="utf-8")
        config = json.loads((folder / "monitor.json").read_text())
        _, points = instrument.prepare(source, config)
        result["reviewed-" + name] = dict(source=instrument.inject(source, points), points=points)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tileops", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--create", action="store_true")
    args = parser.parse_args()
    actual = snapshot(args.tileops)
    baseline = Path(args.baseline)
    if args.create:
        baseline.parent.mkdir(parents=True, exist_ok=True)
        with baseline.open("x", encoding="utf-8") as stream:
            json.dump(actual, stream, ensure_ascii=False, indent=2)
    else:
        expected = json.loads(baseline.read_text(encoding="utf-8"))
        assert actual == expected, "source plan or inserted text changed"
    print(json.dumps(dict(passed=True, cases=len(actual), created=args.create)))
