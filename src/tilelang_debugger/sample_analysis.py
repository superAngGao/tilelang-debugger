"""Offline comparison of independently declared sample-key domains."""
import copy
import hashlib
import json
import math
from pathlib import Path

from .evidence import read, verify_capture
from .numerics import compare, finite_tolerance, json_number, scalar_error, tensor_data
from .records import value


def compare_samples(records, point, spec):
    required = {"schema", "key", "dtype", "samples", "atol", "rtol"}
    if not isinstance(spec, dict) or set(spec) != required or spec["schema"] != 2 or spec["key"] not in {"logical", "execution"}:
        raise ValueError("samples reference requires schema=2, key, dtype, samples, atol, rtol")
    if not isinstance(spec["samples"], list):
        raise ValueError("reference samples must be an independent key/value list")
    dtype = point["dtype"] or spec["dtype"]
    if spec["dtype"] != dtype or dtype not in {"bool", "int32", "int64", "float16", "bfloat16", "float32"}:
        raise ValueError("sample reference dtype must match observation")
    atol, rtol = finite_tolerance(spec["atol"]), finite_tolerance(spec["rtol"])
    integer = dtype in {"bool", "int32", "int64"}
    if integer and (atol or rtol):
        raise ValueError("integer/bool comparison requires zero tolerances")
    def key(row):
        coords, index = row.get("coordinates"), row.get("index")
        if not isinstance(coords, list) or any(type(v) is not int for v in coords) or type(index) is not int or index < 0:
            raise ValueError("invalid reference sample key")
        if spec["key"] == "execution":
            if type(row.get("thread")) is not int or row["thread"] < 0:
                raise ValueError("execution reference requires thread")
            return (row["thread"], tuple(coords), index)
        return (tuple(coords), index)
    expected = {}
    for row in spec["samples"]:
        fields = {"coordinates", "index", "value"} | ({"thread"} if spec["key"] == "execution" else set())
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError("invalid reference row fields")
        k, v = key(row), row["value"]
        if k in expected:
            raise ValueError("duplicate reference sample key")
        if dtype == "bool":
            valid = type(v) is bool
        elif integer:
            width = 32 if dtype == "int32" else 64
            valid = type(v) is int and -(2**(width - 1)) <= v < 2**(width - 1)
        else:
            valid = type(v) in {int, float}
        if not valid:
            raise ValueError("reference value does not fit declared dtype")
        expected[k] = v
    actual_keys = {key(r) for r in records}
    missing, unexpected = set(expected) - actual_keys, actual_keys - set(expected)
    rows, mismatches = [], len(missing) + len(unexpected)
    for r in records:
        k = key(r)
        a, e = value(r["bits"], dtype), expected.get(k)
        if k not in expected:
            matched, error = False, None
        elif integer:
            matched, error = a == e, abs(int(a) - int(e))
        else:
            matched, error, _, _ = scalar_error(a, e, atol, rtol)
        mismatches += int(k in expected and not matched)
        rows.append(dict(point=point["id"], thread=r["thread"], coordinates=r["coordinates"], index=r["index"],
                         bits=r["bits"], actual=json_number(a), expected=json_number(e), matched=matched, abs_error=json_number(error)))
    return dict(point=point["id"], dtype=dtype, key=spec["key"], matched=mismatches == 0,
                observed_records=len(records), expected_keys=len(expected), mismatches=mismatches,
                missing_keys=[repr(k) for k in sorted(missing)], unexpected_keys=[repr(k) for k in sorted(unexpected)]), rows


def analyze(capture, reference, output):
    from .analysis import load_provider, manifest, save, tensors
    capture, reference, output = (Path(p).resolve() for p in (capture, reference, output))
    if output == capture or capture in output.parents:
        raise ValueError("analysis output must be outside capture")
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(schema="sample-analysis-v2", status="running")
    save(output / "analysis.json", summary)
    try:
        run, evidence = read(capture / "run.json"), manifest(capture)
        points = verify_capture(capture, run.get("sanitizer"))
        records = [json.loads(line) for line in (capture / "records.jsonl").read_text().splitlines()]
        inputs, outputs = tensors(capture / "baseline", "inputs"), tensors(capture / "baseline", "outputs")
        source = reference.read_bytes()
        (output / "reference.py").write_bytes(source)
        with load_provider(source, reference) as provider:
            bundle = provider(tuple(t.clone() for t in inputs), copy.deepcopy(points))
        if not isinstance(bundle, dict) or set(bundle) != {"points", "outputs"} or set(bundle["points"]) != {p["id"] for p in points} or len(bundle["outputs"]) != len(outputs):
            raise ValueError("reference must cover exactly every point and output")
        comparisons, rows = [], []
        for p in points:
            report, values = compare_samples([r for r in records if r["point"] == p["id"]], p, bundle["points"][p["id"]])
            comparisons.append(report)
            rows.extend(values)
        output_reports = []
        for tensor, spec in zip(outputs, bundle["outputs"]):
            if set(spec) != {"tensor", "atol", "rtol"}:
                raise ValueError("output reference requires tensor, atol, rtol")
            report, _ = compare(tensor_data(tensor), tensor_data(spec["tensor"]), spec["atol"], spec["rtol"])
            output_reports.append(report)
        if manifest(capture) != evidence:
            raise ValueError("capture evidence changed during analysis")
        summary.update(status="completed", run_id=run["run_id"], coverage=run["coverage"],
                       matched=all(r["matched"] for r in comparisons) and all(r["passed"] for r in output_reports),
                       comparisons=comparisons, outputs=output_reports,
                       provider=dict(path=str(reference), sha256=hashlib.sha256(source).hexdigest()))
        save(output / "evidence.json", evidence)
        (output / "elements.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
        lines = ["# Sample analysis", "", f"Numerical comparison: {'matched' if summary['matched'] else 'mismatch'}.", "",
                 "| Point | Records | Missing keys | Mismatches | Logical coverage | Execution coverage |",
                 "| --- | ---: | ---: | ---: | --- | --- |"]
        for r in comparisons:
            c = run["coverage"][r["point"]]
            lines.append(f"| {r['point']} | {r['observed_records']} | {len(r['missing_keys'])} | {r['mismatches']} | {c['logical_coverage']} | {c['execution_coverage']} |")
        lines += ["", "A matching reference does not upgrade unverified execution coverage. See points.json for original source locations and binding identities.", ""]
        (output / "report.md").write_text("\n".join(lines))
        save(output / "analysis.json", summary)
        return summary
    except Exception as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        summary.pop("matched", None)
        save(output / "analysis.json", summary)
        raise
