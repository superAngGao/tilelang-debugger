"""Revalidate persisted evidence; a previous summary is never sufficient."""
import hashlib
import json
import math
import re

from .records import parse


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def reference_status(references, schema=1):
    values = []
    for ref in references:
        passed = ref.get("passed")
        if type(passed) is bool:
            values.append(passed)
        elif schema in ("source-capture-v1", "source-samples-v2") and ref == {"status": "not_provided", "passed": None}:
            values.append(None)
        else:
            raise ValueError("missing or invalid output reference result")
    if any(v is False for v in values):
        return "failed"
    return "not_checked" if any(v is None for v in values) else "passed"


def snapshots(folder, prefix):
    metadata = read(folder / f"{prefix}.json")
    if not isinstance(metadata, list) or not metadata:
        raise ValueError("missing tensor snapshots")
    for i, m in enumerate(metadata):
        if m["file"] != f"{prefix}-{i}.bin":
            raise ValueError("unexpected snapshot filename")
        raw = (folder / m["file"]).read_bytes()
        size = {"float16": 2, "bfloat16": 2, "float32": 4, "int32": 4, "int64": 8, "bool": 1}[m["dtype"]]
        if len(raw) != m["bytes"] or len(raw) != math.prod(m["shape"]) * size or hashlib.sha256(raw).hexdigest() != m["sha256"]:
            raise ValueError("tensor snapshot bytes do not match metadata")
    return metadata


def verify_capture(folder, sanitizer=None):
    if sanitizer not in (None, "racecheck", "synccheck"):
        raise ValueError("unknown sanitizer mode")
    run = read(folder / "run.json")
    if run.get('schema') == 'source-unified-v3':
        from .runtime.unified_evidence import verify
        return verify(folder, sanitizer)
    if run.get("status") != "passed" or run.get("sanitizer") != sanitizer or run.get("inputs_equal") is not True or run.get("outputs_bitwise_equal") is not True:
        raise ValueError("capture was not successful under the requested sanitizer mode")
    versions, numerical = [], []
    for mode in ("baseline", "instrumented"):
        worker = folder / mode
        execution = read(worker / "execution.json")
        if execution != dict(compiles=1, launches=1, restored=True):
            raise ValueError("worker did not complete exactly one compile/launch")
        if read(worker / "launch-gate.json").get("passed") is not True:
            raise ValueError("worker's pre-launch gate did not pass")
        if read(worker / "process.json") != dict(returncode=0, timeout=False):
            raise ValueError("worker process failed or timed out")
        numerical.append(read(worker / "reference.json"))
        if sanitizer:
            command = read(worker / "command.json")
            if command[:3] != ["compute-sanitizer", "--tool", sanitizer]:
                raise ValueError("worker command does not match requested sanitizer")
            log = (worker / f"{sanitizer}.log").read_text()
            expected = r"RACECHECK SUMMARY: 0 hazards displayed \(0 errors, 0 warnings\)" if sanitizer == "racecheck" else r"ERROR SUMMARY: 0 errors"
            if not re.search(expected, log) or "Target application returned an error" in log:
                raise ValueError("sanitizer did not report a clean completed run")
        versions.append((snapshots(worker, "inputs"), snapshots(worker, "outputs")))
    if versions[0] != versions[1]:
        raise ValueError("persisted baseline/instrumented tensor data differs")
    expected_status = reference_status(numerical, run.get("schema", 1))
    if "numerical_status" in run and run["numerical_status"] != expected_status:
        raise ValueError("numeric status differs from worker references")
    points = read(folder / "points.json")
    if not points or points != read(folder / "instrumented" / "points.json"):
        raise ValueError("point metadata is missing or differs from worker")
    log = (folder / "instrumented" / "stdout.log").read_text()
    if run.get("schema") == "source-samples-v2":
        from .sample_records import parse as parse_samples
        actual, coverage = parse_samples(log, points)
        if coverage != run.get("coverage"):
            raise ValueError("sample coverage differs from source evidence")
    else:
        actual = parse(log, points)
    persisted = [json.loads(line) for line in (folder / "records.jsonl").read_text().splitlines()]
    if actual != persisted or len(actual) != run["records"]:
        raise ValueError("records do not match complete original device log")
    return points


def verify_success(folder, sanitizer=None):
    points = verify_capture(folder, sanitizer)
    if read(folder / 'run.json').get('status') != 'passed':
        raise ValueError('partial capture is not a successful complete capture')
    if any(read(folder / mode / "reference.json")["passed"] is not True for mode in ("baseline", "instrumented")):
        raise ValueError("independent output reference failed")
    return points
