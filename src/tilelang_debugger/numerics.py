"""Host numeric comparison; no TileLang or GPU imports at module load."""
import json
import math
import os
from pathlib import Path

FLOATS = {"float16", "bfloat16", "float32", "float64"}
WIDTH = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8, "int32": 4}


def finite_tolerance(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("tolerance must be a finite, nonnegative number (not bool)")
    return float(value)


def json_number(value):
    if value is None or math.isfinite(value):
        return value
    return "NaN" if math.isnan(value) else ("+Inf" if value > 0 else "-Inf")


def scalar_error(actual, expected, atol, rtol):
    if math.isnan(actual) or math.isnan(expected):
        return False, None, None, "nan"
    if math.isinf(actual) or math.isinf(expected):
        same = actual == expected
        return same, 0.0 if same else math.inf, 0.0 if same else math.inf, "inf_equal" if same else "inf_mismatch"
    error = abs(actual - expected)
    relative = error / abs(expected) if expected else (0.0 if not error else math.inf)
    return error <= atol + rtol * abs(expected), error, relative, "finite"


def tensor_data(tensor):
    import torch
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.layout != torch.strided or not tensor.numel():
        raise ValueError("expected a nonempty CPU strided tensor")
    dtype = str(tensor.dtype).removeprefix("torch.")
    if dtype not in WIDTH:
        raise ValueError(f"unsupported numeric dtype: {dtype}")
    tensor = tensor.detach().contiguous().clone()
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    bits = [int.from_bytes(raw[i:i+WIDTH[dtype]], "little") for i in range(0, len(raw), WIDTH[dtype])]
    return dict(shape=list(tensor.shape), dtype=dtype, values=tensor.reshape(-1).tolist(), bits=bits)


def coordinates(index, shape):
    result = []
    for extent in reversed(shape):
        index, coord = divmod(index, extent)
        result.append(coord)
    return list(reversed(result))


def compare(actual, expected, atol, rtol):
    """Compare immutable host data; retain raw bits separately from numeric values."""
    atol, rtol = finite_tolerance(atol), finite_tolerance(rtol)
    if actual["shape"] != expected["shape"]:
        raise ValueError("reference shape must exactly match actual (no broadcasting)")
    if actual["dtype"] == "int32":
        if expected["dtype"] != "int32" or atol or rtol:
            raise ValueError("int32 requires int32 reference and zero tolerances")
    elif actual["dtype"] not in FLOATS or expected["dtype"] not in FLOATS:
        raise ValueError("floating actual requires a supported floating reference")
    count = math.prod(actual["shape"])
    if count < 1 or any(len(data[key]) != count for data in (actual, expected) for key in ("values", "bits")):
        raise ValueError("invalid or incomplete tensor data")
    rows, failures = [], []
    max_abs, max_rel = 0.0, 0.0
    specials = {side: {"nan": 0, "posinf": 0, "neginf": 0} for side in ("actual", "expected")}
    for i, (a, e) in enumerate(zip(actual["values"], expected["values"])):
        matched, absolute, relative, kind = scalar_error(a, e, atol, rtol)
        for side, number in (("actual", a), ("expected", e)):
            if math.isnan(number):
                specials[side]["nan"] += 1
            elif math.isinf(number):
                specials[side]["posinf" if number > 0 else "neginf"] += 1
        if absolute is not None:
            max_abs = max(max_abs, absolute)
            max_rel = max(max_rel, relative)
        row = dict(index=i, coordinates=coordinates(i, actual["shape"]), actual=json_number(a), expected=json_number(e),
                   actual_bits=actual["bits"][i], expected_bits=expected["bits"][i],
                   abs_error=json_number(absolute), relative_error=json_number(relative), matched=matched, kind=kind)
        rows.append(row)
        if not matched:
            failures.append(row)
    summary = dict(passed=not failures, shape=actual["shape"], actual_dtype=actual["dtype"], expected_dtype=expected["dtype"],
                   atol=atol, rtol=rtol, elements=count, mismatches=len(failures), max_abs_error=json_number(max_abs),
                   max_relative_error=json_number(max_rel), special_values=specials, first_mismatches=failures[:20])
    return summary, rows


def check_output(actual, expected, *, atol, rtol):
    """Standalone drivers assert; capture workers retain a completed numeric mismatch."""
    summary, _ = compare(tensor_data(actual.detach().cpu()), tensor_data(expected.detach().cpu()), atol, rtol)
    folder = os.environ.get("TLDBG_OUTPUT")
    if folder:
        path = Path(folder) / "reference.json"
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temp.replace(path)
    elif not summary["passed"]:
        raise AssertionError(f"output reference mismatch: {summary['mismatches']} elements")
    return summary
