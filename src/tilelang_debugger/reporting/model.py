"""Versioned browser/export model; large integers remain exact decimal strings."""
import math


def portable(value):
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(v) for v in value]
    if type(value) is int and abs(value) > 2**53 - 1:
        return str(value)
    if type(value) is float and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else '+Inf' if value > 0 else '-Inf'
    return value
