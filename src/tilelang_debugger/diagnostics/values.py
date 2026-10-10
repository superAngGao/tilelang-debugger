"""Numeric summaries preserving exact integers and explicit nonfinite values."""
import math

from ..numerics import json_number
from ..protocols.unified import value


def decode(number):
    return {'NaN': math.nan, '+Inf': math.inf, '-Inf': -math.inf}.get(number, number) if isinstance(number, str) else number


def identity(row, *, point=True):
    return (row['launch'], row['compile'], row['point'] if point else None,
            tuple(row['block']), row['thread'], row['visit'], tuple(row['coordinates']),
            tuple(row['ordinals']), row['index'])


def samples(records, comparisons):
    result = []
    for record in records:
        actual = value(record['bits'], record['dtype'])
        row = dict(record, actual=json_number(actual), bits=hex(record['bits']), expected=None,
                   matched=None, abs_error=None, relative_error=None, comparison='not_provided')
        comparison = comparisons.get(identity(record))
        if comparison is not None:
            expected = decode(comparison['expected'])
            error = abs(actual - expected) if expected is not None else None
            if expected is None or math.isnan(actual) or math.isnan(expected):
                error, relative = None, None
            elif math.isinf(actual) or math.isinf(expected):
                error = 0 if actual == expected else math.inf
                relative = error
            else:
                relative = error / abs(expected) if expected else (0 if error == 0 else math.inf)
            row.update(expected=comparison['expected'], matched=comparison['matched'], abs_error=json_number(error),
                       relative_error=json_number(relative), comparison='saved_reference')
        result.append(row)
    return result


def summarize(rows):
    actual = [decode(r['actual']) for r in rows]
    finite = [v for v in actual if math.isfinite(v)]
    errors = [decode(r['abs_error']) for r in rows if r['abs_error'] is not None]
    relative = [decode(r['relative_error']) for r in rows if r['relative_error'] is not None]
    return dict(samples=len(rows), finite=len(finite), nan=sum(math.isnan(v) for v in actual),
                posinf=sum(v == math.inf for v in actual), neginf=sum(v == -math.inf for v in actual),
                minimum=min(finite) if finite else None, maximum=max(finite) if finite else None,
                compared=sum(r['matched'] is not None for r in rows),
                mismatches=sum(r['matched'] is False for r in rows),
                max_abs_error=json_number(max(errors)) if errors else None,
                max_relative_error=json_number(max(relative)) if relative else None)
