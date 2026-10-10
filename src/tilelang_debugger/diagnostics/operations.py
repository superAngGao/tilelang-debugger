"""Source operation labels and checks backed by explicit scalar observations."""
import ast
import math

from .values import decode, identity
from ..numerics import json_number


def describe(source, line):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return dict(statement='', calls=[], sensitive=[], conversion=None)
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.stmt) and n.lineno == line]
    if len(nodes) != 1:
        return dict(statement='', calls=[], sensitive=[], conversion=None)
    statement = nodes[0]
    sensitive, calls = [], []
    for n in ast.walk(statement):
        if isinstance(n, ast.Call):
            name = ast.unparse(n.func)
            calls.append(name)
            if name in {'T.sqrt', 'T.rsqrt'} and n.args:
                sensitive.append(dict(kind='sqrt' if name == 'T.sqrt' else 'rsqrt', operand=ast.unparse(n.args[0])))
        elif isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Div, ast.FloorDiv, ast.Mod)):
            sensitive.append(dict(kind='denominator', operand=ast.unparse(n.right)))
    conversion = None
    if isinstance(statement, (ast.Assign, ast.AnnAssign)):
        target = statement.targets[0] if isinstance(statement, ast.Assign) and len(statement.targets) == 1 else getattr(statement, 'target', None)
        expr = statement.value
        if target is not None and isinstance(expr, ast.Call) and ast.unparse(expr.func) == 'T.cast' and len(expr.args) == 2:
            conversion = dict(input=ast.unparse(expr.args[0]), output=ast.unparse(target), target_dtype=ast.unparse(expr.args[1]))
    return dict(statement=ast.get_source_segment(source, statement) or ast.unparse(statement),
                calls=sorted(set(calls)), sensitive=sensitive, conversion=conversion)


def checks(points, near_zero):
    results = []
    groups = {}
    for point in points:
        groups.setdefault((point['launch'], point['line']), []).append(point)
    for (launch, line), group in groups.items():
        description = group[0]['operation']
        for check in description['sensitive']:
            candidates = [p for p in group if p['mode'] == 'value' and p['when'] == 'before' and p['buffer'] == check['operand'] and (p['shape'] == [] or p.get('memory') == 'local.var')]
            rows = [r for p in candidates for r in p['values']]
            def suspicious(row):
                v = decode(row['actual'])
                return not math.isfinite(v) or (abs(v) <= near_zero if check['kind'] == 'denominator' else v < 0 if check['kind'] == 'sqrt' else v <= 0)
            bad = [r for r in rows if suspicious(r)]
            results.append(dict(launch=launch, line=line, **check, status='observed' if rows else 'not_collected',
                                samples=len(rows), signals=len(bad), examples=bad[:20], near_zero=near_zero,
                                explanation='操作数风险信号；不证明分支内的该运算实际执行。'))
        conversion = description['conversion']
        if conversion:
            before = [p for p in group if p['mode'] == 'value' and p['when'] == 'before' and p['buffer'] == conversion['input'] and (p['shape'] == [] or p.get('memory') == 'local.var')]
            after = [p for p in group if p['mode'] == 'value' and p['when'] == 'after' and p['buffer'] == conversion['output'] and (p['shape'] == [] or p.get('memory') == 'local.var')]
            pairs, errors, seen = [], [], set()
            # Require one unambiguous before/after observation and exact execution keys.
            if len(before) == len(after) == 1:
                original = {identity(r, point=False): r for r in before[0]['values']}
                for row in after[0]['values']:
                    key = identity(row, point=False)
                    if key in original and key not in seen:
                        seen.add(key)
                        a, b = decode(original[key]['actual']), decode(row['actual'])
                        error = abs(a-b) if math.isfinite(a) and math.isfinite(b) else None
                        if error is not None:
                            errors.append(error)
                        pairs.append(dict(before=original[key], after=row, abs_change=json_number(error)))
            results.append(dict(launch=launch, line=line, kind='conversion', **conversion,
                                status='observed' if pairs else 'not_collected', samples=len(pairs),
                                input_dtype=before[0]['dtype'] if len(before) == 1 else None,
                                output_dtype=after[0]['dtype'] if len(after) == 1 else None,
                                max_abs_change=max(errors) if errors else None, examples=pairs[:20],
                                explanation='按执行身份对齐的转换前后变化；舍入变化不自动判为错误。'))
    return results
