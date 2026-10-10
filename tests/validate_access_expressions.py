"""Independent per-thread path/index oracles and unchanged frontend effects."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def configuration(case):
    source = ROOT / 'examples/unified/access_expressions.py'
    lines = source.read_text().splitlines()
    specs = [('and', 'and_value =', 'x', None), ('or', 'or_value =', 'x', None),
             ('chain', 'chain_value =', 'x', None), ('yes', 'conditional_value =', 'x', 0),
             ('no', 'conditional_value =', 'x', 1), ('bit', 'bit_value =', 'x', None),
             ('python', 'python_value =', 'x', None)] if case == 'logic' else [
             ('copy', 'T.copy(src=x', 'x', None), ('add', 'T.atomic_add', 'y', None),
             ('max', 'T.atomic_max', 'y', None), ('min', 'T.atomic_min', 'y', None),
             ('address', 'pointer =', 'y', None)]
    points = []
    for pid, marker, buffer, occurrence in specs:
        p = dict(id=pid, line=next(i for i, s in enumerate(lines, 1) if marker in s),
                 when='before', mode='access', buffer=buffer, block=[0, 0, 0], loops=[])
        if occurrence is not None:
            p['occurrence'] = occurrence
        points.append(p)
    return dict(schema=3, source=str(source), points=points)


def verify(folder, case):
    from tilelang_debugger.runtime.unified_evidence import verify as evidence
    result = json.loads((folder / 'run.json').read_text())
    assert result['status'] == result['numerical_status'] == 'passed', result
    evidence(folder, result['sanitizer'])
    rows = [json.loads(s) for s in (folder / 'accesses.jsonl').read_text().splitlines()]
    pids = ['and', 'or', 'chain', 'yes', 'no', 'bit', 'python'] if case == 'logic' else ['copy', 'add', 'max', 'min', 'address']
    assert Counter((r['point'], r['thread']) for r in rows) == Counter((p, t) for p in pids for t in range(32))
    for row in rows:
        pid, tx = row['point'], row['thread']
        active = tx >= 16 if pid == 'no' else tx < 16 if pid in {'and', 'or', 'chain', 'yes', 'python'} else True
        index = (tx + 1) % 32 if pid == 'no' else (tx << 1) & 31 if pid == 'bit' else 0 if pid == 'copy' else tx
        role = 'read_write' if pid in {'add', 'max', 'min'} else 'address' if pid == 'address' else 'read'
        assert row['active'] == active and row['origin'] == [index] and row['operation'] == role, row
        assert row['extent'] == [32 if pid == 'copy' else 1] and row['shape'] == [32], row
        assert row['status'] == ('in_bounds' if active else 'masked'), row
    # Inspect only frontend artifacts to prove extra observations don't duplicate
    # target reads, synchronization, copy, address or atomic effects.
    import tilelang
    import tvm
    from tvm import tirx
    inventories = []
    for mode in ('baseline', 'instrumented'):
        f = tvm.ir.load_json((folder / mode / 'compiles/0/frontend.json').read_text())
        counts = Counter()
        def visit(n):
            if isinstance(n, tirx.BufferLoad) and str(n.buffer.name) in {'x', 'y', 'shared'}:
                counts['load:' + str(n.buffer.name)] += 1
            if isinstance(n, tirx.Call):
                op = str(getattr(n.op, 'name', ''))
                if any(s in op for s in ('atomic', 'address_of', 'copy', 'storage_sync')):
                    counts[op] += 1
        tirx.stmt_functor.post_order_visit(f.body, visit)
        inventories.append(counts)
    assert inventories[0] == inventories[1], inventories
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for case in ('logic', 'operations'):
        config = configuration(case)
        cfg = output / (case + '.json')
        cfg.write_text(json.dumps(config))
        cmd = [sys.executable, '-m', 'tilelang_debugger', 'run', config['source'], '--monitor', str(cfg),
               '--output', str(output / case), '--sanitizer', 'racecheck', '--', '--case', case]
        with (output / (case + '.log')).open('w') as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=600).returncode
        result = dict(case=case, returncode=code, passed=False)
        if not code:
            try:
                result.update(requests=verify(output / case, case), passed=True)
            except Exception as exc:
                result['error'] = repr(exc)
        results.append(result)
        (output / 'summary.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(result), flush=True)
    return int(any(not r['passed'] for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
