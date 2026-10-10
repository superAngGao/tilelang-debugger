"""Unmodified TileOPs kernels with independently enumerated source accesses."""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def configuration(checkout, case):
    filename = {'pool': 'pool/max_pool2d.py', 'rope': 'rope.py', 'softmax': 'reduction/softmax.py'}[case]
    source = Path(checkout) / 'src/tileops/kernels' / filename
    tree = ast.parse(source.read_text())
    name = {'pool': '_max_pool2d_kernel', 'rope': '_make_rope_neox_position_ids_thd', 'softmax': '_softmax_kernel_tiled'}[case]
    factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    assignments = sorted((n for n in ast.walk(factory) if isinstance(n, ast.Assign)), key=lambda n: n.lineno)
    def assignment(target):
        return next(n for n in assignments if ast.unparse(n.targets[0]) == target)
    def point(pid, node, buffer, **options):
        return dict(id=pid, line=node.lineno, mode='access', when='before', buffer=buffer, block=[0, 0, 0], loops=[], **options)
    if case == 'pool':
        points = [point('input', assignment('val'), 'x')]
    elif case == 'rope':
        points = [point('paired', assignment('paired_val'), 'x'), point('table', assignment('c'), 'cos_table')]
    else:
        copies = sorted((n for n in ast.walk(factory) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'T.copy' and ast.unparse(n.value.args[0]).startswith('x[')), key=lambda n: n.lineno)
        tail = next(n for n in assignments if ast.unparse(n.targets[0]) == 'tile_f32[i, j]' and 'x[' in ast.unparse(n.value))
        points = [point('copy', copies[0], 'x', thread=0), point('tail', tail, 'x')]
    return dict(schema=3, source=str(source.resolve()), points=points)


def verify(folder, case):
    result = json.loads((folder / 'run.json').read_text())
    assert result['status'] == 'passed' and result['numerical_status'] == 'passed'
    rows = [json.loads(line) for line in (folder / 'accesses.jsonl').read_text().splitlines()]
    actual = Counter((r['point'], tuple(r['coordinates']), tuple(r['origin']), r['active']) for r in rows)
    expected = Counter()
    if case == 'pool':
        for i in range(12):
            for kh in range(3):
                for kw in range(3):
                    ih, iw = i // 4 - 1 + kh, i % 4 - 1 + kw
                    if 0 <= ih < 3 and 0 <= iw < 4:
                        expected[('input', (i, kh, kw), (0, 0, ih, iw), True)] += 1
    elif case == 'rope':
        for flat in range(48):
            col = flat % 8
            paired = flat - col + (col + 4 if col < 4 else col - 4)
            coords = (flat // 2, flat % 2)
            expected[('paired', coords, (paired,), True)] += 1
            expected[('table', coords, ((3, 1, 6)[flat // 16], col % 4), True)] += 1
    else:
        for tile in range(2):
            expected[('copy', (tile,), (0, tile * 256), True)] += 1
        for j in range(256):
            expected[('tail', (2, 0, j), (0, 512 + j), j == 0)] += 1
    assert actual == expected, (case, actual - expected, expected - actual)
    for row in rows:
        assert row['status'] == ('in_bounds' if row['active'] else 'masked')
        if row['kind'] == 'copy':
            assert row['extent'] == [1, 256]
    from tilelang_debugger.runtime.unified_evidence import verify as evidence
    evidence(folder, result['sanitizer'])
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tileops', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sanitizer', choices=('racecheck', 'synccheck'))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for case in ('pool', 'rope', 'softmax'):
        cfg = output / f'{case}.json'
        cfg.write_text(json.dumps(configuration(args.tileops, case)))
        folder = output / case
        cmd = [sys.executable, '-m', 'tilelang_debugger', 'run', str(ROOT / 'examples/tileops/nested/run.py'), '--monitor', str(cfg), '--output', str(folder)]
        if args.sanitizer:
            cmd += ['--sanitizer', args.sanitizer]
        cmd += ['--', '--tileops', args.tileops, '--case', case]
        with (output / f'{case}.log').open('w') as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=400).returncode
        result = dict(case=case, returncode=code, passed=False)
        if not code:
            result.update(requests=verify(folder, case), passed=True)
        results.append(result)
        (output / 'summary.json').write_text(json.dumps(results, indent=2))
        print(result, flush=True)
    return int(any(not r['passed'] for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
