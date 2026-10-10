"""Source-only access acceptance; expected indices never come from captured rows."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def configuration(source):
    lines = source.read_text().splitlines()
    def point(pid, needle, buffer, **options):
        return dict(id=pid, line=next(i for i, line in enumerate(lines, 1) if needle in line),
                    when='before', mode='access', buffer=buffer, block=[0, 0, 0], loops=[], **options)
    points = [point('read', 'loaded =', 'x'), point('write', 'y[k *', 'y'),
              point('branch', 'branch_value =', 'x'), point('copy_read', 'T.copy(x[32]', 'x', thread=0),
              point('copy_write', 'T.copy(shared,', 'y', thread=0),
              point('shared_read', 'T.copy(shared,', 'shared', thread=0),
              point('while', 'while_loaded =', 'x', thread=0)]
    points.append(dict(id='value', line=points[0]['line'], when='after', buffer='loaded', block=[0, 0, 0], loops=[]))
    return dict(schema=3, source=str(source), points=points)


def verify(folder, shift):
    run = json.loads((folder / 'run.json').read_text())
    assert run['status'] == 'passed' and run['numerical_status'] == 'passed'
    rows = [json.loads(line) for line in (folder / 'accesses.jsonl').read_text().splitlines()]
    groups = {pid: [r for r in rows if r['point'] == pid] for pid in ('read', 'write', 'branch', 'copy_read', 'copy_write', 'shared_read', 'while')}
    expected = {(tx, k): k * 32 + tx + shift for tx in range(32) for k in range(2)}
    assert {(r['thread'], r['coordinates'][0]): r['origin'][0] for r in groups['read']} == expected
    assert {(r['thread'], r['coordinates'][0]) for r in groups['branch']} == {key for key, i in expected.items() if i < 49}
    for r in groups['read']:
        assert r['active'] == (r['origin'][0] < 49)
        assert r['status'] == ('in_bounds' if r['active'] else 'masked')
        assert r['shape'] == [49] and r['stride'] == [1] and r['extent'] == [1]
        assert r['derived_byte_offset'] == r['origin'][0] * 4
        assert r['ordinals'] == [r['coordinates'][0] + 1]
    assert {(r['thread'], r['coordinates'][0]): r['origin'][0] for r in groups['write']} == {(tx, k): k * 32 + tx for tx in range(32) for k in range(2)}
    for pid in ('copy_read', 'copy_write'):
        assert len(groups[pid]) == 1
        row = groups[pid][0]
        assert row['origin'] == [32] and row['extent'] == [32] and row['thread'] == 0
        assert row['status'] == ('partial' if pid == 'copy_read' else 'in_bounds')
    assert groups['copy_read'][0]['in_bounds_region'] == [[32, 49]]
    assert len(groups['shared_read']) == 1 and groups['shared_read'][0]['memory_scope'] in {'shared', 'shared.dyn'}
    assert groups['shared_read'][0]['origin'] == [0] and groups['shared_read'][0]['extent'] == [32]
    assert [(r['origin'], r['ordinals']) for r in groups['while']] == [([1], [1]), ([3], [3])]
    for mode in ('baseline', 'instrumented'):
        for name in ('frontend.py', 'device.py', 'kernel.cu'):
            assert (folder / mode / 'compiles/0' / name).is_file()
    # Test-only frontend inspection: observation adds no target loads, copies,
    # or synchronization. This is not a product compilation admission check.
    import tilelang
    import tvm
    from tvm import tirx
    from collections import Counter
    inventories = []
    for mode in ('baseline', 'instrumented'):
        frontend = tvm.ir.load_json((folder / mode / 'compiles/0/frontend.json').read_text())
        counts = Counter()
        def inspect(node):
            if isinstance(node, tirx.BufferLoad) and str(node.buffer.name) == 'x':
                counts['x_loads'] += 1
            if isinstance(node, tirx.Call) and str(getattr(node.op, 'name', '')) in {'tl.tileop.copy', 'tirx.tvm_storage_sync'}:
                counts[str(node.op.name)] += 1
        tirx.stmt_functor.post_order_visit(frontend.body, inspect)
        inventories.append(counts)
    assert inventories[0] == inventories[1], inventories
    from tilelang_debugger.runtime.unified_evidence import verify as verify_evidence
    verify_evidence(folder, run['sanitizer'])
    # Changing a derived offset must fail independent evidence revalidation.
    path = folder / 'accesses.jsonl'
    original = path.read_bytes()
    rows[0]['derived_byte_offset'] += 4
    try:
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        try:
            verify_evidence(folder, run['sanitizer'])
        except ValueError as exc:
            assert 'access report differs' in str(exc)
        else:
            raise AssertionError('mutated access report accepted')
    finally:
        path.write_bytes(original)
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--sanitizer', choices=('racecheck', 'synccheck'))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'examples/unified/access.py'
    config = output / 'monitor.json'
    config.write_text(json.dumps(configuration(source)))
    results = []
    for shift in (0, 1):
        folder = output / f'shift-{shift}'
        command = [sys.executable, '-m', 'tilelang_debugger', 'run', str(source), '--monitor', str(config), '--output', str(folder)]
        if args.sanitizer:
            command += ['--sanitizer', args.sanitizer]
        command += ['--', '--shift', str(shift)]
        with (output / f'shift-{shift}.log').open('w') as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=300).returncode
        result = dict(shift=shift, returncode=code, passed=False)
        if code == 0:
            result.update(requests=verify(folder, shift), passed=True)
            reference = output / f'reference-{shift}.py'
            reference.write_text('''import torch
def reference(inputs, points):
    assert [p['id'] for p in points] == ['value']
    x = inputs[0]
    samples = []
    for tx in range(32):
        for k in range(2):
            index = k * 32 + tx + SHIFT
            samples.append(dict(thread=tx, visit=k, coordinates=[k], ordinals=[k+1], index=0, value=int(x[index]) if index < 49 else 0))
    y = torch.zeros((64,), dtype=torch.int32)
    y[:32] = x[SHIFT:32+SHIFT]
    y[32:49] = x[32:49]
    return dict(points={'value': dict(schema=3, key='execution', dtype='int32', samples=samples, atol=0, rtol=0)}, arguments_after=[dict(tensor=x, atol=0, rtol=0)], outputs=[dict(tensor=y, atol=0, rtol=0)])
'''.replace('SHIFT', str(shift)))
            command = [sys.executable, '-m', 'tilelang_debugger', 'analyze', str(folder), '--reference', str(reference), '--output', str(output / f'analysis-{shift}')]
            with (output / f'analysis-{shift}.log').open('w') as log:
                result['analysis_returncode'] = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=180).returncode
            result['passed'] &= result['analysis_returncode'] == 0
        results.append(result)
        (output / 'summary.json').write_text(json.dumps(results, indent=2))
        print(result, flush=True)
    return int(any(not r['passed'] for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
