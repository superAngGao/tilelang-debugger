"""Check access operands in pipeline/group contexts against independent sets."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def configuration(case, selected):
    source = ROOT / 'examples/unified' / ('pipeline_copy.py' if case == 'copy' else 'kernel.py')
    lines = source.read_text().splitlines()
    specs = [('read', case + '_value =', 'x')] if case != 'copy' else [
        ('global_read', 'T.copy(x[k', 'x'),
        ('shared_write', 'T.copy(x[k', 'shared'),
        ('shared_read', 'T.copy(shared, fragment)', 'shared'),
        ('fragment_write', 'T.copy(shared, fragment)', 'fragment'),
        ('global_write', 'T.copy(fragment, y', 'y')]
    points = []
    for pid, needle, buffer in specs:
        point = dict(id=pid, line=next(i for i, s in enumerate(lines, 1) if needle in s),
                     when='before', mode='access', buffer=buffer, block=[0, 0, 0], loops=[])
        if selected:
            point['thread'] = 137 if case == 'group' else 7
        points.append(point)
    return dict(schema=3, source=str(source), points=points)


def verify(folder, case, selected):
    from tilelang_debugger.runtime.unified_evidence import verify as evidence
    result = json.loads((folder / 'run.json').read_text())
    assert result['status'] == 'passed' and result['numerical_status'] == 'passed', result
    evidence(folder, result['sanitizer'])
    rows = [json.loads(s) for s in (folder / 'accesses.jsonl').read_text().splitlines()]
    threads = [137 if case == 'group' else 7] if selected else range(128, 256) if case == 'group' else range(32)
    iterations = range(2, 7) if case == 'pipeline' else range(3) if case == 'group' else range(4)
    launches = range(1 if case == 'copy' else 2)
    pids = ['read'] if case != 'copy' else ['global_read', 'shared_write', 'shared_read', 'fragment_write', 'global_write']
    expected = Counter((launch, pid, tx, k) for launch in launches for pid in pids for tx in threads for k in iterations)
    actual = Counter((r['launch'], r['point'], r['thread'], r['coordinates'][0]) for r in rows)
    assert actual == expected, (actual - expected, expected - actual)
    for row in rows:
        k = row['coordinates'][0]
        tile = case == 'copy'
        global_region = row['point'] in {'global_read', 'global_write'}
        assert row['origin'] == [k * 64 if global_region else 0 if tile else row['thread']], row
        assert row['extent'] == [64 if tile else 1], row
        assert row['shape'] == [256 if global_region or not tile else 64], row
        assert row['stride'] == [1] and row['active'] and row['status'] == 'in_bounds', row
        assert row['ordinals'] == [k - (2 if case == 'pipeline' else 0) + 1], row
        assert row['operation'] == ('write' if row['point'].endswith('write') else 'read'), row
        if row['point'].startswith('shared'):
            assert row['memory_scope'] in {'shared', 'shared.dyn'}, row
        elif row['point'].startswith('fragment'):
            assert row['memory_scope'] == 'local.fragment', row
        else:
            assert row['memory_scope'] == 'global', row
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--sanitizer', choices=('racecheck', 'synccheck'), default='racecheck')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for case in ('pipeline', 'group', 'copy'):
        for selected in (False, True):
            name = case + ('-selected' if selected else '-all')
            config = configuration(case, selected)
            cfg = output / (name + '.json')
            cfg.write_text(json.dumps(config))
            driver = ROOT / 'examples/unified' / ('pipeline_copy.py' if case == 'copy' else 'run.py')
            command = [sys.executable, '-m', 'tilelang_debugger', 'run', str(driver), '--monitor', str(cfg),
                       '--output', str(output / name), '--sanitizer', args.sanitizer]
            if case != 'copy':
                command += ['--', '--case', case, '--launches', '2']
            with (output / (name + '.log')).open('w') as log:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600).returncode
            result = dict(case=name, returncode=code, passed=False)
            if code == 0:
                try:
                    result.update(requests=verify(output / name, case, selected), passed=True)
                except Exception as exc:
                    result['error'] = repr(exc)
            results.append(result)
            (output / 'summary.json').write_text(json.dumps(results, indent=2))
            print(json.dumps(result), flush=True)
    return int(any(not r['passed'] for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
