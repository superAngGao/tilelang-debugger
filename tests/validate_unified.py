"""Run the public unified capture matrix against independent exact references."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CASES = ('parallel', 'pipeline', 'group', 'two_dim', 'dynamic', 'while', 'negative', 'fragment', 'shared', 'global', 'inplace')


def config(case):
    source = ROOT / 'examples/unified/kernel.py'
    lines = source.read_text().splitlines()
    marker = {'two_dim': 'local[1] =', 'fragment': 'T.copy(fragment,', 'shared': 'T.copy(shared,',
              'global': 'y[tx] = y[tx]', 'inplace': 'updated ='}
    needle = marker.get(case, case + '_value =')
    number = next(i for i, line in enumerate(lines, 1) if needle in line)
    buffer = {'two_dim': 'local', 'fragment': 'fragment', 'shared': 'shared', 'global': 'y', 'inplace': 'updated'}.get(case, case + '_value')
    p = dict(id=case, line=number, when='before' if case in {'fragment', 'shared'} else 'after',
             buffer=buffer, block=[0, 0, 0], loops=[])
    if case in {'fragment', 'shared', 'global'}:
        p.update(collective=dict(ready=True, uniform=True), region=[[0, 32, 2]])
    return dict(schema=3, source=str(source), points=[p])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--case', action='append')
    parser.add_argument('--sanitizer', choices=('racecheck', 'synccheck'))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for case in args.case or CASES:
        cfg = output / (case + '.json')
        cfg.write_text(json.dumps(config(case)))
        cmd = [sys.executable, '-m', 'tilelang_debugger', 'run', str(ROOT / 'examples/unified/run.py'), '--monitor', str(cfg), '--output', str(output / case)]
        if args.sanitizer:
            cmd += ['--sanitizer', args.sanitizer]
        cmd += ['--', '--case', case, '--launches', '2']
        if case == 'inplace':
            cmd += ['--rebuild']
        with (output / (case + '.log')).open('w') as log:
            process = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        result = dict(case=case, returncode=process.returncode)
        if process.returncode == 0:
            command = [sys.executable, '-m', 'tilelang_debugger', 'analyze', str(output / case), '--reference', str(ROOT / 'examples/unified/reference.py'), '--output', str(output / (case + '-analysis'))]
            with (output / (case + '-analysis.log')).open('w') as log:
                result['returncode'] = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=180).returncode
        results.append(result)
        print(json.dumps(result), flush=True)
        (output / 'summary.json').write_text(json.dumps(results, indent=2))
    return int(any(r['returncode'] for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
