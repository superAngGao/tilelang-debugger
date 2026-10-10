import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    folder = Path(args.output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'examples/unified/runtime_expansion.py'
    lines = source.read_text().splitlines()
    point = next(i for i, line in enumerate(lines, 1) if 'result =' in line)
    whole = next(i for i, line in enumerate(lines, 1) if 'T.sync_threads()' in line)
    config = dict(schema=3, source=str(source), points=[
        dict(id='scalar', line=point, when='after', buffer='result', block=[0, 0, 0], loops=[]),
        dict(id='access', line=point, when='before', buffer='x', mode='access', block=[0, 0, 0], loops=[]),
        dict(id='whole', line=whole, when='after', buffer='y', block=[0, 0, 0], loops=[], collective=dict(ready=True))])
    cfg = folder / 'monitor.json'
    cfg.write_text(json.dumps(config))
    summary = []
    for case in ('contiguous', 'strided', 'expanded'):
        cmd = [sys.executable, '-m', 'tilelang_debugger', 'run', str(source), '--monitor', str(cfg), '--output', str(folder / case), '--sanitizer', 'racecheck', '--', '--case', case]
        with (folder / f'{case}.log').open('w') as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=300).returncode
        if code == 0:
            rows = [json.loads(line) for line in (folder / case / 'accesses.jsonl').read_text().splitlines()]
            for launch, n in enumerate((17, 39)):
                selected = [r for r in rows if r['launch'] == launch]
                assert sorted(r['origin'][0] for r in selected) == list(range(n))
                assert all(r['shape'] == [n] and r['stride'] == [{'contiguous': 1, 'strided': 2, 'expanded': 0}[case]] and r['status'] == 'in_bounds' for r in selected)
            cmd = [sys.executable, '-m', 'tilelang_debugger', 'analyze', str(folder / case), '--reference', str(ROOT / 'examples/unified/runtime_expansion_reference.py'), '--output', str(folder / (case + '-analysis'))]
            with (folder / f'{case}-analysis.log').open('w') as log:
                code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=180).returncode
        summary.append(dict(case=case, passed=code == 0, returncode=code))
        (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(summary[-1], flush=True)
    return int(any(not r['passed'] for r in summary))


if __name__ == '__main__':
    raise SystemExit(main())
