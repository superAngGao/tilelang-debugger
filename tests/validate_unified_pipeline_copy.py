import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--sanitizer', choices=('racecheck', 'synccheck'), default='racecheck')
    parser.add_argument('--mode', choices=('active', 'mixed', 'inactive'), default='mixed')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'examples/unified/pipeline_copy.py'
    line = next(i for i, text in enumerate(source.read_text().splitlines(), 1) if 'T.copy(fragment, y' in text)
    config = dict(schema=3, source=str(source), points=[dict(id='copy', line=line, when='before', buffer='fragment', block=[0, 0, 0], loops=[], collective=dict(ready=True, uniform=True))])
    if args.mode != 'active':
        if args.mode == 'inactive':
            config['points'] = []
        for name in ('unused', 'unused_after'):
            line = next(i for i, text in enumerate(source.read_text().splitlines(), 1) if f'{name} =' in text)
            config['points'].append(dict(id=name, line=line, when='after', buffer=name, block=[0, 0, 0], loops=[]))
    (output / 'monitor.json').write_text(json.dumps(config))
    commands = [[sys.executable, '-m', 'tilelang_debugger', 'run', str(source), '--monitor', str(output / 'monitor.json'), '--output', str(output / 'capture'), '--sanitizer', args.sanitizer],
                [sys.executable, '-m', 'tilelang_debugger', 'analyze', str(output / 'capture'), '--reference', str(ROOT / 'examples/unified/pipeline_copy_reference.py'), '--output', str(output / 'analysis')]]
    codes = []
    for i, cmd in enumerate(commands):
        with (output / f'command-{i}.log').open('w') as stream:
            codes.append(subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, timeout=600).returncode)
        if codes[-1]:
            break
    result = dict(passed=codes == [0, 0], returncodes=codes)
    (output / 'summary.json').write_text(json.dumps(result))
    print(json.dumps(result))
    return int(not result['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
