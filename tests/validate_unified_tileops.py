"""Real upstream operators, with the existing independent CPU references."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tileops', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sanitizer', choices=('racecheck', 'synccheck'))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    nested = ROOT / 'examples/tileops/nested'
    cfg_module = module(nested / 'configure.py', 'fixture_config')
    jobs = []
    provider = (nested / 'reference.py').read_text().replace('def reference(inputs, points):', 'def old_reference(inputs, points):')
    provider += '''
def reference(inputs, points):
    old = old_reference(inputs, points)
    for p in points:
        spec = old['points'][p['id']]
        spec['schema'] = 3
        for row in spec['samples']:
            cursor = 0
            row['ordinals'] = []
            for scope in p['scopes']:
                if 'aliases' not in scope:
                    continue
                # These pinned fixtures use zero-based serial loops. This is
                # reference knowledge of the examples, not observed DATA.
                row['ordinals'].append(0 if scope['kind'] == 'parallel' else row['coordinates'][cursor] + 1)
                cursor += len(scope['aliases'])
    old['arguments_after'] = [dict(tensor=t.clone(), atol=0, rtol=0) for t in inputs]
    return old
'''
    for case in ('pool', 'indices', 'rope', 'softmax'):
        config = cfg_module.config(args.tileops, case)
        config['schema'] = 3
        jobs.append((case, config, nested / 'run.py', ['--tileops', args.tileops, '--case', case], provider))
    sys.path.insert(0, str(ROOT / 'examples/tileops'))
    capture = module(ROOT / 'examples/tileops/capture.py', 'capture_fixture')
    from common import cases
    case = next(c for c in cases() if c['id'] == 'rms-n257-float16')
    cfg = capture.configuration(case, Path(args.tileops))
    cfg['schema'] = 3
    for point in cfg['points']:
        point['collective'] = dict(ready=True, uniform=True)
    rms_provider = 'import sys\nsys.path.insert(0, ' + repr(str(ROOT / 'examples/tileops')) + ')\nfrom capture_reference import reference_case\n'
    rms_provider += '''
def reference(inputs, points):
    old = reference_case(inputs, points, CASE)
    specs = {}
    for p in points:
        spec = old['points'][p['id']]
        values = spec['tensor'].reshape(-1).tolist()
        specs[p['id']] = dict(schema=3, key='logical', dtype=p['dtype'], atol=spec['atol'], rtol=spec['rtol'],
            samples=[dict(coordinates=[], ordinals=[], index=i, value=v) for i,v in enumerate(values)])
    return dict(points=specs, arguments_after=[dict(tensor=t.clone(), atol=0, rtol=0) for t in inputs], outputs=old['outputs'])
'''.replace('CASE', repr(case))
    jobs.append(('rms', cfg, ROOT / 'examples/tileops/rms_norm/run.py', ['--tileops', args.tileops, '--case', case['id'], '--output', str(output / 'rms-driver')], rms_provider))
    results = []
    for case, cfg, driver, flags, code in jobs:
        config_path, reference = output / (case + '.json'), output / (case + '-reference.py')
        config_path.write_text(json.dumps(cfg))
        reference.write_text(code)
        command = [sys.executable, '-m', 'tilelang_debugger', 'run', str(driver), '--monitor', str(config_path), '--output', str(output / case)]
        if args.sanitizer:
            command += ['--sanitizer', args.sanitizer]
        commands = [command + ['--', *flags], [sys.executable, '-m', 'tilelang_debugger', 'analyze', str(output / case), '--reference', str(reference), '--output', str(output / (case + '-analysis'))]]
        codes = []
        for i, cmd in enumerate(commands):
            with (output / f'{case}-{i}.log').open('w') as stream:
                codes.append(subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, timeout=600).returncode)
            if codes[-1]:
                break
        result = dict(case=case, passed=codes == [0, 0], returncodes=codes)
        results.append(result)
        print(json.dumps(result), flush=True)
        (output / 'summary.json').write_text(json.dumps(results, indent=2))
    return int(any(not r['passed'] for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
