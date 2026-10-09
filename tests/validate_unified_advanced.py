import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def config(case, dtype='int32', scope='scalar', variant=None):
    source = ROOT / 'examples/unified/advanced_kernel.py'
    lines = source.read_text().splitlines()
    marker = {'group_fragment': 'T.copy(fragment, y[0:128])', 'pipeline_fragment': 'T.copy(fragment, y[0:32])',
              'nested': 'nested_value =', 'repeated': 'repeated_value =', 'final_return': 'final_value =', 'alias': 'alias_value ='}
    if case == 'types':
        needle = 'typed_value =' if scope == 'scalar' else 'y[tx] = x[tx]'
        index = {'uint64': 0, 'int64': 1, 'float64': 2}.get(dtype, 3) if scope == 'scalar' else {'fragment': 1, 'shared': 2, 'local': 3}[scope]
        line = [i for i, text in enumerate(lines, 1) if needle in text][index]
        buffer = 'typed_value' if scope == 'scalar' else 'typed_buffer'
        when = 'after' if scope == 'scalar' else 'before'
    else:
        line = next(i for i, text in enumerate(lines, 1) if marker[case] in text)
        buffer = 'fragment' if case in {'group_fragment', 'pipeline_fragment'} else case.removesuffix('_return') + '_value'
        when = 'before' if case in {'group_fragment', 'pipeline_fragment'} else 'after'
    p = dict(id=case, line=line, when=when, buffer=buffer, block=[0, 0, 0], loops=[])
    if case in {'group_fragment', 'pipeline_fragment'} or case == 'types' and scope in {'fragment', 'shared'}:
        p['collective'] = dict(ready=True, uniform=True)
    if case == 'group_fragment':
        p['collective']['barrier'] = 15
        p['collective']['reader'] = 129
    if variant == 'default-reader':
        p['collective'].pop('reader')
    if variant == 'xyz-reader':
        p['collective']['reader'] = [129, 0, 0]
    if variant == 'reject-whole-thread':
        p['thread'] = 129
    if variant == 'reject-reader-cta':
        p['collective']['reader'] = 256
    if variant == 'reject-reader-group':
        p['collective']['reader'] = 0
    if variant == 'reject-direct-collective':
        p['collective'] = dict(ready=True)
    if variant == 'selected':
        p['thread'] = 4
        p['loops'] = [dict(line=i, iteration=2) for i, text in enumerate(lines, 1) if 'for outer' in text or 'for inner' in text]
    if variant == 'not-executed':
        p['thread'] = 1
    if variant == 'oob':
        p['buffer'] = 'x[300]'
    result = dict(schema=3, source=str(source), points=[p])
    if variant == 'truncated':
        result['budget'] = 64 if case == 'pipeline_fragment' else 96
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--types', action='store_true')
    parser.add_argument('--sanitizer', choices=('racecheck', 'synccheck'))
    parser.add_argument('--scope', action='append')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.types:
        cases = [('types', dtype, scope, None) for scope in args.scope or ('scalar', 'local', 'fragment', 'shared')
                 for dtype in ('bool', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'float16', 'bfloat16', 'float32', 'float64')]
    else:
        cases = [(case, 'int32', 'scalar', None) for case in ('group_fragment', 'pipeline_fragment', 'nested', 'repeated', 'final_return', 'alias')]
        cases += [('nested', 'int32', 'scalar', variant) for variant in ('selected', 'not-executed')]
        cases += [('repeated', 'int32', 'scalar', 'truncated'), ('pipeline_fragment', 'int32', 'scalar', 'truncated'), ('final_return', 'int32', 'scalar', 'oob')]
        cases += [('group_fragment', 'int32', 'scalar', variant) for variant in ('default-reader', 'xyz-reader', 'reject-whole-thread', 'reject-reader-cta', 'reject-reader-group')]
        cases += [('final_return', 'int32', 'scalar', 'reject-direct-collective')]
    results = []
    for case, dtype, scope, variant in cases:
        label = '-'.join(str(v) for v in (case, dtype, scope, variant) if v is not None)
        cfg = output / (label + '.json')
        cfg.write_text(json.dumps(config(case, dtype, scope, variant)))
        cmd = [sys.executable, '-m', 'tilelang_debugger', 'run', str(ROOT / 'examples/unified/advanced_run.py'), '--monitor', str(cfg), '--output', str(output / label)]
        if args.sanitizer:
            cmd += ['--sanitizer', args.sanitizer]
        cmd += ['--', '--case', case, '--dtype', dtype, '--scope', scope]
        with (output / (label + '.log')).open('w') as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=600).returncode
        rejected = bool(variant and variant.startswith('reject-'))
        expected = 3 if variant == 'truncated' else 1 if variant == 'oob' or rejected else 0
        passed = code == expected
        if passed and variant != 'oob' and not rejected:
            cmd = [sys.executable, '-m', 'tilelang_debugger', 'analyze', str(output / label), '--reference', str(ROOT / 'examples/unified/advanced_reference.py'), '--output', str(output / (label + '-analysis'))]
            with (output / (label + '-analysis.log')).open('w') as log:
                code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=180).returncode
            passed = code == (2 if variant == 'truncated' else 0)
        if passed and variant == 'oob':
            run = json.loads((output / label / 'run.json').read_text())
            passed = 'out-of-bounds read' in run.get('error', '')
        if passed and rejected:
            errors = {'reject-whole-thread': 'whole-buffer observation selects logical elements',
                      'reject-reader-cta': 'thread selection outside frontend CTA',
                      'reject-reader-group': 'reader outside group',
                      'reject-direct-collective': 'scalar/local/element observation uses thread'}
            folder = output / label / 'instrumented'
            message = '\n'.join((folder / name).read_text() for name in ('stdout.log', 'stderr.log'))
            passed = errors[variant] in message
        result = dict(case=label, passed=passed, returncode=code)
        results.append(result)
        (output / 'summary.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(result), flush=True)
    return int(any(not r['passed'] for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
