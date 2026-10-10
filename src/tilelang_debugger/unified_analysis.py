"""Independent per-launch reference comparison for TLDBG3 captures."""
import copy
import json
from pathlib import Path

from .numerics import finite_tolerance, scalar_error, json_number
from .protocols.dtypes import UNIFIED_WIDTH
from .protocols.unified import value


def compare_samples(records, point, spec):
    if not isinstance(spec, dict) or set(spec) != {'schema', 'key', 'dtype', 'samples', 'atol', 'rtol'} or spec['schema'] != 3 or spec['key'] not in {'logical', 'execution'}:
        raise ValueError('unified reference requires schema=3, key, dtype, samples, atol, rtol')
    dtype = point.get('dtype') or spec['dtype']
    if dtype not in UNIFIED_WIDTH or dtype != spec['dtype']:
        raise ValueError('reference dtype differs')
    atol, rtol = finite_tolerance(spec['atol']), finite_tolerance(spec['rtol'])
    integer = dtype == 'bool' or dtype.startswith(('int', 'uint'))
    if integer and (atol or rtol):
        raise ValueError('integer comparison requires zero tolerance')

    def key(row):
        if not isinstance(row.get('coordinates'), list) or len(row['coordinates']) != point['coordinate_count'] or not isinstance(row.get('ordinals'), list) or len(row['ordinals']) != point['ordinal_count'] or any(type(v) is not int for v in row['coordinates'] + row['ordinals']) or type(row.get('index')) is not int:
            raise ValueError('invalid reference coordinate key')
        result = (tuple(row['coordinates']), tuple(row['ordinals']), row['index'])
        if spec['key'] == 'execution':
            if type(row.get('thread')) is not int or type(row.get('visit')) is not int or row['thread'] < 0 or row['visit'] < 0:
                raise ValueError('execution reference requires thread and visit')
            result += (row['thread'], row['visit'])
        return result

    expected = {}
    if not isinstance(spec['samples'], list):
        raise ValueError('samples must independently declare expected key/value pairs')
    for row in spec['samples']:
        fields = {'coordinates', 'ordinals', 'index', 'value'} | ({'thread', 'visit'} if spec['key'] == 'execution' else set())
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError('invalid reference sample fields')
        k = key(row)
        if k in expected:
            raise ValueError('duplicate reference key')
        v = row['value']
        width = UNIFIED_WIDTH[dtype]
        if dtype == 'bool':
            valid = type(v) is bool
        elif integer:
            minimum = 0 if dtype.startswith('uint') else -(2**(width - 1))
            maximum = 2**width if minimum == 0 else 2**(width - 1)
            valid = type(v) is int and minimum <= v < maximum
        else:
            valid = type(v) in (int, float)
        if not valid:
            raise ValueError('reference value does not fit declared dtype')
        expected[k] = v
    found = {key(row) for row in records}
    missing, unexpected = set(expected) - found, found - set(expected)
    rows, failed = [], len(missing) + len(unexpected)
    for record in records:
        k = key(record)
        actual, reference = value(record['bits'], dtype), expected.get(k)
        if k not in expected:
            matched, error = False, None
        elif integer:
            matched, error = actual == reference, abs(int(actual) - int(reference))
        else:
            matched, error, _, _ = scalar_error(actual, reference, atol, rtol)
        failed += int(k in expected and not matched)
        rows.append(dict(record, actual=json_number(actual), expected=json_number(reference), matched=matched, abs_error=json_number(error)))
    return dict(point=point['id'], matched=not failed, mismatches=failed, expected_keys=len(expected), observed_records=len(records),
                dtype=dtype, atol=atol, rtol=rtol,
                missing_keys=[repr(k) for k in sorted(missing)], unexpected_keys=[repr(k) for k in sorted(unexpected)]), rows


def load_values(folder, prefix):
    import torch
    import struct
    values = []
    from .runtime.tensor_views import restore_storage
    items = json.loads((folder / f'{prefix}.json').read_text())
    raw_storages = restore_storage(items, folder)
    storages = {group: torch.frombuffer(raw, dtype=torch.uint8).clone().untyped_storage() if raw else torch.empty(0, dtype=torch.uint8).untyped_storage() for group, raw in raw_storages.items()}
    for item in items:
        if item['kind'] == 'scalar':
            values.append(struct.unpack('>d', bytes.fromhex(item['bits']))[0] if item['dtype'] == 'float' else item['value'])
        else:
            group = item['alias_group']
            dtype = getattr(torch, item['dtype'])
            tensor = torch.empty(0, dtype=dtype).set_(storages[group], item['storage_offset'], item['shape'], item['stride'])
            values.append(tensor)
    return values


def compare_value(actual, spec):
    import torch
    if isinstance(actual, torch.Tensor):
        if not isinstance(spec, dict) or set(spec) != {'tensor', 'atol', 'rtol'}:
            raise ValueError('tensor reference needs tensor, atol, rtol')
        expected = spec['tensor']
        if not isinstance(expected, torch.Tensor) or expected.device.type != 'cpu' or actual.shape != expected.shape:
            raise ValueError('reference needs exact-shape CPU tensor')
        atol, rtol = finite_tolerance(spec['atol']), finite_tolerance(spec['rtol'])
        if not actual.is_floating_point():
            if expected.dtype != actual.dtype or atol or rtol:
                raise ValueError('integer tensor needs identical dtype and zero tolerance')
            return torch.equal(actual, expected)
        if not expected.is_floating_point():
            raise ValueError('floating tensor needs floating reference')
        return all(scalar_error(a, e, atol, rtol)[0] for a, e in zip(actual.reshape(-1).tolist(), expected.reshape(-1).tolist()))
    return type(actual) is type(spec) and actual == spec


def analyze(capture, reference, output):
    from .analysis import load_provider, manifest, save
    from .runtime.unified_capture import read
    from .runtime.unified_evidence import verify
    capture, reference, output = (Path(p).resolve() for p in (capture, reference, output))
    if output == capture or capture in output.parents:
        raise ValueError('analysis must be outside capture')
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(schema='unified-analysis-v3', status='running')
    save(output / 'analysis.json', summary)
    try:
        run, evidence = read(capture / 'run.json'), manifest(capture)
        contracts = verify(capture, run['sanitizer'])
        records = [json.loads(line) for line in (capture / 'records.jsonl').read_text().splitlines()]
        provider_source = reference.read_bytes()
        (output / 'reference.py').write_bytes(provider_source)
        results, rows = [], []
        with load_provider(provider_source, reference) as provider:
            for contract in contracts:
                lid = contract['launch']
                folder = capture / 'baseline' / 'launches' / str(lid)
                inputs, actual_after = load_values(folder, 'before'), load_values(folder, 'after')
                # Access operands already have a validated logical-range report.
                # Adding an access point must not require inventing a numerical
                # reference for metadata or changing the user's value provider.
                points = copy.deepcopy([p for p in contract['points'] if p.get('mode', 'value') == 'value'])
                for p in points:
                    p.update(launch=lid, compile=contract['compile'])
                bundle = provider(tuple(copy.deepcopy(inputs)), points)
                if not isinstance(bundle, dict) or set(bundle) != {'points', 'arguments_after', 'outputs'} or set(bundle['points']) != {p['id'] for p in points}:
                    raise ValueError('reference must declare points, arguments_after and outputs per launch')
                after = [*bundle['arguments_after'], *bundle['outputs']]
                if len(bundle['arguments_after']) != len(inputs) or len(after) != len(actual_after):
                    raise ValueError('reference must cover every argument and output')
                comparisons = []
                for p in points:
                    report, values = compare_samples([r for r in records if r['launch'] == lid and r['point'] == p['id']], p, bundle['points'][p['id']])
                    comparisons.append(report)
                    rows.extend(values)
                matched = all(compare_value(a, e) for a, e in zip(actual_after, after))
                results.append(dict(launch=lid, comparisons=comparisons, arguments_outputs_matched=matched))
        if manifest(capture) != evidence:
            raise ValueError('capture changed during analysis')
        complete = run['status'] == 'passed' and set(run['configured_points']) <= set(run['launched_points'])
        summary.update(status='completed', run_id=run['run_id'], coverage=run['coverage'], capture_complete=complete,
                       matched=complete and all(r['arguments_outputs_matched'] and all(p['matched'] for p in r['comparisons']) for r in results), launches=results)
        if any(p.get('mode') == 'access' for c in contracts for p in c['points']):
            summary['source_access'] = read(capture / 'access-summary.json')
        save(output / 'evidence.json', evidence)
        (output / 'elements.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in rows))
        lines = ['# Unified capture analysis', '', f"Matched: {summary['matched']}; capture complete: {complete}.", '',
                 '| Launch | Point | Records | Missing keys | Mismatches |', '| ---: | --- | ---: | ---: | ---: |']
        for launch in results:
            for p in launch['comparisons']:
                lines.append(f"| {launch['launch']} | {p['point']} | {p['observed_records']} | {len(p['missing_keys'])} | {p['mismatches']} |")
        if 'source_access' in summary:
            lines += ['', 'Source access operands were checked for capture integrity and logical bounds separately. Numerical reference matching does not prove index correctness. See access-report.md and accesses.jsonl in the capture.']
        (output / 'report.md').write_text('\n'.join(lines) + '\n')
        save(output / 'analysis.json', summary)
        return summary
    except Exception as exc:
        summary.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        save(output / 'analysis.json', summary)
        raise
