"""Validate capture evidence and link saved analysis without executing reference code."""
import json
from pathlib import Path

from ..analysis import manifest
from ..runtime.unified_evidence import verify
from ..diagnostics.values import identity
from ..protocols.unified import value
from ..numerics import json_number
from ..numerics import scalar_error
from ..diagnostics.values import decode


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def load(capture, analysis=None):
    capture = Path(capture).resolve()
    run = read(capture / 'run.json')
    if run.get('schema') != 'source-unified-v3':
        raise ValueError('HTML reports currently require a schema 3 capture')
    hashes = manifest(capture)
    contracts = verify(capture, run['sanitizer'])
    records = rows(capture / 'records.jsonl')
    comparisons, result, analysis_hashes = {}, None, None
    if analysis:
        analysis = Path(analysis).resolve()
        analysis_hashes = manifest(analysis)
        result = read(analysis / 'analysis.json')
        if result.get('schema') != 'unified-analysis-v3' or result.get('status') != 'completed' or result.get('run_id') != run['run_id']:
            raise ValueError('analysis must be completed and belong to this capture')
        if read(analysis / 'evidence.json') != hashes:
            raise ValueError('analysis evidence does not match this capture')
        expected_complete = run['status'] == 'passed' and set(run['configured_points']) <= set(run['launched_points'])
        if result['capture_complete'] != expected_complete or [l['launch'] for l in result['launches']] != [c['launch'] for c in contracts]:
            raise ValueError('analysis launch coverage differs from capture')
        for launch, contract in zip(result['launches'], contracts):
            if [p['point'] for p in launch['comparisons']] != [p['id'] for p in contract['points'] if p.get('mode', 'value') == 'value']:
                raise ValueError('analysis observation coverage differs from capture')
        actual = {identity(r): r for r in records}
        for row in rows(analysis / 'elements.jsonl'):
            key = identity(row)
            if key in comparisons or key not in actual:
                raise ValueError('analysis sample identity differs from capture')
            record = actual[key]
            if any(row[k] != v for k, v in record.items()) or row['actual'] != json_number(value(record['bits'], record['dtype'])):
                raise ValueError('analysis actual values differ from capture')
            comparisons[key] = row
        expected_keys = {identity(r) for c in contracts for p in c['points'] if p.get('mode', 'value') == 'value'
                         for r in records if r['launch'] == c['launch'] and r['point'] == p['id']}
        if set(comparisons) != expected_keys:
            raise ValueError('analysis does not cover all observed numeric samples')
        for launch in result['launches']:
            for comparison in launch['comparisons']:
                selected = [r for r in comparisons.values() if r['launch'] == launch['launch'] and r['point'] == comparison['point']]
                mismatches = sum(r['matched'] is False for r in selected) + len(comparison['missing_keys'])
                if comparison['observed_records'] != len(selected) or comparison['mismatches'] != mismatches or comparison['matched'] != (mismatches == 0):
                    raise ValueError('analysis summary differs from sample outcomes')
                if 'atol' in comparison and 'rtol' in comparison:
                    for r in selected:
                        if r['expected'] is None:
                            continue
                        a, e = decode(r['actual']), decode(r['expected'])
                        matched = a == e if r['dtype'].startswith(('int', 'uint', 'bool')) else scalar_error(a, e, comparison['atol'], comparison['rtol'])[0]
                        if matched != r['matched']:
                            raise ValueError('analysis match differs from saved tolerances')
        matched = result['capture_complete'] and all(l['arguments_outputs_matched'] and all(p['matched'] for p in l['comparisons']) for l in result['launches'])
        if result['matched'] != matched:
            raise ValueError('analysis aggregate differs from saved comparisons')
    return dict(capture=capture, analysis_path=analysis, run=run, contracts=contracts, records=records,
                comparisons=comparisons, analysis=result, capture_hashes=hashes, analysis_hashes=analysis_hashes,
                accesses=rows(capture / 'accesses.jsonl') if (capture / 'accesses.jsonl').exists() else [],
                source=(capture / 'baseline/source/original.py').read_text(encoding='utf-8'),
                monitor=read(capture / 'monitor.json'),
                environment=read(capture / 'instrumented/environment.json'),
                execution=read(capture / 'instrumented/execution.json'))
