"""Offline verification of every launch, raw snapshot and protocol packet."""
import json
import math
import re
import struct

from ..instrument import digest
from ..protocols.dtypes import UNIFIED_WIDTH
from ..protocols.unified import parse
from .unified_capture import read, split_launches


def snapshots(folder, prefix):
    result = read(folder / f'{prefix}.json')
    if not isinstance(result, list):
        raise ValueError('snapshot list missing')
    groups = {}
    for i, item in enumerate(result):
        if item['kind'] == 'scalar':
            if item['dtype'] not in {'int', 'float', 'bool'}:
                raise ValueError('unknown saved scalar type')
            if item['dtype'] == 'float':
                raw = bytes.fromhex(item['bits'])
                if len(raw) != 8 or struct.unpack('>d', raw)[0].hex() != item['value']:
                    raise ValueError('scalar floating bits differ')
            elif type(item['value']).__name__ != item['dtype']:
                raise ValueError('saved scalar type differs')
            continue
        if item['kind'] != 'tensor' or item['file'] != f'{prefix}-{i}.bin':
            raise ValueError('invalid snapshot identity')
        raw = (folder / item['file']).read_bytes()
        width = max(8, UNIFIED_WIDTH[item['dtype']])
        shape, stride = item['shape'], item['stride']
        if not isinstance(shape, list) or not isinstance(stride, list) or len(shape) != len(stride) or any(type(x) is not int or x < 0 for x in shape + stride):
            raise ValueError('invalid tensor shape/stride')
        expected_stride = 1
        for size, actual_stride in reversed(list(zip(shape, stride))):
            if math.prod(shape) and size > 1 and actual_stride != expected_stride:
                raise ValueError('saved tensor is not contiguous')
            expected_stride *= size
        if any(type(item[k]) is not int or item[k] < 0 for k in ('alias_group', 'storage_offset', 'storage_bytes')):
            raise ValueError('invalid tensor storage metadata')
        group = item['alias_group']
        if group not in groups:
            if group != len(groups):
                raise ValueError('noncanonical storage alias group')
            groups[group] = item['storage_bytes']
        if groups[group] != item['storage_bytes'] or (item['storage_offset'] + math.prod(shape)) * width // 8 > item['storage_bytes']:
            raise ValueError('tensor view outside declared storage')
        if len(raw) != item['bytes'] or len(raw) != math.prod(item['shape']) * width // 8 or digest(raw) != item['sha256']:
            raise ValueError('snapshot bytes differ from metadata')
    return result


def verify(folder, sanitizer=None):
    run = read(folder / 'run.json')
    if run['status'] not in {'passed', 'partial'} or run['sanitizer'] != sanitizer or not run.get('inputs_equal') or not run.get('outputs_bitwise_equal'):
        raise ValueError('capture not completed under requested mode')
    if digest(json.dumps(read(folder / 'monitor.json'), sort_keys=True).encode()) != run['config_sha256']:
        raise ValueError('saved observation configuration differs')
    versions = []
    for mode in ('baseline', 'instrumented'):
        worker = folder / mode
        execution = read(worker / 'execution.json')
        if not execution['restored'] or not execution['launches'] or not execution['compiles']:
            raise ValueError('incomplete worker lifecycle')
        if read(worker / 'process.json') != dict(returncode=0, timeout=False):
            raise ValueError('worker failed or timed out')
        source = worker / 'source'
        if digest((source / 'original.py').read_text(encoding='utf-8').encode()) != run['source_sha256'] or digest((source / 'driver.py').read_bytes()) != run['driver_sha256']:
            raise ValueError('source/driver snapshot provenance differs')
        request = read(worker / 'request.json')
        if request['mode'] != mode or request['source_sha256'] != run['source_sha256'] or request['driver_sha256'] != run['driver_sha256']:
            raise ValueError('worker request provenance differs')
        if digest(json.dumps(request['points'], sort_keys=True).encode()) != run['plan_sha256'] or digest((source / 'instrumented.py').read_text(encoding='utf-8').encode()) != run['staged_sha256'][mode]:
            raise ValueError('saved instrumentation/plan hash differs')
        for cid, compilation in enumerate(execution['compiles']):
            where = worker / 'compiles' / str(cid)
            if compilation['compile'] != cid or read(where / 'identity.json') != compilation:
                raise ValueError('compile identity differs')
            build = compilation['build']
            if build is not None and (not 0 <= build < len(execution['builds']) or execution['builds'][build]['build'] != build or execution['builds'][build]['point_ids'] != compilation['point_ids']):
                raise ValueError('compile differs from frontend build lifecycle')
            if build is None and compilation['point_ids']:
                raise ValueError('observed compile lacks frontend identity')
            artifacts = read(where / 'artifacts.json')
            names = {'frontend.py', 'frontend.json', 'device.py', 'device.json', 'kernel.cu', 'compile.json'}
            if set(artifacts) != names or any(digest((where / name).read_bytes()) != artifacts[name] for name in names):
                raise ValueError('compile artifact missing/changed')
            if read(where / 'compile.json')['out_idx'] != compilation['outputs']:
                raise ValueError('compile output indices differ')
            points = read(where / 'points.json')
            if [p['id'] for p in points] != compilation['point_ids']:
                raise ValueError('compile point identities differ')
        if sanitizer:
            if read(worker / 'command.json')[:3] != ['compute-sanitizer', '--tool', sanitizer]:
                raise ValueError('sanitizer command differs')
            log = (worker / f'{sanitizer}.log').read_text()
            expected = r'RACECHECK SUMMARY: 0 hazards displayed \(0 errors, 0 warnings\)' if sanitizer == 'racecheck' else r'ERROR SUMMARY: 0 errors'
            if not re.search(expected, log) or 'Target application returned an error' in log:
                raise ValueError('sanitizer did not complete cleanly')
        samples = []
        for i, launch in enumerate(execution['launches']):
            if launch['launch'] != i or not 0 <= launch['compile'] < len(execution['compiles']):
                raise ValueError('invalid launch manifest')
            where = worker / 'launches' / str(i)
            if read(where / 'identity.json') != launch:
                raise ValueError('launch identity differs')
            before, after = [snapshots(where, prefix) for prefix in ('before', 'after')]
            compilation = execution['compiles'][launch['compile']]
            parameters, outputs = compilation['parameters'], compilation['outputs']
            inputs = [i for i in range(len(parameters)) if i not in outputs]
            if len(before) != launch['arguments'] or len(before) != len(inputs) or len(after) != launch['arguments'] + launch['returned'] or launch['returned'] != len(outputs):
                raise ValueError('snapshot cardinality differs from launch/compile contract')
            for items, roles in ((before, inputs), (after, inputs + outputs)):
                for item, index in zip(items, roles):
                    param = parameters[index]
                    if item['kind'] != param['kind']:
                        raise ValueError('snapshot role differs from parameter')
                    if item['kind'] == 'tensor' and (item['dtype'] != param['dtype'] or item['shape'] != param['shape']):
                        raise ValueError('tensor snapshot differs from frontend parameter')
                    if item['kind'] == 'scalar':
                        expected_type = 'bool' if param['dtype'] == 'bool' else 'float' if param['dtype'].startswith('float') else 'int'
                        if item['dtype'] != expected_type:
                            raise ValueError('scalar snapshot differs from frontend parameter')
            samples.append([before, after])
        logs = split_launches((worker / 'stdout.log').read_text(), execution['launches'])
        if mode == 'baseline' and any(logs.values()):
            raise ValueError('baseline contains debug packets')
        versions.append((execution, samples))
    if versions[0] != versions[1]:
        raise ValueError('persisted launches/parameters/outputs differ')
    execution = versions[1][0]
    contracts, records, coverage = [], [], {}
    for launch in execution['launches']:
        lid, cid = launch['launch'], launch['compile']
        points = read(folder / 'instrumented' / 'compiles' / str(cid) / 'points.json')
        contracts.append(dict(launch=lid, compile=cid, points=points))
        rows, states = parse(logs[lid], points, launch=lid, compile_id=cid)
        records.extend(rows)
        coverage[str(lid)] = states
    saved = [json.loads(line) for line in (folder / 'records.jsonl').read_text().splitlines()]
    if contracts != read(folder / 'points.json') or records != saved or len(records) != run['records'] or coverage != run['coverage']:
        raise ValueError('persisted protocol records/coverage differ')
    configured = [p['id'] for p in read(folder / 'monitor.json')['points']]
    launched = sorted({p['id'] for contract in contracts for p in contract['points']})
    missing = sorted(set(configured) - set(launched))
    if run['configured_points'] != configured or run['launched_points'] != launched or run['unlaunched_points'] != missing:
        raise ValueError('point lifecycle summary differs')
    lifecycle = {pid: dict(built=any(pid in b['point_ids'] for b in execution['builds']),
                           compiled=any(pid in c['point_ids'] for c in execution['compiles']),
                           launched=pid not in missing) for pid in configured}
    if lifecycle != run['point_lifecycle']:
        raise ValueError('point lifecycle differs from worker manifest')
    partial = bool(missing) or any(p['capture_integrity'] != 'complete' for c in coverage.values() for p in c.values())
    if run['status'] != ('partial' if partial else 'passed'):
        raise ValueError('partial capture incorrectly reported')
    refs = [read(folder / mode / 'reference.json').get('passed') for mode in ('baseline', 'instrumented')]
    numerical = 'failed' if False in refs else 'passed' if all(v is True for v in refs) else 'not_checked'
    if run['numerical_status'] != numerical:
        raise ValueError('numerical status differs from driver evidence')
    return contracts
