"""Sequential multi-launch capture with typed arguments and mutation snapshots."""
import copy
import ctypes
import importlib
import inspect
import json
import os
from pathlib import Path
import runpy
import struct
import sys
import uuid

from ..capture import configure_runtime, save_json, subprocess_worker
from ..instrument import digest
from ..source_capture import export
from ..source import load
from ..source_analysis.control_flow import prepare
from ..instrumentation.unified import inject
from ..protocols.unified import parse
from ..records import write_records
from .source_loader import source_loader
from . import builds
from .parameters import parameters, bind
from .point_bindings import resolve as resolve_points


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save_values(values, folder, prefix):
    import torch
    groups, result = {}, []
    for i, value in enumerate(values):
        if isinstance(value, torch.Tensor):
            if not value.is_cuda:
                raise ValueError('GPU capture arguments need CUDA tensors')
            storage = value.untyped_storage()
            group = groups.setdefault((str(value.device), storage.data_ptr()), len(groups))
            raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
            filename = f'{prefix}-{i}.bin'
            (folder / filename).write_bytes(raw)
            result.append(dict(kind='tensor', shape=list(value.shape), stride=list(value.stride()),
                               dtype=str(value.dtype).removeprefix('torch.'), file=filename, bytes=len(raw), sha256=digest(raw),
                               alias_group=group, storage_offset=value.storage_offset(), storage_bytes=storage.nbytes()))
        elif type(value) in (int, float, bool):
            item = dict(kind='scalar', dtype=type(value).__name__, value=value.hex() if type(value) is float else value)
            if type(value) is float:
                item['bits'] = struct.pack('>d', value).hex()
            result.append(item)
        else:
            raise ValueError(f'unsupported runtime argument: {type(value).__name__}')
    save_json(folder / f'{prefix}.json', result)
    return result


def worker(folder):
    os.environ['TILELANG_DISABLE_CACHE'] = '1'
    folder = Path(folder).resolve()
    request = read(folder / 'request.json')
    os.environ['TLDBG_OUTPUT'] = str(folder)
    from ..frontend.compatibility import check
    compatibility = check()
    save_json(folder / 'compatibility.json', compatibility)
    torch, tilelang = configure_runtime(folder, strict=False)
    # Record the exact installed product used by this GPU worker.
    import tilelang_debugger
    environment = read(folder / 'environment.json')
    environment['debugger_path'] = str(Path(tilelang_debugger.__file__).resolve())
    save_json(folder / 'environment.json', environment)
    jit = importlib.import_module('tilelang.jit')
    original, original_jit = tilelang.compile, jit.compile
    signature = inspect.signature(original)
    original_path, original_argv = list(sys.path), list(sys.argv)
    compiles, launches = [], []
    registry = None
    source, driver = Path(request['source_path']), Path(request['driver'])
    staged = (folder / 'source' / 'instrumented.py').read_text(encoding='utf-8')
    imported = {'restored': False, 'loads': 0}
    instrumented = request['mode'] == 'instrumented'

    def compile_wrapper(prim_func, *args, **kwargs):
        bound_args = signature.bind(prim_func, *args, **kwargs)
        out_idx = bound_args.arguments.get('out_idx')
        if prim_func.attrs and 'tilelang_out_idx' in prim_func.attrs:
            out_idx = [int(i) for i in prim_func.attrs['tilelang_out_idx']]
        count = len(prim_func.params)
        outputs = [] if out_idx is None else [out_idx] if type(out_idx) is int else list(out_idx)
        if any(type(i) is not int or not -count <= i < count for i in outputs):
            raise ValueError('invalid output parameter indices')
        outputs = [i % count for i in outputs]
        if len(outputs) != len(set(outputs)):
            raise ValueError('duplicate output indices')
        inputs = [i for i in range(count) if i not in outputs]
        build_id = int(prim_func.attrs['tldbg_build_id']) if prim_func.attrs and 'tldbg_build_id' in prim_func.attrs else None
        if build_id is not None and build_id not in registry['builds']:
            raise ValueError('unknown frontend build identity')
        points = copy.deepcopy(registry['builds'][build_id]) if build_id is not None else []
        cid = len(compiles)
        where = folder / 'compiles' / str(cid)
        where.mkdir(parents=True)
        kernel = original(prim_func, *args, **kwargs)
        export(kernel, prim_func, where, outputs)
        metadata = dict(compile=cid, build=build_id, outputs=outputs,
                        parameters=parameters(prim_func),
                        point_ids=[p['id'] for p in points])
        compiles.append(metadata)
        save_json(where / 'identity.json', metadata)
        save_json(where / 'points.json', points)
        save_json(where / 'artifacts.json', {name: digest((where / name).read_bytes()) for name in ('frontend.py', 'frontend.json', 'device.py', 'device.json', 'kernel.cu', 'compile.json')})

        class Proxy:
            def __getattr__(self, name):
                return getattr(kernel, name)

            def __call__(self, *values, **launch_kwargs):
                names = [str(prim_func.buffer_map[prim_func.params[i]].name) if prim_func.params[i] in prim_func.buffer_map else str(prim_func.params[i].name) for i in inputs]
                call_signature = inspect.Signature([inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in names])
                arguments = call_signature.bind(*values, **launch_kwargs).arguments
                values = tuple(arguments[name] for name in names)
                if torch.cuda.is_current_stream_capturing():
                    raise ValueError('host snapshots cannot run inside CUDA Graph capture')
                for value, i in zip(values, inputs):
                    param = prim_func.params[i]
                    if param in prim_func.buffer_map:
                        buf = prim_func.buffer_map[param]
                        if not isinstance(value, torch.Tensor) or str(value.dtype).removeprefix('torch.') != str(buf.dtype):
                            raise ValueError('tensor argument differs from frontend shape/dtype')
                    elif type(value) not in (int, float, bool):
                        raise ValueError('scalar parameter needs a Python numeric value')
                    else:
                        dtype = str(param.dtype)
                        if dtype.startswith(('int', 'uint')):
                            width = int(dtype.removeprefix('uint').removeprefix('int'))
                            lower = 0 if dtype.startswith('uint') else -(2**(width - 1))
                            upper = 2**width if dtype.startswith('uint') else 2**(width - 1)
                            if type(value) is not int or not lower <= value < upper:
                                raise ValueError('scalar integer outside frontend dtype')
                        elif dtype == 'bool' and type(value) is not bool:
                            raise ValueError('scalar bool requires bool')
                        elif dtype.startswith('float') and type(value) is not float:
                            raise ValueError('scalar floating parameter requires float')
                lid = len(launches)
                where = folder / 'launches' / str(lid)
                where.mkdir(parents=True)
                torch.cuda.synchronize()
                before = save_values(values, where, 'before')
                symbol_bindings = bind(metadata['parameters'], before, inputs)
                if instrumented:
                    save_json(where / 'points.json', resolve_points(points, symbol_bindings))
                if instrumented and before != read(folder.parent / 'baseline' / 'launches' / str(lid) / 'before.json'):
                    raise ValueError('baseline/instrumented inputs or alias structure differ')
                ctypes.CDLL(None).fflush(None)
                print(f'TLHOST3|{lid}|B|{cid}|Z', flush=True)
                output = kernel(*values)
                torch.cuda.synchronize()
                ctypes.CDLL(None).fflush(None)
                print(f'TLHOST3|{lid}|E|{cid}|Z', flush=True)
                returned = [] if output is None else [output] if isinstance(output, torch.Tensor) else list(output)
                save_values([*values, *returned], where, 'after')
                entry = dict(launch=lid, compile=cid, arguments=len(values), returned=len(returned))
                entry['symbols'] = symbol_bindings
                entry['stream'] = 'caller_current_stream'
                launches.append(entry)
                save_json(where / 'identity.json', entry)
                return output
        return Proxy()

    tilelang.compile = jit.compile = compile_wrapper
    sys.path.insert(0, str(driver.parent))
    sys.argv[:] = [str(driver), *request['driver_args']]
    try:
        if digest(source.read_text(encoding='utf-8').encode()) != request['source_sha256'] or digest(driver.read_bytes()) != request['driver_sha256']:
            raise ValueError('source/driver changed')
        with builds.session(request['points'], instrumented) as registry, source_loader(source, staged) as imported:
            try:
                if source == driver:
                    imported['loads'] += 1
                    exec(compile(staged, str(driver), 'exec'), dict(__name__='__main__', __file__=str(driver), __package__=None, __spec__=None))
                else:
                    runpy.run_path(str(driver), run_name='__main__')
            except SystemExit as exc:
                if exc.code is not None and exc.code != 0:
                    raise
            if not imported['loads'] or not compiles or not launches:
                raise ValueError('selected source must be built and launched')
        if not (folder / 'reference.json').exists():
            save_json(folder / 'reference.json', dict(status='not_provided', passed=None))
        if digest(source.read_text(encoding='utf-8').encode()) != request['source_sha256'] or digest(driver.read_bytes()) != request['driver_sha256']:
            raise ValueError('source/driver changed during execution')
    finally:
        tilelang.compile, jit.compile = original, original_jit
        sys.path[:], sys.argv[:] = original_path, original_argv
        built = [] if registry is None else [dict(build=i, point_ids=[p['id'] for p in ps]) for i, ps in registry['builds'].items()]
        save_json(folder / 'execution.json', dict(builds=built, compiles=compiles, launches=launches, restored=imported.get('restored', False)))


def split_launches(log, launches):
    expected = {r['launch']: r for r in launches}
    result, current, ended = {}, None, set()
    for line in log.splitlines():
        if line.startswith('TLHOST3|'):
            parts = line.split('|')
            if len(parts) != 5 or parts[-1] != 'Z':
                raise ValueError('malformed host launch boundary')
            _, lid, kind, cid, _ = parts
            lid, cid = int(lid), int(cid)
            if lid not in expected or cid != expected[lid]['compile']:
                raise ValueError('host boundary identity mismatch')
            if kind == 'B' and current is None and lid not in result and lid == len(result):
                current = lid
                result[lid] = []
            elif kind == 'E' and current == lid:
                current = None
                ended.add(lid)
            else:
                raise ValueError('duplicate/out-of-order host boundary')
        elif 'TLDBG' in line:
            if current is None:
                raise ValueError('device event outside a launch boundary')
            result[current].append(line)
    if current is not None or set(result) != set(expected) or ended != set(expected):
        raise ValueError('incomplete launch boundary sequence')
    return {k: '\n'.join(v) for k, v in result.items()}


def run(driver, config_file, output, timeout=240, sanitizer=None, *, source_path=None, driver_args=()):
    if sys.platform != 'linux':
        raise RuntimeError('GPU capture requires Linux')
    source, config, provenance = load(config_file, source_path)
    points = prepare(source, config)
    driver, output = Path(driver).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema='source-unified-v3', run_id=uuid.uuid4().hex, status='running', source_path=config['source'],
                  source_sha256=digest(source.encode()), driver_sha256=digest(driver.read_bytes()), sanitizer=sanitizer,
                  staged_sha256={}, plan_sha256=digest(json.dumps(points, sort_keys=True).encode()),
                  config_sha256=digest(json.dumps(config, sort_keys=True).encode()))
    save_json(output / 'run.json', result)
    save_json(output / 'monitor.json', config)
    save_json(output / 'source.json', provenance)
    try:
        for mode in ('baseline', 'instrumented'):
            folder = output / mode
            (folder / 'source').mkdir(parents=True)
            (folder / 'source' / 'original.py').write_text(source, encoding='utf-8')
            (folder / 'source' / 'driver.py').write_bytes(driver.read_bytes())
            staged = inject(source, points, enabled=mode == 'instrumented')
            (folder / 'source' / 'instrumented.py').write_text(staged, encoding='utf-8')
            result['staged_sha256'][mode] = digest(staged.encode())
            save_json(folder / 'request.json', dict(result, mode=mode, driver=str(driver), driver_args=list(driver_args), points=points))
            subprocess_worker(folder, timeout, sanitizer, '_unified_worker')
        executions = [read(output / mode / 'execution.json') for mode in ('baseline', 'instrumented')]
        if executions[0] != executions[1] or not executions[0]['restored']:
            raise ValueError('compile/launch sequences differ or hook not restored')
        execution = executions[1]
        logs = split_launches((output / 'instrumented' / 'stdout.log').read_text(), execution['launches'])
        records, coverage, contracts = [], {}, []
        for launch in execution['launches']:
            lid, cid = launch['launch'], launch['compile']
            for kind in ('before', 'after'):
                if read(output / 'baseline' / 'launches' / str(lid) / f'{kind}.json') != read(output / 'instrumented' / 'launches' / str(lid) / f'{kind}.json'):
                    raise ValueError('baseline/instrumented parameter mutation/output differs')
            bound = read(output / 'instrumented' / 'launches' / str(lid) / 'points.json')
            data, complete = parse(logs[lid], bound, launch=lid, compile_id=cid)
            records.extend(data)
            coverage[str(lid)] = complete
            contracts.append(dict(launch=lid, compile=cid, points=bound))
        save_json(output / 'points.json', contracts)
        write_records(output / 'records.jsonl', records)
        refs = [read(output / mode / 'reference.json').get('passed') for mode in ('baseline', 'instrumented')]
        numerical = 'failed' if False in refs else 'passed' if all(v is True for v in refs) else 'not_checked'
        missing = sorted({p['id'] for p in points} - {p['id'] for c in contracts for p in c['points']})
        partial = bool(missing) or any(p['capture_integrity'] != 'complete' for c in coverage.values() for p in c.values())
        lifecycle = {p['id']: dict(built=any(p['id'] in b['point_ids'] for b in execution['builds']),
                                  compiled=any(p['id'] in c['point_ids'] for c in execution['compiles']),
                                  launched=p['id'] not in missing) for p in points}
        result.update(status='partial' if partial else 'passed', coverage=coverage, records=len(records),
                      numerical_status=numerical, inputs_equal=True, outputs_bitwise_equal=True,
                      unlaunched_points=missing, point_lifecycle=lifecycle,
                      configured_points=[p['id'] for p in points], launched_points=sorted({p['id'] for c in contracts for p in c['points']}))
        save_json(output / 'run.json', result)
        if any(p.get('mode') == 'access' for p in points):
            from ..protocols.access import report, markdown
            summary, requests = report(records, contracts, coverage)
            save_json(output / 'access-summary.json', summary)
            write_records(output / 'accesses.jsonl', requests)
            (output / 'access-report.md').write_text(markdown(summary), encoding='utf-8')
        from .unified_evidence import verify
        verify(output, sanitizer)
        return result
    except BaseException as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        save_json(output / 'run.json', result)
        raise
