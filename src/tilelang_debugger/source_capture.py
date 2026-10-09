"""Source-selected capture. Compilation is delegated unchanged to TileLang."""
from contextlib import nullcontext
import importlib
import json
import os
from pathlib import Path
import runpy
import sys
import uuid

from .capture import configure_runtime, save_json, snapshot, subprocess_worker
from .instrument import digest
from .source_instrument import inject, prepare, source_loader
from .records import parse, write_records

SCHEMA = "source-capture-v1"


def output_indices(value, count):
    indices = [value] if type(value) is int else value
    if not isinstance(indices, (list, tuple)) or not indices or any(type(i) is not int or not -count <= i < count for i in indices):
        raise ValueError("nonempty explicit out_idx int/list with valid indices is required")
    indices = [i % count for i in indices]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate output indices")
    return indices


def export(kernel, prim_func, folder, outputs):
    """Save compiler products verbatim; do not interpret device IR/layouts."""
    import tvm
    if kernel.artifact is None or kernel.artifact.device_mod is None:
        raise ValueError("compiled artifact missing; cache must be disabled")
    for name, obj in (("frontend", prim_func), ("device", kernel.artifact.device_mod)):
        (folder / f"{name}.py").write_text(obj.script(), encoding="utf-8")
        (folder / f"{name}.json").write_text(tvm.ir.save_json(obj), encoding="utf-8")
    (folder / "kernel.cu").write_text(kernel.get_kernel_source(), encoding="utf-8")
    save_json(folder / "compile.json", dict(target=str(kernel.target), execution_backend=kernel.execution_backend,
              pass_configs={str(k): str(v) for k, v in (kernel.pass_configs or {}).items()},
              compile_flags=list(kernel.compile_flags or []), out_idx=outputs))


def tensor_arguments(tensors, buffers):
    import torch
    if isinstance(tensors, torch.Tensor):
        tensors = (tensors,)
    if len(tensors) != len(buffers):
        raise ValueError("actual tensor count differs from frontend parameters")
    for tensor, buf in zip(tensors, buffers):
        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError("only contiguous CUDA tensors are supported")
        if list(tensor.shape) != [int(s) for s in buf.shape] or str(tensor.dtype).removeprefix("torch.") != str(buf.dtype):
            raise ValueError("actual tensor shape/dtype differs from frontend buffer")


def no_storage_alias(inputs, selected):
    ranges = [(t.untyped_storage().data_ptr(), t.untyped_storage().nbytes()) for t in inputs]
    for i in selected:
        a, n = ranges[i]
        for j, (b, m) in enumerate(ranges):
            if i != j and a < b + m and b < a + n:
                raise ValueError("observed global input shares storage with another argument")


def worker(folder):
    os.environ["TILELANG_DISABLE_CACHE"] = "1"
    folder = Path(folder).resolve()
    request = json.loads((folder / "request.json").read_text(encoding="utf-8"))
    os.environ["TLDBG_OUTPUT"] = str(folder)
    torch, tilelang = configure_runtime(folder)
    from . import monitor
    jit_module = importlib.import_module("tilelang.jit")
    original_compile, original_jit_compile = tilelang.compile, jit_module.compile
    saved_path, saved_argv = list(sys.path), list(sys.argv)
    compiles = launches = 0
    points = request["points"]
    mode = request["mode"]
    source_path, driver = Path(request["source_path"]), Path(request["driver"])
    original_source = (folder / "source" / "original.py").read_text(encoding="utf-8")
    staged = (folder / "source" / "instrumented.py").read_text(encoding="utf-8") if mode == "instrumented" else original_source
    imported = {"loads": 0, "restored": False}
    bound = {}

    def compile_wrapper(prim_func, *args, **kwargs):
        nonlocal compiles
        compiles += 1
        if compiles != 1 or args or "out_idx" not in kwargs or not imported["loads"]:
            raise ValueError("one compile with explicit out_idx after selected source import is required")
        buffers = []
        for var in prim_func.params:
            if var not in prim_func.buffer_map:
                raise ValueError("scalar kernel parameters are not supported")
            buf = prim_func.buffer_map[var]
            if any(int(s) <= 0 for s in buf.shape):
                raise ValueError("positive static buffer shapes required")
            buffers.append(buf)
        outs = output_indices(kwargs["out_idx"], len(buffers))
        ins = [i for i in range(len(buffers)) if i not in outs]
        globals_selected = []
        if mode == "instrumented":
            if any(not p.get("root_built" if p.get("schema") == 2 else "bound") for p in bound.values()):
                raise ValueError("selected source point was not constructed by this kernel")
            bindings = monitor.buffer_bindings()
            for p in bound.values():
                if p["scope"] == "global":
                    matches = [i for i in ins if buffers[i].same_as(bindings[p["id"]])]
                    if len(matches) != 1:
                        raise ValueError("global observation must bind one readonly input parameter")
                    p["input_index"] = ins.index(matches[0])
                    globals_selected.append(p["input_index"])
            save_json(folder / "points.json", list(bound.values()))
        kernel = original_compile(prim_func, **kwargs)
        export(kernel, prim_func, folder, outs)
        save_json(folder / "launch-gate.json", dict(passed=True, engine="source", checks="source scope, frontend buffer and launch arguments; no lowered IR proof"))

        class Proxy:
            def __getattr__(self, name):
                return getattr(kernel, name)

            def __call__(self, *inputs, **launch_kwargs):
                nonlocal launches
                launches += 1
                if launches != 1 or launch_kwargs:
                    raise ValueError("one positional tensor launch required")
                tensor_arguments(inputs, [buffers[i] for i in ins])
                no_storage_alias(inputs, globals_selected)
                snapshot(inputs, folder, "inputs")
                if mode == "instrumented" and (folder / "inputs.json").read_bytes() != (folder.parent / "baseline" / "inputs.json").read_bytes():
                    raise ValueError("baseline/instrumented inputs differ")
                out = kernel(*inputs)
                torch.cuda.synchronize()
                tensor_arguments(out, [buffers[i] for i in outs])
                snapshot(out, folder, "outputs")
                return out
        return Proxy()

    tilelang.compile = jit_module.compile = compile_wrapper
    sys.path.insert(0, str(driver.parent))
    sys.argv[:] = [str(driver), *request["driver_args"]]
    try:
        if digest(source_path.read_text(encoding="utf-8").encode()) != request["source_sha256"] or digest(driver.read_bytes()) != request["driver_sha256"]:
            raise ValueError("source/driver changed since request")
        with source_loader(source_path, staged) as imported:
            with monitor.session(points) if mode == "instrumented" else nullcontext({}) as bound:
                try:
                    if source_path == driver:
                        imported["loads"] += 1
                        exec(compile(staged, str(driver), "exec"), dict(__name__="__main__", __file__=str(driver), __package__=None, __spec__=None))
                    else:
                        runpy.run_path(str(driver), run_name="__main__")
                except SystemExit as exc:
                    if exc.code is not None and exc.code != 0:
                        raise
                if imported["loads"] != 1 or compiles != 1 or launches != 1:
                    raise ValueError("driver must load selected source, compile and launch exactly once")
        reference = folder / "reference.json"
        if not reference.exists():
            save_json(reference, dict(status="not_provided", passed=None))
        if digest(source_path.read_text(encoding="utf-8").encode()) != request["source_sha256"] or digest(driver.read_bytes()) != request["driver_sha256"]:
            raise ValueError("source/driver changed during execution")
    finally:
        tilelang.compile, jit_module.compile = original_compile, original_jit_compile
        sys.path[:], sys.argv[:] = saved_path, saved_argv
        restored = tilelang.compile is original_compile and jit_module.compile is original_jit_compile and imported.get("restored", False)
        save_json(folder / "execution.json", dict(compiles=compiles, launches=launches, restored=restored))


def run(driver, config_file, output, timeout=240, sanitizer=None, *, source_path=None, driver_args=()):
    if sys.platform != "linux":
        raise RuntimeError("GPU source capture requires Linux")
    from .source import load
    from .evidence import reference_status, verify_capture
    source, config, provenance = load(config_file, source_path)
    points = prepare(source, config)
    driver, output = Path(driver).resolve(), Path(output).resolve()
    driver_bytes = driver.read_bytes()
    output.mkdir(parents=True, exist_ok=False)
    schema = "source-samples-v2" if config.get("schema") == 2 else SCHEMA
    result = dict(schema=schema, engine="source", run_id=uuid.uuid4().hex, status="running",
                  source_path=config["source"], source_sha256=digest(source.encode()), driver_sha256=digest(driver_bytes), sanitizer=sanitizer)
    save_json(output / "run.json", result)
    save_json(output / "monitor.json", config)
    provenance["worker_snapshot"] = "source/original.py"
    save_json(output / "source.json", provenance)
    try:
        for mode in ("baseline", "instrumented"):
            folder = output / mode
            (folder / "source").mkdir(parents=True)
            (folder / "source" / "driver.py").write_bytes(driver_bytes)
            (folder / "source" / "original.py").write_text(source, encoding="utf-8")
            if mode == "instrumented":
                (folder / "source" / "instrumented.py").write_text(inject(source, points), encoding="utf-8")
            save_json(folder / "request.json", dict(result, mode=mode, points=points, driver=str(driver), driver_args=list(driver_args)))
            subprocess_worker(folder, timeout, sanitizer, "_source_worker")
        for prefix in ("inputs", "outputs"):
            if (output / "baseline" / f"{prefix}.json").read_bytes() != (output / "instrumented" / f"{prefix}.json").read_bytes():
                raise ValueError(f"baseline/instrumented {prefix} differ")
        bound = json.loads((output / "instrumented" / "points.json").read_text())
        log = (output / "instrumented" / "stdout.log").read_text()
        if schema == "source-samples-v2":
            from .sample_records import parse as parse_samples
            records, coverage = parse_samples(log, bound)
            result["coverage"] = coverage
        else:
            records = parse(log, bound)
        write_records(output / "records.jsonl", records)
        save_json(output / "points.json", bound)
        refs = [json.loads((output / m / "reference.json").read_text()) for m in ("baseline", "instrumented")]
        result.update(status="passed", records=len(records), inputs_equal=True, outputs_bitwise_equal=True,
                      numerical_status=reference_status(refs, schema))
        save_json(output / "run.json", result)
        verify_capture(output, sanitizer)
        return result
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save_json(output / "run.json", result)
        raise
