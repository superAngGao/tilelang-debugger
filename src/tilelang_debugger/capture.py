"""One compilation and launch per isolated worker; pre-launch verification gates."""
import ctypes
import importlib
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import uuid

from .instrument import digest, inject, prepare
from .records import parse, write_records


def save_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def snapshot(tensors, folder, prefix):
    import torch
    if isinstance(tensors, torch.Tensor):
        tensors = (tensors,)
    result = []
    for i, tensor in enumerate(tensors):
        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError("only contiguous CUDA tensor arguments/outputs are supported")
        raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        name = f"{prefix}-{i}.bin"
        (folder / name).write_bytes(raw)
        result.append(dict(file=name, shape=list(tensor.shape), stride=list(tensor.stride()),
                           dtype=str(tensor.dtype).removeprefix("torch."), bytes=len(raw), sha256=digest(raw)))
    save_json(folder / f"{prefix}.json", result)
    return result


def printf_capacity():
    # CUDA driver API avoids resolving a second, incompatible libcudart.
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuCtxSetLimit.argtypes = [ctypes.c_int, ctypes.c_size_t]
    cuda.cuCtxGetLimit.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
    size = ctypes.c_size_t()
    status = cuda.cuCtxSetLimit(1, 64 * 1024 * 1024)
    status_get = cuda.cuCtxGetLimit(ctypes.byref(size), 1)
    if status or status_get:
        raise RuntimeError(f"failed to configure CUDA printf FIFO: CUDA statuses {status}/{status_get}")
    if size.value < 64 * 1024 * 1024:
        raise RuntimeError("CUDA printf FIFO is below capture budget")
    return size.value


def configure_runtime(folder):
    import torch
    import tilelang
    if tilelang.__version__ != "0.1.12" or "H200" not in torch.cuda.get_device_name(0):
        raise RuntimeError("this release is validated only for TileLang 0.1.12 / NVIDIA H200")
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.init()
    torch.empty(1, device="cuda")  # Make PyTorch's primary context current on this host thread.
    fifo = printf_capacity()
    print_mod = importlib.import_module("tilelang.language.print_op")
    helper_hash = digest(Path(print_mod.__file__).read_bytes())
    if helper_hash != "55d1d925f24f2f0567744191d1af1089da74dea70d62fb32fd52d37bd6800f5b":
        raise RuntimeError("unvalidated TileLang print helper revision")
    from tilelang.env import TILELANG_TEMPLATE_PATH
    headers = {"reduce.h": "d47d59137eb03b8f5076ec28e9767fd70885b354330badfd0e40bd4ff98c39bf",
               "intrin.h": "88a8b7ec73c833f94bde974932a515b715138ed155c9702b2d2cb3c02b1203ec",
               "barrier.h": "ec5df8a0cb627fdd169256e11bc0b7429d70f97d0859b8f6e11a0b5a6a54bf09"}
    actual_headers = {name: digest((Path(TILELANG_TEMPLATE_PATH) / "tl_templates" / "cuda" / name).read_bytes()) for name in headers}
    if actual_headers != headers:
        raise RuntimeError("unvalidated synchronization/reduction helper revision")
    nvcc = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin" / "nvcc"
    nvcc_version = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True, check=True).stdout.strip()
    save_json(folder / "environment.json", dict(tilelang=tilelang.__version__, tilelang_path=tilelang.__file__,
              print_helper_sha256=helper_hash, torch=torch.__version__, cuda=torch.version.cuda,
              device=torch.cuda.get_device_name(0), printf_fifo_bytes=fifo, cache_disabled=True, headers=actual_headers,
              nvcc=str(nvcc), nvcc_version=nvcc_version))
    return torch, tilelang


def worker(folder):
    # Set before importing TileLang; a fresh Python process alone doesn't prevent disk-cache hits.
    os.environ["TILELANG_DISABLE_CACHE"] = "1"
    folder = Path(folder).resolve()
    request = json.loads((folder / "request.json").read_text(encoding="utf-8"))
    os.environ["TLDBG_OUTPUT"] = str(folder)
    torch, tilelang = configure_runtime(folder)
    from . import ir, monitor
    mode, points, contract = request["mode"], request["points"], request["contract"]
    baseline = json.loads((folder.parent / "baseline" / "compile.json").read_text()) if mode == "instrumented" else None
    original_compile = tilelang.compile
    compiles = launches = 0

    def compile_wrapper(prim_func, *args, **kwargs):
        nonlocal compiles
        compiles += 1
        if compiles != 1 or args or "out_idx" not in kwargs:
            raise ValueError("exactly one compile with explicit keyword out_idx is required")
        kernel = original_compile(prim_func, **kwargs)
        data, events, variables, allocations, cuda_source = ir.export(kernel, prim_func, folder, kwargs["out_idx"])
        if any(int(data["thread_extent"][f"threadIdx.{axis}"]) != expected
               for axis, expected in zip("xyz", [contract["cta_threads"], 1, 1])):
            raise ValueError("actual launch thread extents differ from reviewed contract")
        if mode == "instrumented":
            gate = ir.check_instrumented(data, events, variables, allocations, cuda_source, baseline, points, contract["cta_threads"], contract.get("allow_auto_cta_change", False), monitor.buffer_bindings(), folder / "observed-layouts.json")
            save_json(folder / "launch-gate.json", gate)
        else:
            save_json(folder / "launch-gate.json", dict(passed=True, baseline=True))

        def launch(*inputs, **launch_kwargs):
            nonlocal launches
            launches += 1
            if launches != 1 or launch_kwargs:
                raise ValueError("exactly one positional-tensor launch is required")
            snapshot(inputs, folder, "inputs")
            if mode == "instrumented":
                current = json.loads((folder / "inputs.json").read_text())
                previous = json.loads((folder.parent / "baseline" / "inputs.json").read_text())
                if current != previous:
                    raise ValueError("actual input bytes/dtype/shape/stride differ from baseline")
            out = kernel(*inputs)
            torch.cuda.synchronize()
            snapshot(out, folder, "outputs")
            return out
        return launch

    tilelang.compile = compile_wrapper
    sys.path.insert(0, str(folder / "source"))
    try:
        if mode == "instrumented":
            with monitor.session(points) as bound:
                runpy.run_path(str(folder / "source" / "run.py"), run_name="__main__")
            save_json(folder / "points.json", list(bound.values()))
        else:
            runpy.run_path(str(folder / "source" / "run.py"), run_name="__main__")
        if compiles != 1 or launches != 1:
            raise ValueError("driver must compile and launch exactly once")
    finally:
        tilelang.compile = original_compile
        sys.path.pop(0)
    save_json(folder / "execution.json", dict(compiles=compiles, launches=launches, restored=tilelang.compile is original_compile))


def subprocess_worker(folder, timeout, sanitizer=None, command_name="_worker"):
    command = [sys.executable, "-m", "tilelang_debugger", command_name, str(folder)]
    if sanitizer:
        command = ["compute-sanitizer", "--tool", sanitizer, "--error-exitcode", "86",
                   "--log-file", str(folder / f"{sanitizer}.log")] + command
    save_json(folder / "command.json", command)
    with (folder / "stdout.log").open("w", encoding="utf-8") as out, (folder / "stderr.log").open("w", encoding="utf-8") as err:
        process = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
        try:
            status = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            save_json(folder / "process.json", dict(returncode=process.returncode, timeout=True))
            raise RuntimeError(f"worker timed out after {timeout}s: {folder}")
    save_json(folder / "process.json", dict(returncode=status, timeout=False))
    if status:
        raise RuntimeError(f"worker exited {status}; see {folder / 'stderr.log'}")


def run(driver, config_file, output, timeout=240, sanitizer=None, *, source_path=None):
    if sys.platform != "linux":
        raise RuntimeError("GPU workers currently require Linux; CPU source/record tests also run on Windows")
    driver, output = Path(driver).resolve(), Path(output).resolve()
    from .source import load
    source, config, provenance = load(config_file, source_path)
    contract, points = prepare(source, config, driver.read_bytes())
    for point in points:
        point["source_path"] = provenance["source_path"]
    output.mkdir(parents=True, exist_ok=False)
    run_id = uuid.uuid4().hex
    save_json(output / "monitor.json", config)
    save_json(output / "source.json", provenance)
    save_json(output / "run.json", dict(schema=1, run_id=run_id, status="running", source_sha256=digest(source.encode()),
                                       source_path=provenance["source_path"],
                                       driver_sha256=digest(driver.read_bytes()), sanitizer=sanitizer))
    try:
        for mode in ("baseline", "instrumented"):
            folder = output / mode
            stage = folder / "source"
            stage.mkdir(parents=True)
            (stage / "run.py").write_bytes(driver.read_bytes())
            if mode == "instrumented":
                baseline = json.loads((output / "baseline" / "compile.json").read_text())
                used = {x["id"] for x in baseline["barriers"]} | {0, 1, 2}  # AllReduce's vetted helper reserves phases 1/2.
                free = [i for i in range(15, 0, -1) if i not in used]
                if len(free) < len(points):
                    raise ValueError("not enough unused named barriers")
                for point, barrier in zip(points, free):
                    point["barrier"] = barrier
                staged_source = inject(source, points)
            else:
                staged_source = source
            (stage / "kernel.py").write_text(staged_source, encoding="utf-8")
            save_json(folder / "request.json", dict(mode=mode, points=points, contract=contract, source_path=provenance["source_path"]))
            subprocess_worker(folder, timeout, sanitizer)
        baseline_outputs = json.loads((output / "baseline" / "outputs.json").read_text())
        instrumented_outputs = json.loads((output / "instrumented" / "outputs.json").read_text())
        if baseline_outputs != instrumented_outputs:
            raise ValueError("instrumentation changed output bits/dtype/shape/stride")
        bound = json.loads((output / "instrumented" / "points.json").read_text())
        records = parse((output / "instrumented" / "stdout.log").read_text(), bound)
        write_records(output / "records.jsonl", records)
        save_json(output / "points.json", bound)
        numerical = [json.loads((output / mode / "reference.json").read_text(encoding="utf-8")).get("passed")
                     for mode in ("baseline", "instrumented")]
        if any(type(passed) is not bool for passed in numerical):
            raise ValueError("driver did not produce a valid output reference result")
        result = dict(schema=1, run_id=run_id, status="passed", records=len(records), inputs_equal=True, outputs_bitwise_equal=True,
                      source_path=provenance["source_path"],
                      source_sha256=digest(source.encode()), driver_sha256=digest(driver.read_bytes()), sanitizer=sanitizer,
                      numerical_status="passed" if all(numerical) else "failed")
        save_json(output / "run.json", result)
        return result
    except Exception as exc:
        save_json(output / "run.json", dict(schema=1, run_id=run_id, status="failed", sanitizer=sanitizer, source_path=provenance["source_path"], error=str(exc)))
        raise
