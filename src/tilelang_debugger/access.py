"""Isolated baseline and access workers for the reviewed H200 examples."""
import json
import os
from pathlib import Path
import runpy
import sys
import uuid
from .capture import configure_runtime, save_json, snapshot, subprocess_worker
from .access_contracts import prepare, pins
from .instrument import digest


def worker(folder):
    os.environ["TILELANG_DISABLE_CACHE"] = "1"
    folder = Path(folder).resolve()
    os.environ["TLDBG_OUTPUT"] = str(folder)
    request = json.loads((folder/"request.json").read_text())
    torch, tilelang = configure_runtime(folder)
    from . import access_ir
    original_compile = tilelang.compile
    compiles = launches = 0
    enabled = request["mode"] == "instrumented"
    def compile_wrapper(prim_func,*args,**kwargs):
        nonlocal compiles
        compiles += 1
        if compiles != 1 or args or "out_idx" not in kwargs:
            raise ValueError("one compile with explicit out_idx required")
        with access_ir.session(folder,request["points"],enabled) as state:
            kernel = original_compile(prim_func,**kwargs)
        if state["pipeline_calls"] != 1 or state["codegen_calls"] != 1 or kernel.execution_backend != "tvm_ffi":
            raise ValueError("unreviewed compilation path/backend")
        cuda = kernel.get_kernel_source()
        (folder/"kernel.cu").write_text(cuda,encoding="utf-8")
        (folder/"frontend.py").write_text(prim_func.script(),encoding="utf-8")
        import tvm
        (folder/"frontend.json").write_text(tvm.ir.save_json(prim_func),encoding="utf-8")
        if not enabled and digest(cuda.encode()) != pins()["cuda"][request["case"]]:
            raise ValueError("baseline CUDA differs from reviewed access contract")
        config = dict(backend=kernel.execution_backend,target=str(kernel.target),out_idx=kwargs["out_idx"],
                      pass_configs={str(k):str(v) for k,v in kernel.pass_configs.items()},compile_flags=list(kernel.compile_flags or []))
        save_json(folder/"compile.json",config)
        if enabled and config != json.loads((folder.parent/"baseline/compile.json").read_text()):
            raise ValueError("compilation settings differ")
        from .access_host import HostBindings
        host = HostBindings(state["module"],prim_func,kwargs["out_idx"],request["case"])
        save_json(folder/"launch-gate.json",dict(passed=True,baseline_cuda_authorized=True,ir_erasure_verified=True,
                  codegen_erasure_verified=state["codegen_erasure_verified"],host=host.evidence))
        def launch(*inputs,**launch_kwargs):
            nonlocal launches
            launches += 1
            if launches != 1 or launch_kwargs:
                raise ValueError("one positional tensor launch required")
            inp = snapshot(inputs,folder,"inputs")
            if enabled and inp != json.loads((folder.parent/"baseline/inputs.json").read_text()):
                raise ValueError("actual input bytes/shape/stride differ from baseline")
            host.bind_inputs(inputs)
            out = kernel(*inputs)
            torch.cuda.synchronize()
            snapshot(out,folder,"outputs")
            descriptors = host.finish(out)
            save_json(folder/"descriptors.json",descriptors)
            return out
        return launch
    tilelang.compile = compile_wrapper
    sys.path.insert(0,str(folder/"source"))
    try:
        runpy.run_path(str(folder/"source/run.py"),run_name="__main__")
        if compiles != 1 or launches != 1:
            raise ValueError("driver must compile and launch exactly once")
    finally:
        tilelang.compile = original_compile
        sys.path.pop(0)
    save_json(folder/"execution.json",dict(compiles=compiles,launches=launches,restored=tilelang.compile is original_compile))


def run(driver,config_file,output,timeout=240,sanitizer=None):
    if sys.platform != "linux":
        raise RuntimeError("access workers require Linux/H200")
    driver, output = Path(driver).resolve(), Path(output).resolve()
    source = driver.with_name("kernel.py").read_text(encoding="utf-8")
    config = json.loads(Path(config_file).read_text(encoding="utf-8"))
    case, contract, points = prepare(source,driver.read_bytes(),config)
    output.mkdir(parents=True,exist_ok=False)
    run_id = uuid.uuid4().hex
    info = dict(schema="TLACC1",run_id=run_id,case=case,source_sha256=digest(source.encode()),driver_sha256=digest(driver.read_bytes()),sanitizer=sanitizer)
    save_json(output/"access.json",config)
    save_json(output/"run.json",dict(info,status="running"))
    try:
        for mode in ("baseline","instrumented"):
            folder=output/mode
            (folder/"source").mkdir(parents=True)
            (folder/"source/run.py").write_bytes(driver.read_bytes())
            (folder/"source/kernel.py").write_text(source,encoding="utf-8")
            save_json(folder/"request.json",dict(mode=mode,case=case,points=points,contract=contract))
            subprocess_worker(folder,timeout,sanitizer,"_access_worker")
        for kind in ("inputs","outputs"):
            if json.loads((output/f"baseline/{kind}.json").read_text()) != json.loads((output/f"instrumented/{kind}.json").read_text()):
                raise ValueError(f"access instrumentation changed {kind} bits/shape/stride")
        from .access_records import parse, report, validate_workers
        validate_workers(output,sanitizer)
        manifest=json.loads((output/"instrumented/access-points.json").read_text())
        descriptors=json.loads((output/"instrumented/descriptors.json").read_text())
        records=parse((output/"instrumented/stdout.log").read_text(),manifest,run_id)
        save_json(output/"access-points.json",manifest)
        (output/"access-records.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
        analysis, markdown=report(records,manifest,descriptors)
        save_json(output/"access-analysis.json",analysis)
        (output/"access-report.md").write_text(markdown,encoding="utf-8")
        numerical=[json.loads((output/f"{m}/reference.json").read_text()).get("passed") for m in ("baseline","instrumented")]
        if any(type(n) is not bool for n in numerical):
            raise ValueError("missing output reference result")
        result=dict(info,status="passed",records=len(records),inputs_equal=True,outputs_bitwise_equal=True,numerical_status="passed" if all(numerical) else "failed")
        save_json(output/"run.json",result)
        return result
    except Exception as exc:
        save_json(output/"run.json",dict(info,status="failed",complete=False,error=str(exc)))
        raise
