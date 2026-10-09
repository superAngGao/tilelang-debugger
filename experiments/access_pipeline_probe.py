"""Historical failed FFI-hook experiment; NOT the product tracing path.

The parent preserves teardown failure even if the worker finishes its function body.
"""
import argparse
import importlib
import json
from pathlib import Path
import runpy
import sys
import os
import gc
import subprocess

os.environ["TILELANG_DISABLE_CACHE"] = "1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("driver")
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    folder = Path(args.output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    if not args.worker:
        from tilelang_debugger.capture import save_json
        command=[sys.executable,str(Path(__file__).resolve()),args.driver,"--output",str(folder/"worker"),"--worker"]
        save_json(folder/"command.json",command)
        with (folder/"stdout.log").open("w") as out,(folder/"stderr.log").open("w") as err:
            result=subprocess.run(command,stdout=out,stderr=err,timeout=240)
        save_json(folder/"process.json",dict(returncode=result.returncode,timeout=False))
        save_json(folder/"validation.json",dict(passed=result.returncode==0))
        raise SystemExit(result.returncode)
    import tilelang
    import tvm
    import tvm_ffi
    from tilelang.backend.pass_pipeline.pipeline import get_pipeline, register_pipeline, PassPipeline
    from tilelang_debugger.capture import save_json
    pipeline = get_pipeline("cuda")
    lower = importlib.import_module("tilelang.engine.lower")
    prepare = lower._prepare_device_codegen_mod
    encode = tvm_ffi.get_global_func("__tvm_tensormap_create_tiled")
    callbacks = []

    def traced_pipeline(mod, target):
        mod = pipeline.lower(mod, target)
        (folder / "pipeline.py").write_text(mod.script(show_meta=True))
        (folder / "pipeline.json").write_text(tvm.ir.save_json(mod))
        for gv, func in mod.functions.items():
            print(gv.name_hint, list(func.attrs.keys()) if func.attrs else [])
        return mod

    def prepared(mod):
        mod = prepare(mod)
        (folder / "codegen.py").write_text(mod.script(show_meta=True))
        return mod

    def encoded(*values):
        callbacks.append([dict(type=type(v).__name__, value=str(v)) for v in values])
        result = encode(*values)
        print("DESCRIPTOR", callbacks[-1], "RETURN", result, flush=True)
        save_json(folder / "constructors.json", callbacks)
        return result

    register_pipeline(PassPipeline("cuda", traced_pipeline))
    lower._prepare_device_codegen_mod = prepared
    tvm_ffi.register_global_func("__tvm_tensormap_create_tiled", encoded, override=True)
    driver = Path(args.driver).resolve()
    sys.path.insert(0, str(driver.parent))
    try:
        runpy.run_path(str(driver), run_name="__main__")
    finally:
        gc.collect()
        sys.path.pop(0)
        register_pipeline(pipeline)
        lower._prepare_device_codegen_mod = prepare
        tvm_ffi.register_global_func("__tvm_tensormap_create_tiled", encode, override=True)
        gc.collect()
    save_json(folder / "body-completed.json", dict(body_completed=True, descriptors=len(callbacks), restored=get_pipeline("cuda") is pipeline and lower._prepare_device_codegen_mod is prepare))


if __name__ == "__main__":
    main()
