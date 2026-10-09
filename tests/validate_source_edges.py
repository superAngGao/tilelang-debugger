"""Real source-engine boundary cases on H200; keep rejected workers as evidence."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys

from tilelang_debugger.capture import save_json
from tilelang_debugger.evidence import verify_capture

KERNEL = '''import tilelang
import tilelang.language as T
@tilelang.jit(out_idx=[1])
def build():
    @T.prim_func
    def main(x: T.Tensor((256,), "float32"), y: T.Tensor((256,), "float32")):
        with T.Kernel(2, threads=128) as bx:
            buf = T.alloc_fragment((128,), "float32")
            T.copy(x[bx * 128], buf)
            for k in T.serial(3, 9, 2):
                for i in T.Parallel(128):
                    buf[i] = buf[i] + 1.0
            T.copy(buf, y[bx * 128])
    return main
'''
DRIVER = '''import torch
from package.custom_name import build
kernel = build()
kernel(torch.arange(256, device="cuda", dtype=torch.float32))
'''
REFERENCE = '''def reference(inputs, points):
    x, = inputs
    spec = lambda t: dict(tensor=t, atol=0, rtol=0)
    return dict(points={p['id']:spec(x[128:256]+2) for p in points}, outputs=[spec(x+3)])
'''


def execute(command, folder):
    folder.mkdir(parents=True)
    save_json(folder / "command.json", command)
    with (folder / "stdout.log").open("w") as out, (folder / "stderr.log").open("w") as err:
        result = subprocess.run(command, stdout=out, stderr=err)
    save_json(folder / "process.json", dict(returncode=result.returncode))
    return result.returncode


def selection(source):
    tree = ast.parse(source)
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For) and ast.unparse(n.iter.func) == "T.serial")
    parallel = next(n for n in loop.body if isinstance(n, ast.For))
    return dict(id="selected", line=parallel.lineno, when="after", buffer="buf", block=[1,0,0], loops=[dict(line=loop.lineno, iteration=2)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve(); root.mkdir(parents=True, exist_ok=False)
    results = []
    for name in ("package", "renamed", "same_file", "wrong_math", "driver_error", "missing_import", "dynamic_loop", "out_of_range", "global_alias"):
        folder = root / name; folder.mkdir()
        package = folder / "package"; package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "settings.py").write_text("WIDTH = 256\n")
        source = KERNEL
        driver_text = DRIVER
        if name == "package":
            source = "from .settings import WIDTH\n" + source.replace("(256,)", "(WIDTH,)")
        if name == "renamed": source = "\n\n# relocated source\n" + source.replace("buf", "changed_buffer")
        if name == "wrong_math": source = source.replace("+ 1.0", "+ 2.0")
        if name == "dynamic_loop": source = source.replace("3, 9, 2", "3, T.get_thread_binding() + 9, 2")
        if name == "driver_error": driver_text += "raise RuntimeError('deliberate driver failure')\n"
        if name == "missing_import": driver_text = "print('selected source intentionally not imported')\n"
        selected = selection(source)
        if name == "renamed": selected["buffer"] = "changed_buffer"
        if name == "out_of_range": selected["loops"][0]["iteration"] = 99
        if name == "global_alias":
            source = '''import tilelang
import tilelang.language as T
@tilelang.jit(out_idx=[2])
def build():
    @T.prim_func
    def main(x: T.Tensor((256,), "float32"), z: T.Tensor((256,), "float32"), y: T.Tensor((256,), "float32")):
        with T.Kernel(2, threads=128) as bx:
            for i in T.Parallel(128):
                y[bx*128+i] = x[bx*128+i] + z[bx*128+i]
    return main
'''
            driver_text = "import torch\nfrom package.custom_name import build\nx=torch.arange(256,device='cuda',dtype=torch.float32)\nbuild()(x,x)\n"
            selected = dict(id="selected", line=8, when="before", buffer="x", block=[1,0,0], loops=[])
        source_path = package / "custom_name.py"
        driver = folder / "driver.py"
        if name == "same_file":
            source += DRIVER.replace("from package.custom_name import build\n", "")
            source_path = driver
        else:
            driver.write_text(driver_text)
        source_path.write_text(source)
        original = source_path.read_bytes()
        save_json(folder / "monitor.json", dict(source=str(source_path), points=[selected]))
        command = [sys.executable, "-m", "tilelang_debugger", "run", str(driver), "--monitor", str(folder / "monitor.json"), "--output", str(folder / "capture")]
        status = execute(command, folder / "process")
        if name in {"package", "renamed", "same_file", "wrong_math"}:
            if status != 0: raise ValueError(f"{name}: unexpected capture exit {status}")
            verify_capture(folder / "capture")
            run = json.loads((folder / "capture/run.json").read_text())
            if run["numerical_status"] != "not_checked": raise ValueError("missing reference misclassified")
            (folder / "reference.py").write_text(REFERENCE)
            analysis_status = execute([sys.executable, "-m", "tilelang_debugger", "analyze", str(folder / "capture"), "--reference", str(folder / "reference.py"), "--output", str(folder / "analysis")], folder / "analysis-process")
            if analysis_status != (2 if name == "wrong_math" else 0): raise ValueError("unexpected analysis status")
        else:
            if status == 0: raise ValueError(f"{name}: expected rejection")
            mode = "baseline" if name in {"driver_error", "missing_import"} else "instrumented"
            worker = folder / "capture" / mode
            execution = json.loads((worker / "execution.json").read_text())
            if execution["restored"] is not True: raise ValueError("hooks not restored after failure")
            stderr = (worker / "stderr.log").read_text()
            reason = {"driver_error": "deliberate driver failure", "missing_import": "load selected source",
                      "dynamic_loop": "", "out_of_range": "outside the static source loop", "global_alias": "shares storage"}[name]
            if reason and reason not in stderr: raise ValueError(f"wrong failure reason for {name}")
            if name in {"dynamic_loop", "out_of_range"} and execution["launches"] != 0:
                raise ValueError("unsupported loop reached launch")
        if source_path.read_bytes() != original: raise ValueError("user source changed")
        results.append(dict(case=name, passed=True, returncode=status))
        save_json(root / "summary.json", dict(passed=len(results) == 9, results=results))
        print(json.dumps(results[-1]), flush=True)


if __name__ == "__main__":
    main()
