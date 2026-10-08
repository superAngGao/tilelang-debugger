"""GPU end-to-end raw bit preservation, including NaN payloads and signed zero."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

PATTERNS = {
    "float16": [0, 0x8000, 0x7c00, 0xfc00, 0x7e01, 0x7e55, 0xfe07, 0x7bff, 1, 0x0400],
    "bfloat16": [0, 0x8000, 0x7f80, 0xff80, 0x7fc1, 0x7fe5, 0xffc7, 0x7f7f, 1, 0x0080],
    "float32": [0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00001, 0x7fc12345, 0xffc00007, 0x7f7fffff, 1, 0x00800000],
    "int32": [0, 1, 0xffffffff, 0x80000000, 0x7fffffff, 0x12345678, 0xabcdef12],
}


def worker(dtype, folder):
    os.environ["TILELANG_DISABLE_CACHE"] = "1"
    import torch
    import tilelang
    import tilelang.language as T
    from tilelang_debugger.monitor import session, capture, buffer_bindings
    from tilelang_debugger.capture import printf_capacity, snapshot, save_json
    from tilelang_debugger.ir import export, check_instrumented, layout_digest
    if tilelang.__version__ != "0.1.12" or "H200" not in torch.cuda.get_device_name(0):
        raise RuntimeError("fidelity acceptance requires the pinned TileLang/H200 environment")
    save_json(folder / "environment.json", dict(tilelang=tilelang.__version__, device=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda))
    torch.empty(1, device="cuda")
    printf_capacity()
    n = 128
    @T.prim_func
    def baseline(A: T.Tensor((256,), dtype), O: T.Tensor((256,), dtype)):
        with T.Kernel(2, threads=128) as bx:
            fragment = T.alloc_fragment((n,), dtype)
            T.copy(A[bx * n:(bx + 1) * n], fragment)
            T.copy(fragment, O[bx * n:(bx + 1) * n])
    base = tilelang.compile(baseline, out_idx=[1], target="cuda")
    base_dir = folder / "baseline"
    base_dir.mkdir()
    data, *_ = export(base, baseline, base_dir, [1])
    p = dict(id="bits", line=0, when="after", buffer="fragment", shape=[n], dtype=dtype,
             block=[1,0,0], loop_values=[], loop_vars=[], leader=0, threads=128, barrier=15,
             layout_sha256=[layout_digest([(i, i, 0) for i in range(n)])])
    with session([p]) as bound:
        @T.prim_func
        def instrumented(A: T.Tensor((256,), dtype), O: T.Tensor((256,), dtype)):
            with T.Kernel(2, threads=128) as bx:
                fragment = T.alloc_fragment((n,), dtype)
                T.copy(A[bx * n:(bx + 1) * n], fragment)
                capture(fragment, "bits", (bx, 0, 0), ())
                T.copy(fragment, O[bx * n:(bx + 1) * n])
        inst = tilelang.compile(instrumented, out_idx=[1], target="cuda")
        result, events, variables, allocations, source = export(inst, instrumented, folder, [1])
        save_json(folder / "launch-gate.json", check_instrumented(result, events, variables, allocations, source, data, [p], 128, bindings=buffer_bindings()))
    save_json(folder / "points.json", list(bound.values()))
    width = 16 if dtype in ("float16", "bfloat16") else 32
    expected = [PATTERNS[dtype][i % len(PATTERNS[dtype])] for i in range(256)]
    raw = b"".join(x.to_bytes(width//8, "little") for x in expected)
    x = torch.frombuffer(bytearray(raw), dtype=getattr(torch, dtype)).clone().cuda()
    snapshot([x], folder, "inputs")
    out = inst(x)
    torch.cuda.synchronize()
    snapshot([out], folder, "outputs")
    if (folder / "outputs-0.bin").read_bytes() != raw:
        raise AssertionError("bit pattern changed in output")
    save_json(folder / "expected.json", expected[n:])


def run(folder, sanitizer=None):
    from tilelang_debugger.records import parse
    results = []
    for dtype in PATTERNS:
        target = folder / dtype
        target.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", dtype, "--output", str(target)]
        if sanitizer:
            command = ["compute-sanitizer", "--tool", sanitizer, "--error-exitcode", "86", "--log-file", str(target / f"{sanitizer}.log")] + command
        with (target / "stdout.log").open("w") as out, (target / "stderr.log").open("w") as err:
            subprocess.run(command, stdout=out, stderr=err, timeout=240, check=True)
        points = json.loads((target / "points.json").read_text())
        records = parse((target / "stdout.log").read_text(), points)
        if [r["bits"] for r in records] != json.loads((target / "expected.json").read_text()):
            raise AssertionError(f"device bit pattern loss for {dtype}")
        results.append(dict(dtype=dtype, passed=True, records=len(records), patterns=PATTERNS[dtype]))
    (folder / "summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", choices=PATTERNS)
    parser.add_argument("--sanitizer", choices=("racecheck", "synccheck"))
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, Path(args.output))
    else:
        run(Path(args.output).resolve(), args.sanitizer)
