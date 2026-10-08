"""Isolate T.print fragment staging synchronization on an NVIDIA GPU.

This is a diagnostic experiment, not a production monkey patch. The patched
mode reserves barrier 14 for exactly 128 participants in this isolated kernel.
No installed TileLang files are modified. Each mode should run in a fresh process.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


def worker(args):
    import torch
    import tilelang
    import tilelang.language as T
    from tvm import tirx
    from tilelang.language.utils import index_to_coordinates

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    print_module = importlib.import_module("tilelang.language.print_op")
    source = Path(print_module.__file__).read_bytes()
    (output / "print_op.snapshot.py").write_bytes(source)
    (output / "environment.json").write_text(json.dumps({
        "tilelang": tilelang.__version__, "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
        "print_op_sha256": hashlib.sha256(source).hexdigest(),
        "mode": args.mode, "group": args.group, "elements": args.elements,
        "repeats": args.repeats,
    }, indent=2))

    @T.macro
    def synchronized_fragment_print(buffer, elems, condition, msg):
        smem = T.alloc_shared(buffer.shape, buffer.dtype, "shared")
        T.copy(buffer, smem)
        T.sync_threads(14, 128)
        if condition:
            for i in T.serial(elems):
                coords = index_to_coordinates(i, buffer.shape)
                tirx.call_extern("handle", "debug_print_buffer_value", msg,
                                buffer.name, i, smem[coords])
        T.sync_threads(14, 128)

    def patched_helper(condition, buffer, elems, msg=""):
        synchronized_fragment_print(buffer, elems, condition, msg)

    original_helper = print_module.print_fragment_buffer_with_condition
    external_sync = args.mode == "surround"
    n = args.elements
    try:
        if args.mode == "patched":
            print_module.print_fragment_buffer_with_condition = patched_helper

        if args.group == "cta":
            @T.prim_func
            def program(A: T.Tensor((n,), "int32"), O: T.Tensor((n,), "int32")):
                with T.Kernel(1, threads=128):
                    fragment = T.alloc_fragment((n,), "int32")
                    T.copy(A, fragment)
                    if external_sync:
                        T.sync_threads(14, 128)
                    T.print(fragment, msg="print_sync_probe")
                    if external_sync:
                        T.sync_threads(14, 128)
                    T.copy(fragment, O)
        else:
            @T.prim_func
            def program(A: T.Tensor((n,), "int32"), O: T.Tensor((n,), "int32")):
                with T.Kernel(1, threads=384):
                    with T.ws(1):
                        fragment = T.alloc_fragment((n,), "int32")
                        T.copy(A, fragment)
                        if external_sync:
                            T.sync_threads(14, 128)
                        T.print(fragment, msg="print_sync_probe", warp_group_id=1)
                        if external_sync:
                            T.sync_threads(14, 128)
                        T.copy(fragment, O)

        (output / "frontend.py").write_text(program.script())
        kernel = tilelang.compile(program, target="cuda", pass_configs={
            "tl.disable_thread_storage_sync": args.mode != "auto",
            "tl.disable_warp_specialized": True,
            "tl.device_compile_flags": ["-lineinfo"],
        })
        (output / "kernel.cu").write_text(kernel.get_kernel_source())
    finally:
        print_module.print_fragment_buffer_with_condition = original_helper

    a = torch.arange(n, device="cuda", dtype=torch.int32) + 10000
    out = torch.empty_like(a)
    for _ in range(args.repeats):
        kernel(a, out)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, a, rtol=0, atol=0)
    (output / "execution.json").write_text(json.dumps({
        "kernel_output_matches": True,
        "helper_restored": print_module.print_fragment_buffer_with_condition is original_helper,
    }, indent=2))


def run_matrix(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for group in args.groups:
        for mode in args.modes:
            case_dir = output / f"{group}-{mode}"
            case_dir.mkdir(exist_ok=True)
            command = [sys.executable, str(Path(__file__).resolve()), "worker",
                       "--mode", mode, "--group", group,
                       "--elements", str(args.elements), "--repeats", str(args.repeats),
                       "--output", str(case_dir)]
            if args.sanitizer:
                command = [args.sanitizer, "--tool", "racecheck", "--racecheck-report", "analysis",
                           "--error-exitcode", "86", "--log-file", str(case_dir / "racecheck.log")] + command
            (case_dir / "command.json").write_text(json.dumps(command, indent=2))
            try:
                with (case_dir / "stdout.log").open("w") as stdout, (case_dir / "stderr.log").open("w") as stderr:
                    completed = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=120)
                status = completed.returncode
            except subprocess.TimeoutExpired:
                status = "timeout"
            log = (case_dir / "stdout.log").read_text(errors="replace")
            parsed = re.findall(r"msg='print_sync_probe'.*?index=(\d+), dtype=\w+ value=(-?\d+)", log)
            actual = Counter((int(index), int(value)) for index, value in parsed)
            expected = Counter({(i, i + 10000): args.repeats for i in range(args.elements)})
            extra = actual - expected
            missing = expected - actual
            race_file = case_dir / "racecheck.log"
            race_log = race_file.read_text(errors="replace") if race_file.exists() else ""
            summaries = [line for line in race_log.splitlines() if "SUMMARY" in line]
            result = {
                "group": group, "mode": mode, "returncode": status,
                "printed_records": sum(actual.values()), "expected_records": sum(expected.values()),
                "print_matches": actual == expected,
                "extra_or_wrong": sum(extra.values()), "missing": sum(missing.values()),
                "wrong_samples": [[*key, count] for key, count in list(extra.items())[:8]],
                "racecheck_summary": summaries,
            }
            results.append(result)
            print(json.dumps(result), flush=True)
            (output / "summary.json").write_text(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output", required=True)
    common.add_argument("--elements", type=int, default=256)
    common.add_argument("--repeats", type=int, default=3)
    child = commands.add_parser("worker", parents=[common])
    child.add_argument("--mode", choices=["auto", "disabled", "surround", "patched"], required=True)
    child.add_argument("--group", choices=["cta", "wg1"], required=True)
    matrix = commands.add_parser("matrix", parents=[common])
    matrix.add_argument("--groups", nargs="+", choices=["cta", "wg1"], default=["cta", "wg1"])
    matrix.add_argument("--modes", nargs="+", choices=["auto", "disabled", "surround", "patched"],
                        default=["auto", "disabled", "surround", "patched"])
    matrix.add_argument("--sanitizer")
    args = parser.parse_args()
    if args.command == "worker":
        worker(args)
    else:
        run_matrix(args)


if __name__ == "__main__":
    main()
