"""Complete kernel and captured-tile acceptance. Any failed case exits nonzero."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def tensor(folder, prefix, i=0):
    import torch
    meta = json.loads((folder / f"{prefix}.json").read_text())[i]
    raw = bytearray((folder / meta["file"]).read_bytes())
    return torch.frombuffer(raw, dtype=getattr(torch, meta["dtype"])).clone().reshape(meta["shape"])


def tiles(folder):
    import torch
    points = json.loads((folder / "points.json").read_text())
    records = [json.loads(line) for line in (folder / "records.jsonl").read_text().splitlines()]
    result = {}
    for p in points:
        width = 2 if p["dtype"] in ("float16", "bfloat16") else 4
        raw = bytearray(b"".join(r["bits"].to_bytes(width, "little") for r in records if r["point"] == p["id"]))
        result[p["id"]] = torch.frombuffer(raw, dtype=getattr(torch, p["dtype"])).clone().reshape(p["shape"])
    return points, result


def verify(folder, name, sanitizer=None):
    import torch
    from tilelang_debugger.evidence import verify_success
    verify_success(folder, sanitizer)
    points, values = tiles(folder)
    baseline = folder / "baseline"
    x = tensor(baseline, "inputs")
    out = tensor(baseline, "outputs")
    errors = {}
    for p in points:
        block = p["block"]
        if name == "gelu":
            expected = (x if p["buffer"] == "x_reg" else out)[block[0]*2048:(block[0]+1)*2048]
            atol = rtol = 0
        elif name.startswith("sum"):
            rows = x[block[0]*2:(block[0]+1)*2].float()
            if p["buffer"] == "x_f32":
                expected = torch.nn.functional.pad(rows, (0, p["shape"][1] - rows.shape[1]))
                atol = rtol = 0
            else:
                expected = rows.sum(dim=1)
                atol, rtol = 0.0001, 0.0001
        elif name == "gemm":
            b = tensor(baseline, "inputs", 1)
            end_k = p["loops"][0]["iteration"] * 64
            expected = x[block[1]*128:(block[1]+1)*128, :end_k].float() @ b[block[0]*128:(block[0]+1)*128, :end_k].float().T
            atol, rtol = 0.001, 0.0001
        elif name == "gqa":
            k = tensor(baseline, "inputs", 1)
            row = block[0]*128 + (p["leader"]//128)*64
            expected = x[block[2], row:row+64, block[1], :].float() @ k[block[2], :128, 0, :].float().T
            atol, rtol = 0.001, 0.0001
        else:
            raise ValueError(name)
        actual = values[p["id"]]
        torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)
        if not atol and not rtol and actual.contiguous().view(torch.uint8).numpy().tobytes() != expected.contiguous().view(torch.uint8).numpy().tobytes():
            raise AssertionError("exact capture differs in bits")
        errors[p["id"]] = dict(max_abs_error=(actual.float()-expected.float()).abs().max().item(), atol=atol, rtol=rtol)
    result = dict(passed=True, tile_reference=errors,
                  output_reference=json.loads((folder / "instrumented" / "reference.json").read_text()))
    (folder / "validation.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    from tilelang_debugger.capture import run, save_json
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--sanitizer", choices=("racecheck", "synccheck"))
    parser.add_argument("--cases", nargs="+", default=["gelu", "sum", "sum_unpadded", "gemm", "gqa"])
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    save_json(output / "summary.json", dict(status="running", sanitizer=args.sanitizer, cases=results))
    try:
        for name in args.cases:
            example = ROOT / "examples" / ("sum" if name == "sum_unpadded" else name)
            folder = output / name
            if not args.verify_existing:
                run(example / ("run_unpadded.py" if name == "sum_unpadded" else "run.py"),
                    example / "monitor.json", folder, timeout=360, sanitizer=args.sanitizer)
            results[name] = verify(folder, name, args.sanitizer)
            print(json.dumps({name: results[name]}), flush=True)
            save_json(output / "summary.json", dict(status="running", sanitizer=args.sanitizer, cases=results))
    except Exception as exc:
        if folder.exists():
            save_json(folder / "validation.json", dict(passed=False, error=str(exc)))
        save_json(output / "summary.json", dict(status="failed", sanitizer=args.sanitizer, cases=results, error=str(exc)))
        raise
    save_json(output / "summary.json", dict(status="passed", sanitizer=args.sanitizer, cases=results))


if __name__ == "__main__":
    main()
