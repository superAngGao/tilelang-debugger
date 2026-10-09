"""Directly compile the upstream single/tiled softmax factory."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import run_example
from reference import reference


def build(module, case):
    import torch
    dtype = getattr(torch, case["dtype"])
    x = torch.randn(case["M"], case["N"], device="cuda", dtype=dtype)
    args = (case["M"], case["N"], "softmax", case["dtype"])
    if "tile_n" in case:
        args += (case["tile_n"],)
    kernel = getattr(module, case["factory"])(*args)(case["block_m"], case["threads"])
    return kernel, (x,)


if __name__ == "__main__":
    raise SystemExit(run_example("softmax", build, reference))
