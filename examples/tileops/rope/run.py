"""Keep upstream direct parallel scalar accesses; do not add a fragment."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import run_example
from reference import reference


def build(module, case):
    import torch
    dtype = getattr(torch, case["dtype"])
    seq, dim = case["seq_len"], case["head_dim"]
    x = torch.randn(seq, dim, device="cuda", dtype=dtype)
    angles = torch.outer(torch.arange(seq, device="cuda", dtype=torch.float32),
                         1.0 / (10000 ** (torch.arange(dim // 2, device="cuda", dtype=torch.float32) / (dim // 2))))
    inputs = (x, angles.cos().to(dtype), angles.sin().to(dtype))
    kernel = getattr(module, case["factory"])(seq, dim, case["dtype"], case["threads"], case["num_per_thread"])(case["threads"], case["num_per_thread"])
    return kernel, inputs


if __name__ == "__main__":
    raise SystemExit(run_example("rope", build, reference))
