"""Use exactly the input-padding contract of upstream RMSNormKernel.forward."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import run_example
from reference import reference


def build(module, case):
    import torch
    import torch.nn.functional as F
    dtype = getattr(torch, case["dtype"])
    x = torch.randn(case["M"], case["N"], device="cuda", dtype=dtype)
    weight = torch.randn(case["N"], device="cuda", dtype=dtype)
    pad = (case["N"] + 255) // 256 * 256 - case["N"]
    inputs = (F.pad(x, (0, pad)), F.pad(weight, (0, pad)))
    kernel = getattr(module, case["factory"])(case["M"], case["N"], case["eps"], case["dtype"])(case["block_m"], case["threads"])
    return kernel, inputs


if __name__ == "__main__":
    raise SystemExit(run_example("rms_norm", build, reference))
