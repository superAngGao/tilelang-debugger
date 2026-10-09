"""Real upstream nested operators; no copied kernels or product adaptations."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_upstream


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tileops", required=True)
    parser.add_argument("--case", choices=("pool", "indices", "rope", "softmax"), required=True)
    args = parser.parse_args()
    if args.case in {"pool", "indices"}:
        module, provenance = load_upstream(args.tileops, "tileops.kernels.pool.max_pool2d")
        x = ((torch.arange(12, device="cuda", dtype=torch.float32) * 7) % 19).reshape(1, 1, 3, 4)
        factory = module._max_pool2d_kernel if args.case == "pool" else module._max_pool2d_with_indices_kernel
        kernel = factory(1, 1, 3, 4, 3, 3, 1, 1, 1, 1, 1, 1, False, "float32")(16, 32)
        output = kernel(x)
        ref = torch.nn.functional.max_pool2d(x, 3, 1, 1, return_indices=args.case == "indices")
        passed = all(torch.equal(a, b) for a, b in zip(output, ref)) if args.case == "indices" else torch.equal(output, ref)
    elif args.case == "rope":
        module, provenance = load_upstream(args.tileops, "tileops.kernels.rope")
        x = torch.arange(48, device="cuda", dtype=torch.float32) / 16
        cos = torch.ones((8, 4), device="cuda", dtype=torch.float32)
        sin = torch.zeros_like(cos)
        positions = torch.tensor([3, 1, 6], device="cuda", dtype=torch.int32)
        kernel = module._make_rope_neox_position_ids_thd(3, 2, 8, 8, 8, "float32", 32, 2)(32, 2)
        output = kernel(x, cos, sin, positions)
        passed = torch.equal(output, x)
    else:
        module, provenance = load_upstream(args.tileops, "tileops.kernels.reduction.softmax")
        x = torch.randn((2, 513), device="cuda", dtype=torch.float32)
        kernel = module._softmax_kernel_tiled(2, 513, "softmax", "float32", 256)(1, 32)
        output = kernel(x)
        passed = torch.allclose(output[:, :513], torch.softmax(x, -1), atol=1e-5, rtol=1e-5) and bool((output[:, 513:] == 0).all())
    folder = Path(os.environ["TLDBG_OUTPUT"])
    (folder / "reference.json").write_text(json.dumps(dict(passed=passed)) + "\n")
    (folder / "upstream.json").write_text(json.dumps(provenance, indent=2) + "\n")
