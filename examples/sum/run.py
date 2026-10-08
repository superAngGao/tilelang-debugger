import json
import os
from pathlib import Path
import torch
import tilelang
from kernel import build_kernel

torch.manual_seed(1234)
x = torch.randn((4, 257), device="cuda", dtype=torch.float16)
kernel = tilelang.compile(build_kernel(M=4, N=257), out_idx=[1], target="cuda")
out = kernel(x)
ref = x.float().sum(dim=1)
torch.testing.assert_close(out.float(), ref, atol=0.03125, rtol=0.002)
if os.getenv("TLDBG_OUTPUT"):
    Path(os.environ["TLDBG_OUTPUT"], "reference.json").write_text(json.dumps({
        "passed": True, "atol": 0.03125, "rtol": 0.002, "max_abs_error": (out.float()-ref).abs().max().item()}))
