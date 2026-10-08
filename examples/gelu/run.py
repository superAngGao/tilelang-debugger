import json
import os
from pathlib import Path
import torch
import tilelang
from kernel import build_kernel

torch.manual_seed(1234)
x = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
kernel = tilelang.compile(build_kernel(N_total=4096), out_idx=[1], target="cuda")
out = kernel(x)
ref = torch.nn.functional.gelu(x.float(), approximate="none")
torch.testing.assert_close(out.float(), ref, atol=0.03125, rtol=0.02)
if os.getenv("TLDBG_OUTPUT"):
    Path(os.environ["TLDBG_OUTPUT"], "reference.json").write_text(json.dumps({
        "passed": True, "atol": 0.03125, "rtol": 0.02, "max_abs_error": (out.float()-ref).abs().max().item()}))
