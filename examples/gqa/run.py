import json
import os
from pathlib import Path
import torch
import tilelang
from kernel import build_kernel

torch.manual_seed(1234)
torch.backends.cuda.matmul.allow_tf32 = False
q = torch.randn((1, 256, 2, 64), device="cuda", dtype=torch.float16)
k = torch.randn((1, 384, 1, 64), device="cuda", dtype=torch.float16)
v = torch.randn_like(k)
built = build_kernel(seq_len_q=256, seq_len_kv=384, batch=1, heads=2, heads_kv=1)
kernel = tilelang.compile(built.prim_func, out_idx=[3], target="cuda",
                          pass_configs=built.pass_configs, compile_flags=built.compile_flags)
out = kernel(q, k, v)
qh = q.float().permute(0, 2, 1, 3)
kh = k.float().repeat_interleave(2, dim=2).permute(0, 2, 1, 3)
vh = v.float().repeat_interleave(2, dim=2).permute(0, 2, 1, 3)
ref = (torch.softmax((qh @ kh.transpose(-1, -2)) * 0.125, dim=-1) @ vh).permute(0, 2, 1, 3)
torch.testing.assert_close(out.float(), ref, atol=0.01, rtol=0.01)
if os.getenv("TLDBG_OUTPUT"):
    Path(os.environ["TLDBG_OUTPUT"], "reference.json").write_text(json.dumps({
        "passed": True, "atol": 0.01, "rtol": 0.01, "max_abs_error": (out.float()-ref).abs().max().item()}))
