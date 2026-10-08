from tilelang_debugger.numerics import check_output
import torch
import tilelang
from kernel import build_kernel

torch.manual_seed(1234)
x = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
kernel = tilelang.compile(build_kernel(N_total=4096), out_idx=[1], target="cuda")
out = kernel(x)
ref = torch.nn.functional.gelu(x.float(), approximate="none")
check_output(out, ref, atol=0.03125, rtol=0.02)
