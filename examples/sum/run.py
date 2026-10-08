from tilelang_debugger.numerics import check_output
import torch
import tilelang
from kernel import build_kernel

torch.manual_seed(1234)
x = torch.randn((4, 257), device="cuda", dtype=torch.float16)
kernel = tilelang.compile(build_kernel(M=4, N=257), out_idx=[1], target="cuda")
out = kernel(x)
ref = x.float().sum(dim=1)
check_output(out, ref, atol=0.03125, rtol=0.002)
