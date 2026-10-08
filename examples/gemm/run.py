from tilelang_debugger.numerics import check_output
import torch
import tilelang
from kernel import build_kernel

torch.manual_seed(1234)
torch.backends.cuda.matmul.allow_tf32 = False
a = torch.randn((128, 256), device="cuda", dtype=torch.float16)
b = torch.randn((256, 256), device="cuda", dtype=torch.float16)
prim = build_kernel(128, 256, 256).get_tir(block_m=128, block_n=128, block_k=64, num_stages=3)
kernel = tilelang.compile(prim, out_idx=[2], target="cuda",
                          pass_configs={"tl.disable_warp_specialized": True}, compile_flags=["-O3", "-DENABLE_BF16"])
out = kernel(a, b)
ref = a.float() @ b.float().T
check_output(out, ref, atol=0.0625, rtol=0.002)
