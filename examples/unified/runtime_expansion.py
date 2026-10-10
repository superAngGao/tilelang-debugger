"""Dynamic shapes, strided/overlapping inputs and caller-current CUDA streams."""
import argparse
import json
import os
from pathlib import Path
import torch
import tilelang
import tilelang.language as T


def build():
    n = T.symbolic('n')
    stride = T.symbolic('stride')
    @T.prim_func
    def main(x: T.StridedTensor((n,), (stride,), 'int32'), y: T.Tensor((n,), 'int32')):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(n):
                result = x[i] + 3
                y[i] = result
            T.sync_threads()
    return main


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=('contiguous', 'strided', 'expanded'), default='strided')
    args = parser.parse_args()
    compiled = tilelang.compile(build(), out_idx=[1], target='cuda')
    stream = torch.cuda.Stream()
    passed = True
    for n in (17, 39):
        with torch.cuda.stream(stream):
            base = torch.arange(n * 2 + 3, dtype=torch.int32, device='cuda')
            x = base[1:1 + n] if args.case == 'contiguous' else base[1:1 + n * 2:2] if args.case == 'strided' else base[1:2].expand(n)
            y = compiled(x=x)
            passed &= torch.equal(y.cpu(), x.cpu() + 3)
    folder = os.environ.get('TLDBG_OUTPUT')
    if folder:
        (Path(folder) / 'reference.json').write_text(json.dumps({'passed': passed}))
    assert passed
