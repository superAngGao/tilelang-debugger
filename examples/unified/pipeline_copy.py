"""A staged global/shared/fragment copy pipeline, without debug calls."""
import json
import os
from pathlib import Path

import tilelang
import tilelang.language as T
import torch


def build(inactive=True):
    @T.prim_func
    def main(x: T.Tensor((256,), 'int32'), y: T.Tensor((256,), 'int32')):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((64,), 'int32')
            fragment = T.alloc_fragment((64,), 'int32')
            if not inactive:
                unused = T.get_thread_binding() + 100
            for k in T.Pipelined(4, num_stages=3):
                T.copy(x[k * 64:(k + 1) * 64], shared)
                T.copy(shared, fragment)
                for i in T.Parallel(64):
                    fragment[i] += k
                T.copy(fragment, y[k * 64:(k + 1) * 64])
            if not inactive:
                unused_after = T.get_thread_binding() + 200
    return main


if __name__ == '__main__':
    kernel = tilelang.compile(build(), out_idx=[1], target='cuda')
    x = torch.arange(256, dtype=torch.int32, device='cuda')
    result = kernel(x).cpu()
    expected = torch.arange(256, dtype=torch.int32) + torch.arange(4, dtype=torch.int32).repeat_interleave(64)
    passed = torch.equal(result, expected)
    (Path(os.environ['TLDBG_OUTPUT']) / 'reference.json').write_text(json.dumps({'passed': passed}))
    if not passed:
        raise AssertionError('copy pipeline output mismatch')
