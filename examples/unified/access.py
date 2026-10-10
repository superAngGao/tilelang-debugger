"""Source access requests: masked tails, branches, loops, copies and values."""
import argparse
import json
import os
from pathlib import Path

import torch
import tilelang
import tilelang.language as T


def build(shift):
    @T.prim_func
    def main(x: T.Tensor((49,), 'int32'), y: T.Tensor((64,), 'int32')):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            shared = T.alloc_shared((32,), 'int32')
            for k in T.serial(2):
                index = k * 32 + tx + shift
                loaded = T.if_then_else(index < 49, x[index], 0)
                y[k * 32 + tx] = loaded
                if index < 49:
                    branch_value = x[index] + 1
            T.copy(x[32], shared)
            T.copy(shared, y[32])
            cursor = T.alloc_var('int32', init=0)
            while cursor < 5:
                cursor = cursor + 1
                if cursor == 2:
                    continue
                if cursor == 4:
                    break
                while_loaded = x[cursor]
    return main


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shift', type=int, default=0)
    args = parser.parse_args()
    x = torch.arange(49, device='cuda', dtype=torch.int32) * 3
    y = tilelang.compile(build(args.shift), out_idx=[1], target='cuda')(x)
    expected = torch.zeros((64,), dtype=torch.int32)
    expected[:32] = x.cpu()[args.shift:32 + args.shift]
    expected[32:49] = x.cpu()[32:49]
    passed = torch.equal(y.cpu(), expected)
    if os.environ.get('TLDBG_OUTPUT'):
        (Path(os.environ['TLDBG_OUTPUT']) / 'reference.json').write_text(json.dumps(dict(passed=passed)))
    assert passed
