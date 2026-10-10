"""Logical paths and scalar atomic/address operands, without debug calls."""
import argparse
import json
import os
from pathlib import Path

import tilelang
import tilelang.language as T
import torch


def build(case):
    @T.prim_func
    def main(x: T.Tensor((32,), 'int32'), y: T.Tensor((32,), 'int32')):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            shared = T.alloc_shared((32,), 'int32')
            if case == 'logic':
                and_value = T.And(tx < 16, x[tx] > 0)
                or_value = T.Or(tx >= 16, x[tx] > 0)
                chain_value = tx < 16 < x[tx]
                conditional_value = x[tx] if tx < 16 else x[(tx + 1) % 32]
                bit_value = x[(tx << 1) & 31]
                python_value = tx < 16 and x[tx] > 0
                y[tx] = T.cast(and_value, 'int32') + T.cast(or_value, 'int32') + T.cast(chain_value, 'int32') + conditional_value + bit_value + T.cast(python_value, 'int32')
            else:
                T.copy(src=x, dst=shared)
                T.sync_threads()
                T.copy(src=shared, dst=y)
                T.sync_threads()
                T.atomic_add(dst=y[tx], value=1)
                T.atomic_max(dst=y[tx], value=x[tx] + 2)
                T.atomic_min(dst=y[tx], value=x[tx] + 1)
                pointer = T.address_of(y[tx])
    return main


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=('logic', 'operations'), required=True)
    args = parser.parse_args()
    kernel = tilelang.compile(build(args.case), out_idx=[1], target='cuda')
    x = torch.arange(32, device='cuda', dtype=torch.int32)
    result = kernel(x).cpu()
    expected = torch.tensor([
        int(i < 16 and i > 0) + int(i >= 16 or i > 0) + int(i < 16 < i)
        + (i if i < 16 else (i + 1) % 32) + ((i << 1) & 31) + int(i < 16 and i > 0)
        for i in range(32)], dtype=torch.int32) if args.case == 'logic' else torch.arange(32, dtype=torch.int32) + 1
    passed = torch.equal(result, expected)
    (Path(os.environ['TLDBG_OUTPUT']) / 'reference.json').write_text(json.dumps(dict(passed=passed)))
    if not passed:
        raise AssertionError('expression fixture output mismatch')
