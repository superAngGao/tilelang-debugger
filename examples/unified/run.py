import argparse
import json
import os
from pathlib import Path

import tilelang
import torch
from kernel import build, inplace_build


def expected(case, x, n):
    result = x.clone()
    if case == 'parallel':
        result[:61] += torch.arange(61, dtype=x.dtype)
    elif case == 'pipeline':
        result[:32] += 6
    elif case == 'group':
        result[128:256] += 2
    elif case == 'two_dim':
        result[:32] += torch.arange(32, dtype=x.dtype) + 17
    elif case == 'dynamic':
        for tx in range(32):
            if tx % n:
                result[tx] += tx % n - 1
    elif case == 'while':
        result[:32] += 3
    elif case == 'negative':
        result[:32] += 2
    elif case == 'fragment':
        result[:64] += 3
    elif case == 'global':
        result[:32] += 7
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True)
    parser.add_argument('--launches', type=int, default=1)
    parser.add_argument('--rebuild', action='store_true')
    args = parser.parse_args()
    kernel = None
    passed = True
    x = torch.arange(32 if args.case == 'inplace' else 256, dtype=torch.int32, device='cuda')
    for launch in range(args.launches):
        if kernel is None or args.rebuild:
            kernel = tilelang.compile(inplace_build() if args.case == 'inplace' else build(args.case), out_idx=[] if args.case == 'inplace' else [1], target='cuda')
        original = x.cpu().clone()
        if args.case == 'inplace':
            kernel(x, 5)
            passed &= torch.equal(x.cpu(), original + 5)
        else:
            actual = kernel(x, 5)
            passed &= torch.equal(actual.cpu(), expected(args.case, original, 5))
    folder = os.environ.get('TLDBG_OUTPUT')
    if folder:
        (Path(folder) / 'reference.json').write_text(json.dumps({'passed': passed}))
    if not passed:
        raise AssertionError('independent output reference differs')
