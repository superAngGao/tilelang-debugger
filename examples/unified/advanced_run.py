import argparse
import json
import os
from pathlib import Path

import tilelang
import torch
from advanced_kernel import build, typed_build, helper_build, alias_build


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True)
    parser.add_argument('--dtype', default='int32')
    parser.add_argument('--scope', default='scalar')
    args = parser.parse_args()
    if args.case == 'alias':
        # Compile both before launching either; the first has no selected point.
        helper = tilelang.compile(helper_build(), out_idx=[], target='cuda')
        main = tilelang.compile(alias_build(), out_idx=[], target='cuda')
        x = torch.arange(32, dtype=torch.int32, device='cuda')
        scalar = torch.tensor([7], dtype=torch.int32, device='cuda')
        helper(x)
        main(x, x, scalar)
        passed = torch.equal(x.cpu(), torch.arange(32, dtype=torch.int32) + 8)
    elif args.case == 'types':
        main = tilelang.compile(typed_build(args.dtype, args.scope), out_idx=[1], target='cuda')
        x = torch.arange(32, dtype=torch.int32, device='cuda')
        passed = torch.equal(main(x).cpu(), x.cpu())
    else:
        main = tilelang.compile(build(args.case), out_idx=[1], target='cuda')
        x = torch.arange(256, dtype=torch.int32, device='cuda')
        output = main(x).cpu()
        expected = x.cpu()
        if args.case == 'group_fragment':
            expected[:128] += 11
        elif args.case == 'pipeline_fragment':
            expected[:32] += 3
        elif args.case == 'nested':
            expected[:32:2] += 11
        elif args.case == 'final_return':
            expected[:32] += 1
        passed = torch.equal(output, expected)
    folder = os.environ.get('TLDBG_OUTPUT')
    if folder:
        (Path(folder) / 'reference.json').write_text(json.dumps({'passed': passed}))
    if not passed:
        raise AssertionError('advanced reference output mismatch')
