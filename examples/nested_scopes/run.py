"""One compile/launch; the debugger supplies observation locations separately."""
import argparse
import json
import os
from pathlib import Path

import torch
import tilelang
from kernel import build, local_build


def expected_output(x, local=False):
    if local:
        return x * 2 + 7
    y = torch.empty(69, dtype=torch.int32)
    for tx in range(32):
        total = 0
        for outer in range(2, 4):
            if tx % 2 == 0:
                for j in range(1, 3):
                    if j == 1:
                        for inner in range(3, 5):
                            total += tx * 1000 + j * 100 + inner
                    elif tx % 4 == 0:
                        total += tx * 1000 + j
            else:
                total += tx + outer
        y[tx] = total
    y[32:] = x + torch.arange(37, dtype=torch.int32) * 10
    return y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    x = torch.arange(32 if args.local else 37, dtype=torch.int32, device="cuda")
    compiled = tilelang.compile((local_build if args.local else build)(), out_idx=[1], target="cuda")
    actual = compiled(x)
    passed = torch.equal(actual.cpu(), expected_output(x.cpu(), args.local))
    output = os.environ.get("TLDBG_OUTPUT")
    if output:
        (Path(output) / "reference.json").write_text(json.dumps({"passed": passed}) + "\n")
    elif not passed:
        raise AssertionError("nested fixture output differs")
