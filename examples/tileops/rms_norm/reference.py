"""CPU RMS normalization; the divisor is original N, not padded width."""
def reference(inputs, case):
    import torch
    x, weight = inputs
    r = torch.rsqrt(x.float().square().sum(-1, keepdim=True) / case["N"] + case["eps"])
    return (x.float() * r * weight.float()).to(x.dtype)
