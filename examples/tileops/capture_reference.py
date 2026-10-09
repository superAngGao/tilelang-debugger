"""Independent CPU references for source-capture fixtures, never product dispatch."""


def reference_case(inputs, points, case):
    import torch
    import torch.nn.functional as F
    atol, rtol = {"float16": (1e-3, 1e-3), "bfloat16": (.016, .016), "float32": (1e-5, 1e-5)}[case["dtype"]]
    x = inputs[0]
    spec = lambda tensor, a=atol, r=rtol: dict(tensor=tensor.contiguous(), atol=a, rtol=r)
    values = {}
    for p in points:
        row = p["block"][0] * case.get("block_m", 1)
        selected = x[row:row + case.get("block_m", 1)].float()
        name = p["buffer"]
        if case["example"] == "softmax":
            maximum = selected.amax(-1)
            if name == "row_max": expected = maximum
            elif name == "row_sum": expected = (selected - maximum[:, None]).exp().sum(-1)
            elif name == "tile_sum":
                t = p["loop_values"][0]
                end = min((t + 1) * case["tile_n"], case["N"])
                running_max = selected[:, :end].amax(-1, keepdim=True)
                expected = (selected[:, t * case["tile_n"]:end] - running_max).exp().sum(-1)
            else: raise ValueError(f"unknown softmax reference point: {name}")
        elif case["example"] == "rms_norm":
            sumsq = selected.square().sum(-1)
            rrms = torch.rsqrt(sumsq / case["N"] + case["eps"])
            if name == "sumsq": expected = sumsq
            elif name == "rrms": expected = rrms
            elif name == "x_local": expected = (selected * rrms[:, None] * inputs[1].float()).to(x.dtype)
            else: raise ValueError(f"unknown RMSNorm reference point: {name}")
        else:
            if name != "x": raise ValueError("RoPE fixture only observes its input")
            expected = x
        values[p["id"]] = spec(expected, 0 if name == "x" else atol, 0 if name == "x" else rtol)
    if case["example"] == "softmax":
        width = case.get("tile_n", 256)
        padded = ((case["N"] + 255) // 256 * 256 + width - 1) // width * width
        output = F.pad(x.float().softmax(-1).to(x.dtype), (0, padded - case["N"]))
    elif case["example"] == "rms_norm":
        r = torch.rsqrt(x.float().square().sum(-1, keepdim=True) / case["N"] + case["eps"])
        output = (x.float() * r * inputs[1].float()).to(x.dtype)
    else:
        cos, sin = inputs[1:]
        half = x.shape[-1] // 2
        wide = x.float()
        rotated = torch.cat((-wide[:, half:], wide[:, :half]), -1)
        output = (wide * cos.float().repeat(1, 2) + rotated * sin.float().repeat(1, 2)).to(x.dtype)
    return dict(points=values, outputs=[spec(output)])
