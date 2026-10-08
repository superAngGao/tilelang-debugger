"""CPU QK and complete attention reference for the reviewed GQA configuration."""
import torch


def reference(inputs, points):
    q, k, v = inputs
    result = {}
    for p in points:
        if p["buffer"] != "acc_s" or p["leader"] not in (0, 128):
            raise ValueError("unknown GQA consumer/buffer")
        bx, head, batch = p["block"]
        row = bx * 128 + (p["leader"] // 128) * 64
        tile = q[batch, row:row+64, head, :].float() @ k[batch, :128, 0, :].float().T
        result[p["id"]] = dict(tensor=tile, atol=0.001, rtol=0.0001)
    qh = q.float().permute(0, 2, 1, 3)
    kh = k.float().repeat_interleave(2, dim=2).permute(0, 2, 1, 3)
    vh = v.float().repeat_interleave(2, dim=2).permute(0, 2, 1, 3)
    out = (torch.softmax((qh @ kh.transpose(-1, -2)) * 0.125, dim=-1) @ vh).permute(0, 2, 1, 3)
    return dict(points=result, outputs=[dict(tensor=out, atol=0.01, rtol=0.01)])
