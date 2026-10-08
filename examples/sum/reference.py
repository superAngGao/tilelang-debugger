"""CPU reference for both reviewed Sum paths."""
import torch


def reference(inputs, points):
    x, = inputs
    result = {}
    for p in points:
        start = p["block"][0] * 2
        rows = x[start:start+2].float()
        if p["buffer"] == "x_f32":
            result[p["id"]] = dict(tensor=torch.nn.functional.pad(rows, (0, p["shape"][1]-rows.shape[1])), atol=0, rtol=0)
        elif p["buffer"] == "acc":
            result[p["id"]] = dict(tensor=rows.sum(dim=1), atol=0.0001, rtol=0.0001)
        else:
            raise ValueError("unknown Sum buffer")
    return dict(points=result, outputs=[dict(tensor=x.float().sum(dim=1), atol=0.03125, rtol=0.002)])
