"""Independent CPU GELU reference from captured inputs."""
import torch


def reference(inputs, points):
    x, = inputs
    y = torch.nn.functional.gelu(x.float(), approximate="none")
    result = {}
    for p in points:
        start = p["block"][0] * 2048
        if p["buffer"] == "x_reg":
            result[p["id"]] = dict(tensor=x[start:start+2048], atol=0, rtol=0)
        elif p["buffer"] == "y_reg":
            result[p["id"]] = dict(tensor=y[start:start+2048], atol=0.03125, rtol=0.02)
        else:
            raise ValueError("unknown GELU buffer")
    return dict(points=result, outputs=[dict(tensor=y, atol=0.03125, rtol=0.02)])
