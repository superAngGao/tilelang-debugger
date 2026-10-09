import torch


def reference(inputs, points):
    x = inputs[0]
    rows = [dict(coordinates=[k], ordinals=[k + 1], thread=0, visit=k, index=i, value=int(x[k * 64 + i]) + k) for k in range(4) for i in range(64)]
    output = x + torch.arange(4, dtype=x.dtype).repeat_interleave(64)
    spec = lambda tensor: dict(tensor=tensor, atol=0, rtol=0)
    return dict(points={p['id']: dict(schema=3, key='execution', dtype='int32', samples=rows if p['id'] == 'copy' else [], atol=0, rtol=0) for p in points},
                arguments_after=[spec(x.clone())], outputs=[spec(output)])
