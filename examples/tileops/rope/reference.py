"""CPU neox rotation using the actual input frequency tables."""
def reference(inputs, case):
    import torch
    x, cos, sin = inputs
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[:, half:], x[:, :half]), dim=-1).float()
    return (x.float() * torch.cat((cos, cos), -1).float()
            + rotated * torch.cat((sin, sin), -1).float()).to(x.dtype)
