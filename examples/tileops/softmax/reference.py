"""CPU mathematical reference; this is a fixture reference, not an analyze provider."""
def reference(inputs, case):
    import torch.nn.functional as F
    x, = inputs
    width = case.get("tile_n", 256)
    padded = ((case["N"] + 255) // 256 * 256 + width - 1) // width * width
    y = F.softmax(x.float(), dim=-1).to(x.dtype)
    return F.pad(y, (0, padded - case["N"]))
