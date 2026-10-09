"""Independent CPU intermediates with explicit logical key domains."""
def reference(inputs, points):
    import torch
    x = inputs[0]
    samples = {p["id"]: [] for p in points}
    def add(name, coords, val):
        samples[name].append(dict(coordinates=coords, index=0, value=val))
    if "max_pool2d" in points[0]["source_path"]:
        indices = any(p["id"] == "max_idx" for p in points)
        height, width = x.shape[-2:]
        for out in range(height * width):
            oh, ow = divmod(out, width)
            current, first = float("-inf"), True
            if indices:
                add("has_nan", [out], False)
            for kh in range(3):
                for kw in range(3):
                    ih, iw = oh - 1 + kh, ow - 1 + kw
                    if 0 <= ih < height and 0 <= iw < width:
                        val = float(x[0, 0, ih, iw])
                        coords = [out, kh, kw]
                        add("val", coords, val)
                        if indices:
                            if first:
                                current, first = val, False
                            elif val > current:
                                current = val
                                add("max_idx", coords, ih * width + iw)
                        else:
                            current = max(current, val)
                            add("max_val", coords, current)
                            add("has_nan", coords, False)
        result = torch.nn.functional.max_pool2d(x, 3, 1, 1, return_indices=indices)
        outputs = list(result) if indices else [result]
    elif "rope.py" in points[0]["source_path"]:
        for flat in range(48):
            i, j = divmod(flat, 2)
            col = flat % 8
            add("pos", [i, j], int(inputs[3][flat // 16]))
            add("val", [i, j], float(x[flat]))
            add("paired_val", [i, j], float(x[flat - col + (col + 4 if col < 4 else col - 4)]))
        outputs = [x.clone()]
    else:
        for t in range(3):
            for j in range(256):
                col = t * 256 + j
                add("full" if t < 2 else "tail", [t, 0, j], float(x[0, col]) if col < 513 else float("-inf"))
        padded = torch.zeros((2, 768), dtype=x.dtype)
        padded[:, :513] = torch.softmax(x, -1)
        outputs = [padded]
    return dict(points={p["id"]: dict(schema=2, key="logical", dtype=p["dtype"], samples=samples[p["id"]], atol=0, rtol=0) for p in points},
                outputs=[dict(tensor=t, atol=0 if t.dtype in {torch.int64, torch.int32} else 1e-5, rtol=0 if t.dtype in {torch.int64, torch.int32} else 1e-5) for t in outputs])
