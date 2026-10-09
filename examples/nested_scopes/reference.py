"""Independent exact arithmetic and expected key sets, never derived from DATA."""
def reference(inputs, points):
    import torch
    x = inputs[0]
    local = len(x) == 32
    output = torch.empty(32 if local else 69, dtype=torch.int32)
    samples = {p["id"]: [] for p in points}
    for p in points:
        name = p["id"]
        threads = [p["thread"]] if p["thread"] is not None else range(32)
        def add(tx, coords, val, index=0):
            for scope in p["scopes"]:
                if "selected_values" in scope and coords[scope["depth"]:scope["depth"] + len(scope["names"])] != scope["selected_values"]:
                    return
            row = dict(coordinates=coords, index=index, value=val)
            if name != "tail":
                row["thread"] = tx
            samples[name].append(row)
        for tx in threads:
            if name == "local_buf":
                add(tx, [], int(x[tx]), 0)
                add(tx, [], int(x[tx]) + 7, 1)
            elif name == "flag":
                add(tx, [], tx % 2 == 0)
            elif name == "wide":
                add(tx, [], 9007199254740993 + tx)
            elif name in {"f16", "bf16", "f32"}:
                add(tx, [], tx * 0.25 - 3.5)
            else:
                for outer in range(2, 4):
                    if name == "value" and tx % 2 == 0:
                        for inner in range(3, 5):
                            add(tx, [outer, 1, inner], tx * 1000 + 100 + inner)
                    if name == "other" and tx % 4 == 0:
                        add(tx, [outer, 2], tx * 1000 + 2)
                    if name == "odd" and tx % 2:
                        add(tx, [outer], tx + outer)
        if name == "tail":
            if p["thread"] is not None:
                raise ValueError("tail reference does not assume a compiler thread mapping")
            for q in range(37):
                add(None, [q], int(x[q]) + q * 10)
    if local:
        output = x * 2 + 7
    else:
        for tx in range(32):
            if tx % 2:
                output[tx] = 2 * tx + 5
            else:
                output[tx] = 2 * (2 * tx * 1000 + 207) + (2 * (tx * 1000 + 2) if tx % 4 == 0 else 0)
        output[32:] = x + torch.arange(37, dtype=torch.int32) * 10
    specs = {p["id"]: dict(schema=2, key="logical" if p["id"] == "tail" else "execution", dtype=p["dtype"] or "int32",
                            samples=samples[p["id"]], atol=0, rtol=0) for p in points}
    return dict(points=specs, outputs=[dict(tensor=output, atol=0, rtol=0)])
