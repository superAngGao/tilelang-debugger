"""CPU partial-K and full GEMM references; no captured values used as truth."""


def reference(inputs, points):
    a, b = inputs
    result = {}
    for p in points:
        if p["buffer"] != "c_local":
            raise ValueError("unknown GEMM buffer")
        end_k = p["loops"][0]["iteration"] * 64
        row, col = p["block"][1]*128, p["block"][0]*128
        tile = a[row:row+128, :end_k].float() @ b[col:col+128, :end_k].float().T
        result[p["id"]] = dict(tensor=tile, atol=0.001, rtol=0.0001)
    return dict(points=result, outputs=[dict(tensor=a.float() @ b.float().T, atol=0.0625, rtol=0.002)])
