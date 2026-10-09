"""Direct source expression/local-buffer printf; no collective or lowering hook."""
import math

from .. import capture_state as monitor

from ..protocols.dtypes import SAMPLE_WIDTH as WIDTH


def active(point_id):
    if monitor._active is None or point_id not in monitor._active:
        raise ValueError("observation outside active session")
    return monitor._active[point_id]


def budget():
    total = sum(p.get("event_budget", 0) for p in monitor._active.values())
    if total > 65536:
        raise ValueError("capture exceeds 65536-event budget (including witnesses)")


def observe_root(point_id, bounds):
    from tilelang.language.kernel import KernelLaunchFrame
    p = active(point_id)
    if p.get("root_built"):
        raise ValueError("point root constructed more than once")
    frame = KernelLaunchFrame.Current()
    if frame is None or frame.get_thread_extents()[1:] != [1, 1]:
        raise ValueError("samples require a one-dimensional CTA")
    threads = frame.get_num_threads()
    grid = [int(frame.get_block_extent(i)) for i in range(len(frame.get_block_bindings()))]
    grid += [1] * (3 - len(grid))
    if any(v >= e for v, e in zip(p["block"], grid)) or (p["thread"] is not None and p["thread"] >= threads):
        raise ValueError("selected block/thread outside launch")
    actual = [[int(v) for v in b] for b in bounds]
    cursor, volume = 0, 1
    witnesses = 1
    for s in p["scopes"]:
        if s["kind"] == "branch":
            witnesses += volume
            continue
        ranges = []
        for _ in s["names"]:
            b = actual[cursor]
            cursor += 1
            if b[2] <= 0 or any(not -(2**31) <= v < 2**31 for v in b):
                raise ValueError("counted loops need positive step and int32 bounds")
            domain = range(*b)
            if len(domain) > 65536:
                raise ValueError("loop extent exceeds event budget")
            ranges.append(b)
            volume *= len(domain)
        s["actual_bounds"] = ranges
        if "selection" in s:
            if s["kind"] == "serial":
                domain = range(*ranges[0])
                if s["selection"] > len(domain):
                    raise ValueError("selected iteration outside loop")
                s["selected_values"] = [domain[s["selection"] - 1]]
            else:
                if any(v not in range(*b) for v, b in zip(s["selection"], ranges)):
                    raise ValueError("selected coordinate outside Parallel")
                s["selected_values"] = s["selection"]
    p.update(root_built=True, threads=threads, grid=grid, dtype=None, shape=[], scope="inactive",
             witness_budget=witnesses * threads, volume=volume,
             event_budget=(witnesses + volume) * threads, built_branches=[])
    budget()
    emit(p, "R", -1, (), 0, 0)


def observe_branch(point_id, scope, arm, coords):
    p = active(point_id)
    p["built_branches"].append([scope, arm])
    emit(p, "B", scope, coords, arm, 0)


def observe_value(value, point_id, coords):
    import tilelang.language as T
    from tvm import tirx
    from tilelang.language.utils import index_to_coordinates
    p = active(point_id)
    if p.get("bound"):
        raise ValueError("point constructed more than once")
    if isinstance(value, tirx.Buffer):
        scope, shape = value.scope(), [int(x) for x in value.shape]
        if scope not in {"local", "local.var"}:
            raise ValueError(f"samples emitter does not support {scope}; collective tile observations use the existing tile path")
        if not shape or any(x <= 0 for x in shape):
            raise ValueError("local buffer requires positive static shape")
    elif isinstance(value, tirx.PrimExpr):
        scope, shape = "scalar", []
    else:
        raise ValueError("observe an existing PrimExpr or local Buffer")
    dtype = str(value.dtype)
    if dtype not in WIDTH:
        raise ValueError(f"unsupported scalar dtype: {dtype}")
    count = math.prod(shape)
    p.update(bound=True, scope=scope, shape=shape, dtype=dtype, elements=count,
             event_budget=p["witness_budget"] + p["volume"] * p["threads"] * count)
    budget()
    if scope == "scalar":
        emit(p, "D", -1, coords, 0, value)
    else:
        @T.macro
        def local():
            for index in T.serial(count):
                emit(p, "D", -1, coords, index, value[index_to_coordinates(index, shape)])
        local()


def emit(p, kind, scope, coords, index, value):
    import tilelang.language as T
    from tilelang.language.kernel import KernelLaunchFrame
    from tvm import tirx
    frame = KernelLaunchFrame.Current()
    block = tuple(frame.get_block_bindings())
    block += (0,) * (3 - len(block))
    tx = T.get_thread_binding()
    selector = T.bool(True)
    for v, expected in zip(block, p["block"]):
        selector = T.And(selector, v == expected)
    if p["thread"] is not None:
        selector = T.And(selector, tx == p["thread"])
    # Witnesses are deliberately unfiltered by deeper selected iterations.
    if kind == "D":
        for s in p["scopes"]:
            if "selected_values" in s:
                for d, expected in enumerate(s["selected_values"]):
                    selector = T.And(selector, coords[s["depth"] + d] == expected)
    low, high = T.uint32(0), T.uint32(0)
    if kind == "D":
        dtype = p["dtype"]
        if dtype == "bool":
            low = T.cast(value, "uint32")
        elif dtype == "int64":
            raw = T.reinterpret("uint64", value)
            low, high = T.cast(raw, "uint32"), T.cast(raw >> 32, "uint32")
        else:
            low = T.cast(T.reinterpret("uint16" if WIDTH[dtype] == 16 else "uint32", value), "uint32")
    args = [T.cast(v, "int32") for v in (*block, tx, *coords, index)] + [low, high]
    fmt = f"TLDBG2|{p['id']}|{kind}|{scope}|0|" + "|".join(["%d"] * (len(args) - 2) + ["%u", "%u"]) + "|E\n"
    @T.macro
    def output():
        if selector:
            tirx.call_extern("int32", "printf", fmt, *args)
    output()
