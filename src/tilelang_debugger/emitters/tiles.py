"""Existing whole-tile Python print and synchronization implementations."""
import math
from .. import capture_state as state


def capture(buffer, point_id, block, loops):
    import tilelang.language as T
    from tvm import tirx
    from tilelang.language.utils import index_to_coordinates

    if state._active is None or point_id not in state._active:
        raise RuntimeError("monitor called outside capture worker")
    p = state._active[point_id]
    actual = dict(shape=[int(x) for x in buffer.shape], dtype=str(buffer.dtype), scope=buffer.scope())
    if actual != dict(shape=p["shape"], dtype=p["dtype"], scope="local.fragment"):
        raise ValueError(f"{point_id}: actual buffer differs from contract: {actual}")
    if p.get("bound"):
        raise ValueError(f"{point_id}: repeated construction/ambiguous lexical binding")
    p.update(bound=True, actual_buffer_name=buffer.name, scope=actual["scope"])
    state._buffers[point_id] = buffer
    if len(block) != 3 or len(loops) != len(p["loop_values"]):
        raise ValueError("monitor identity arity mismatch")
    selector = T.bool(True)
    for x, expected in zip(block, p["block"]):
        selector = T.And(selector, x == expected)
    for x, expected in zip(loops, p["loop_values"]):
        selector = T.And(selector, x == expected)
    bx, by, bz = block
    l0, l1 = (tuple(loops) + (0, 0))[:2]
    count, leader, barrier = p["threads"], p["leader"], p["barrier"]
    elems = math.prod(actual["shape"])
    bits_dtype = "uint16" if actual["dtype"] in ("float16", "bfloat16") else "uint32"
    fmt = f"TLDBG1|{point_id}|0|%d|%d|%d|%d|%d|%d|%u\n"
    fence_regs = p.get("fence_regs", 0)
    full_cta = p.get("engine") == "source"

    @T.macro
    def emit():
        if selector:
            if fence_regs:
                T.warpgroup_fence_operand(buffer, num_regs=fence_regs)
            scratch = T.alloc_shared(buffer.shape, buffer.dtype, "shared")
            T.copy(buffer, scratch)
            if full_cta:
                T.sync_threads()
            else:
                T.sync_threads(barrier, count)
            if T.get_thread_binding() == leader:
                for idx in T.serial(elems):
                    coords = index_to_coordinates(idx, buffer.shape)
                    tirx.call_extern("int32", "printf", fmt,
                                    T.cast(bx, "int32"), T.cast(by, "int32"), T.cast(bz, "int32"),
                                    T.cast(l0, "int32"), T.cast(l1, "int32"), T.cast(idx, "int32"),
                                    T.cast(T.reinterpret(bits_dtype, scratch[coords]), "uint32"))
            if full_cta:
                T.sync_threads()
            else:
                T.sync_threads(barrier, count)

    emit()


def capture_source(buffer, point_id, loops, bounds):
    """Bind source values in the frontend and use the existing patched print structure."""
    import tilelang.language as T
    from tilelang.language.kernel import KernelLaunchFrame
    from tilelang.language.utils import index_to_coordinates
    from tvm import tirx

    if state._active is None or point_id not in state._active:
        raise ValueError("source observation outside active session")
    p = state._active[point_id]
    frame = KernelLaunchFrame.Current()
    if frame is None or frame.get_thread_extents()[1:] != [1, 1]:
        raise ValueError("source capture needs a one-dimensional CTA")
    if p.get("bound"):
        raise ValueError("observation constructed more than once")
    # int(Var) is intentionally rejected: the entire loop domain must be static.
    static_bounds = [[int(v) for v in b] for b in bounds]
    if len(static_bounds) != len(p["loops"]):
        raise ValueError("loop identity arity differs")
    for b, selection, expected in zip(static_bounds, p["loops"], p["loop_values"]):
        domain = range(*b)
        if selection["iteration"] > len(domain) or domain[selection["iteration"] - 1] != expected:
            raise ValueError("selected iteration is outside the static source loop")
    block_vars = frame.get_block_bindings()
    grid = [frame.get_block_extent(i) for i in range(len(block_vars))]
    grid += [1] * (3 - len(grid))
    if any(b >= e for b, e in zip(p["block"], grid)):
        raise ValueError("selected block is outside actual launch grid")
    shape, dtype, scope = [int(s) for s in buffer.shape], str(buffer.dtype), buffer.scope()
    if not shape or any(s <= 0 for s in shape) or dtype not in {"float16", "bfloat16", "float32", "int32"}:
        raise ValueError("capture needs positive static shape and supported dtype")
    if scope not in {"local.fragment", "global"}:
        raise ValueError("source capture supports fragments and readonly global inputs")
    if scope == "global" and not p["global_readonly_syntax"]:
        raise ValueError("global input observation is outside simple readonly source syntax")
    p.update(shape=shape, dtype=dtype, scope=scope, grid=grid, threads=frame.get_num_threads(),
             leader=0, barrier=0, actual_loop_bounds=static_bounds)
    if sum(math.prod(q.get("shape", [0])) for q in state._active.values()) > 65536:
        raise ValueError("capture exceeds 65536-element budget")
    block = tuple(block_vars) + (0,) * (3 - len(block_vars))
    if scope == "local.fragment":
        capture(buffer, point_id, block, loops)
        return
    p.update(bound=True, actual_buffer_name=buffer.name)
    state._buffers[point_id] = buffer
    selector = T.bool(True)
    for v, expected in zip(block, p["block"]):
        selector = T.And(selector, v == expected)
    for v, expected in zip(loops, p["loop_values"]):
        selector = T.And(selector, v == expected)
    bx, by, bz = block
    l0, l1 = (tuple(loops) + (0, 0))[:2]
    elems = math.prod(shape)
    bits_dtype = "uint16" if dtype in ("float16", "bfloat16") else "uint32"
    fmt = f"TLDBG1|{point_id}|0|%d|%d|%d|%d|%d|%d|%u\n"

    @T.macro
    def emit_global():
        if selector:
            T.sync_threads()
            if T.get_thread_binding() == 0:
                for idx in T.serial(elems):
                    coords = index_to_coordinates(idx, buffer.shape)
                    tirx.call_extern("int32", "printf", fmt,
                                    T.cast(bx, "int32"), T.cast(by, "int32"), T.cast(bz, "int32"),
                                    T.cast(l0, "int32"), T.cast(l1, "int32"), T.cast(idx, "int32"),
                                    T.cast(T.reinterpret(bits_dtype, buffer[coords]), "uint32"))
            T.sync_threads()
    emit_global()
