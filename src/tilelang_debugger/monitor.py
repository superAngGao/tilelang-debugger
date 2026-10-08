"""Existing TileLang macros/intrinsics only; no added lowering pass."""
import math
from contextlib import contextmanager

_active = None
_buffers = None


@contextmanager
def session(points):
    global _active, _buffers
    if _active is not None:
        raise RuntimeError("nested monitor session")
    state = {p["id"]: dict(p) for p in points}
    _active = state
    _buffers = {}
    try:
        yield state
        if any(not p.get("bound") for p in state.values()):
            raise RuntimeError("an observation point was not built")
    finally:
        _active = None
        _buffers = None


def buffer_bindings():
    return dict(_buffers or {})


def capture(buffer, point_id, block, loops):
    import tilelang.language as T
    from tvm import tirx
    from tilelang.language.utils import index_to_coordinates

    if _active is None or point_id not in _active:
        raise RuntimeError("monitor called outside capture worker")
    p = _active[point_id]
    actual = dict(shape=[int(x) for x in buffer.shape], dtype=str(buffer.dtype), scope=buffer.scope())
    if actual != dict(shape=p["shape"], dtype=p["dtype"], scope="local.fragment"):
        raise ValueError(f"{point_id}: actual buffer differs from contract: {actual}")
    if p.get("bound"):
        raise ValueError(f"{point_id}: repeated construction/ambiguous lexical binding")
    p.update(bound=True, actual_buffer_name=buffer.name, scope=actual["scope"])
    _buffers[point_id] = buffer
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

    @T.macro
    def emit():
        if selector:
            if fence_regs:
                T.warpgroup_fence_operand(buffer, num_regs=fence_regs)
            scratch = T.alloc_shared(buffer.shape, buffer.dtype, "shared")
            T.copy(buffer, scratch)
            T.sync_threads(barrier, count)
            if T.get_thread_binding() == leader:
                for idx in T.serial(elems):
                    coords = index_to_coordinates(idx, buffer.shape)
                    tirx.call_extern("int32", "printf", fmt,
                                    T.cast(bx, "int32"), T.cast(by, "int32"), T.cast(bz, "int32"),
                                    T.cast(l0, "int32"), T.cast(l1, "int32"), T.cast(idx, "int32"),
                                    T.cast(T.reinterpret(bits_dtype, scratch[coords]), "uint32"))
            T.sync_threads(barrier, count)

    emit()
