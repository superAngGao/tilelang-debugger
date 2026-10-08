"""Export the object actually compiled, and inspect synchronization before launch."""
import hashlib
import json
import math
import operator
import re
from collections import Counter

KNOWN_OPS = set("tvm_access_ptr type_annotation tvm_storage_sync reinterpret if_then_else exp2 infinity bitwise_xor shift_right handle_add_byte_offset tl_shuffle_elect set_max_nreg prefetch_tma_descriptor ptx_init_barrier_thread_count ptx_fence_barrier_init mbarrier_expect_tx mbarrier_wait_parity ptx_arrive_barrier tma_load tma_store tma_store_arrive tma_store_wait initialize_wgmma_descriptor increase_descriptor_offset ptx_wgmma_ss ptx_wgmma_rs warpgroup_arrive warpgroup_commit_batch warpgroup_fence_operand wait_wgmma named_barrier_arrive fence_proxy_async address_of bitwise_and shift_left call_extern".split())
REDUCE = re.compile(r"tl::AllReduce<tl::(?:Sum|Max)Op, (?:4|64|128), 1, (?:0|128), tl::NamedBarrier<128>>::run")


def validate_call(node):
    name = node.op.name
    if name.split(".")[-1] not in KNOWN_OPS or name.split(".")[0] not in ("tl", "tirx"):
        raise ValueError(f"unreviewed lowered call: {name}")
    if name.endswith("call_extern"):
        external = str(node.args[0].value)
        if external != "printf" and not REDUCE.fullmatch(external):
            raise ValueError(f"unreviewed external helper: {external}")


def plain(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, "items"):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)) or type(obj).__name__ == "Array":
        return [plain(x) for x in obj]
    return str(obj)


def walk_statements(node, visit, guards=(), loops=()):
    """Retain branch polarity; post_order_visit alone loses participant domains."""
    from tvm import tirx
    if isinstance(node, tirx.SeqStmt):
        for child in node.seq:
            walk_statements(child, visit, guards, loops)
    elif isinstance(node, tirx.IfThenElse):
        walk_statements(node.then_case, visit, guards + (node.condition,), loops)
        if node.else_case is not None:
            walk_statements(node.else_case, visit, guards + (tirx.Not(node.condition),), loops)
    elif isinstance(node, tirx.For):
        walk_statements(node.body, visit, guards, loops + (node,))
    elif isinstance(node, tirx.Evaluate):
        visit(node.value, guards, loops)
    elif isinstance(node, tirx.BufferStore):
        visit(node, guards, loops)
    elif isinstance(node, (tirx.AllocBuffer, tirx.DeclBuffer, tirx.Bind)):
        pass
    elif hasattr(node, "body"):
        walk_statements(node.body, visit, guards, loops)
    elif hasattr(node, "block"):
        block = node.block
        if block.init is not None:
            walk_statements(block.init, visit, guards, loops)
        walk_statements(block.body, visit, guards, loops)
    else:
        raise ValueError(f"unhandled device IR statement {type(node).__name__}")


def inspect_device(module):
    from tvm import tirx
    events, variables, allocations = [], {}, {}
    funcs = [f for f in module.functions.values() if isinstance(f, tirx.PrimFunc)]
    if len(funcs) != 1:
        raise ValueError("expected exactly one device function")
    func = funcs[0]

    def inventory(node):
        if isinstance(node, tirx.Call):
            validate_call(node)
        if isinstance(node, tirx.Var):
            variables[node.name] = node
        if isinstance(node, tirx.AllocBuffer):
            allocations[node.buffer.data] = (str(node.buffer.dtype), [int(x) for x in node.buffer.shape])
        if isinstance(node, tirx.AttrStmt) and node.attr_key == "thread_extent":
            variables[node.node.thread_tag] = node.node.var
    tirx.stmt_functor.post_order_visit(func.body, inventory)

    def visit(value, guards, loops):
        if isinstance(value, tirx.BufferStore):
            events.append(dict(kind="store", buffer=value.buffer.name, guards=guards, loops=loops, expr=value))
            return
        if not isinstance(value, tirx.Call):
            return
        name = value.op.name
        args = list(value.args)
        if name.endswith("call_extern"):
            name = str(args.pop(0).value)
        event = dict(kind=name, args=args, guards=guards, loops=loops, expr=value)
        if name.endswith("tvm_storage_sync"):
            if len(args) == 1:
                event.update(kind="cta_sync", id=0, count=0)
            elif len(args) == 3:
                event.update(kind="named_sync", id=int(args[1]), count=int(args[2]))
            else:
                raise ValueError("unknown storage sync form")
        elif "__sync_thread_partial" in name or "named_barrier_arrive" in name:
            event.update(kind="named_arrive" if "arrive" in name else "named_sync", id=int(args[0]), count=int(args[1]))
        elif "asm" in name:
            raise ValueError("unreviewed inline asm in device IR")
        events.append(event)
    walk_statements(func.body, visit)
    return events, variables, allocations


def manifest(events):
    return [dict(kind=e["kind"], id=e["id"], count=e["count"],
                 guards=[str(g) for g in e["guards"]]) for e in events if "id" in e]


def protocol(events, excluded=(), omit_cta=False, renames=None):
    def canonical(text):
        for modified, original in (renames or {}).items():
            text = re.sub(r"\b" + re.escape(modified) + r"\b", original, text)
        return text
    result = []
    for i, e in enumerate(events):
        if i in excluded or (omit_cta and e["kind"] == "cta_sync"):
            continue
        if e["kind"] == "store" or e["kind"] == "printf":
            continue
        result.append(dict(kind=e["kind"], args=[canonical(str(a)) for a in e["args"]],
                           guards=[str(g) for g in e["guards"]],
                           loops=[dict(var=l.loop_var.name, min=str(l.min), extent=str(l.extent), kind=str(l.kind)) for l in e["loops"]]))
    return result


def export(kernel, prim_func, output, out_idx):
    if kernel.artifact is None or kernel.artifact.device_mod is None:
        raise ValueError("actual compiled device IR unavailable; cache must be disabled")
    (output / "frontend.py").write_text(prim_func.script(), encoding="utf-8")
    (output / "device.py").write_text(kernel.artifact.device_mod.script(), encoding="utf-8")
    source = kernel.get_kernel_source()
    (output / "kernel.cu").write_text(source, encoding="utf-8")
    events, variables, allocations = inspect_device(kernel.artifact.device_mod)
    cuda_named = []
    for text, kind in (("tl::__sync_thread_partial", "named_sync"), ("tl::__named_barrier_arrive", "named_arrive")):
        matches = re.findall(re.escape(text) + r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", source)
        if len(matches) != source.count(text + "("):
            raise ValueError("dynamic/unrecognized named barrier in generated CUDA")
        cuda_named.extend((kind, int(b), int(c)) for b, c in matches)
    ir_named = [(e["kind"], e["id"], e["count"]) for e in events if e["kind"] in ("named_sync", "named_arrive")]
    if Counter(cuda_named) != Counter(ir_named) or re.search(r"\basm\s*(?:volatile\s*)?\(", source):
        raise ValueError("generated CUDA named barriers/inline asm differ from reviewed IR resources")
    config = dict(target=str(kernel.target), execution_backend=kernel.execution_backend,
                  pass_configs=plain(kernel.pass_configs), compile_flags=plain(kernel.compile_flags), out_idx=plain(out_idx))
    func = next(iter(kernel.artifact.device_mod.functions.values()))
    data = dict(config=config, thread_extent=plain(func.attrs["thread_extent"]), barriers=manifest(events), protocol=protocol(events),
                cuda_named_barriers=cuda_named, reserved_helper_barriers=[0,1,2],
                allocations=[dict(name=v.name, dtype=d, shape=s) for v, (d,s) in allocations.items()])
    (output / "compile.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data, events, variables, allocations, source


def participants(event, variables, point, cta_threads):
    import tvm
    from tvm import tirx
    values = {"blockIdx.x": point["block"][0], "blockIdx.y": point["block"][1], "blockIdx.z": point["block"][2],
              "threadIdx.y": 0, "threadIdx.z": 0, **dict(zip(point["loop_vars"], point["loop_values"]))}
    result = []
    for tx in range(cta_threads):
        mapping = {var: tirx.IntImm(var.dtype, tx if name == "threadIdx.x" else values[name])
                   for name, var in variables.items() if name == "threadIdx.x" or name in values}
        active = True
        for guard in event["guards"]:
            simplified = tvm.arith.Analyzer().simplify(tirx.stmt_functor.substitute(guard, mapping))
            if not isinstance(simplified, tirx.IntImm):
                raise ValueError(f"unproved monitor participant condition: {simplified}")
            active &= bool(int(simplified))
        if active:
            result.append(tx)
    return result


def check_instrumented(data, events, variables, allocations, source, baseline, points, cta_threads, allow_auto_cta_change=False, bindings=None, observations=None):
    from tvm import tirx
    if data["config"] != baseline["config"]:
        raise ValueError("baseline/instrumented effective compile settings differ")
    new_ids = {p["barrier"] for p in points}
    original = [e for e in data["barriers"] if e["id"] not in new_ids]
    # Preserve the full original synchronization call sequence and guards.
    auto_change = original != baseline["barriers"]
    if auto_change:
        if not allow_auto_cta_change or data["config"]["pass_configs"].get("tl.disable_thread_storage_sync", False):
            raise ValueError("original barrier uses changed; requires a reviewed lowering contract")
        if any(e["kind"] != "cta_sync" or e["guards"] for e in original + baseline["barriers"]):
            raise ValueError("only unconditional automatic CTA barriers may change")
        if any(any(word in e["kind"].lower() for word in ("async", "tma", "mbarrier", "wgmma")) for e in list(events) + baseline["protocol"]):
            raise ValueError("automatic CTA barrier exception cannot contain asynchronous operations")
    excluded, observed_layouts = set(), {}
    for p in points:
        matches = [(i, e) for i, e in enumerate(events) if e.get("id") == p["barrier"]]
        prints = [(i, e) for i, e in enumerate(events) if e["kind"] == "printf" and str(e["args"][0].value).startswith(f"TLDBG1|{p['id']}|")]
        if len(matches) != 2 or len(prints) != 1:
            raise ValueError(f"{p['id']}: ambiguous/missing monitor barrier or print")
        (first, b0), (last, b1) = matches
        excluded.update((first, last))
        pi, pe = prints[0]
        if not first < pi < last:
            raise ValueError("print not enclosed by monitor barriers")
        expected = list(range(p["leader"], p["leader"] + p["threads"]))
        check_monitor_loops(b0, b1, pe, p)
        for b in (b0, b1):
            if b["kind"] != "named_sync" or b["count"] != p["threads"] or participants(b, variables, p, cta_threads) != expected:
                raise ValueError("monitor barrier participant domain differs from contract")
        if participants(pe, variables, p, cta_threads) != [p["leader"]]:
            raise ValueError("monitor printing leader differs from contract")
        loads = []
        tirx.stmt_functor.post_order_visit(pe["args"][-1], lambda n: loads.append(n) if isinstance(n, tirx.BufferLoad) else None)
        if len(loads) != 1 or loads[0].buffer.scope() != "shared":
            raise ValueError("printf must read exactly one private shared staging element")
        scratch = loads[0].buffer
        check_print_read(pe, loads[0], variables, p)
        stores = [(i, e) for i, e in enumerate(events) if e["kind"] == "store" and e["expr"].buffer.data.same_as(scratch.data)]
        if len(stores) != 1 or not stores[0][0] < first:
            raise ValueError("scratch must have one complete staging loop before the first barrier")
        si, store = stores[0]
        writers = participants(store, variables, p, cta_threads)
        if not writers or not set(writers).issubset(expected):
            raise ValueError("staging writers lie outside barrier participants")
        source_loads = []
        tirx.stmt_functor.post_order_visit(store["expr"].value, lambda n: source_loads.append(n) if isinstance(n, tirx.BufferLoad) else None)
        if len(source_loads) != 1 or source_loads[0].buffer.scope() != "local":
            raise ValueError("staging must copy directly from the selected local fragment")
        if not store["expr"].value.same_as(source_loads[0]):
            raise ValueError("staging must not transform fragment values")
        if bindings is None:
            raise ValueError("missing frontend buffer bindings")
        expected_name = bindings[p["id"]].data.name
        candidates = [v for v in allocations if v.name == expected_name]
        if len(candidates) != 1 or not source_loads[0].buffer.data.same_as(candidates[0]):
            raise ValueError("staging source is not the unique lowered allocation of this frontend buffer")
        observed_layouts[p["id"]] = check_staging_coverage(store, source_loads[0], allocations, variables, p, writers)
        if str(source_loads[0].dtype).split("x")[0] != p["dtype"] or str(scratch.dtype) != p["dtype"]:
            raise ValueError("staging altered the selected dtype")
        if p.get("fence_regs"):
            # Find the exact accumulator pointer used by the monitor fence, then
            # check its actual lowered local allocation, not merely a literal 64.
            fences = [(i, e) for i, e in enumerate(events[:first]) if "warpgroup_fence_operand" in e["kind"]]
            if not fences:
                raise ValueError("missing monitor operand fence")
            fi, fence = fences[-1]
            excluded.add(fi)
            if participants(fence, variables, p, cta_threads) != expected:
                raise ValueError("operand fence in wrong group")
            args = fence["args"]
            # Lowered intrinsic: dtype, pointer, offset, register count.
            ptr = args[1]
            alloc = allocations.get(ptr)
            if alloc != ("float32", [p["fence_regs"]]) or int(args[-1]) != p["fence_regs"] or int(args[-2]) != 0:
                raise ValueError(f"cannot prove actual accumulator fence coverage: {fence['expr']} {alloc}")
            if not source_loads[0].buffer.data.same_as(ptr) or not fi < si:
                raise ValueError("fence does not cover the accumulator actually staged")
            if not p["loop_vars"]:  # GQA prologue; GEMM ring-dispatch waits checked by preserved protocol + source contract.
                waits = [(i, e) for i, e in enumerate(events[:fi]) if e["kind"].endswith("wait_wgmma")]
                if not waits or int(waits[-1][1]["args"][0]) != 0 or participants(waits[-1][1], variables, p, cta_threads) != expected:
                    raise ValueError("monitor fence is not after the same group's wait(0)")
    remaining = protocol(events, excluded, omit_cta=auto_change, renames={p["instrumented_buffer"]: p["buffer"] for p in points if "instrumented_buffer" in p})
    original_protocol = [e for e in baseline["protocol"] if not (auto_change and e["kind"] == "cta_sync")]
    cursor, added_proxy_fences = 0, []
    for entry in remaining:
        if cursor < len(original_protocol) and entry == original_protocol[cursor]:
            cursor += 1
        elif entry["kind"] == "tl.fence_proxy_async" and not entry["args"]:
            # Existing InjectFenceProxy may add an ordering fence because the
            # monitor introduces generic shared stores. It does not wait for
            # async completion or change a named-barrier participant protocol.
            candidates = [e for e in events if protocol([e]) == [entry]]
            if not candidates or not any(participants(e, variables, p, cta_threads) == list(range(p["leader"],p["leader"]+p["threads"])) for e in candidates for p in points):
                raise ValueError("new proxy fence has an unproved participant domain")
            added_proxy_fences.append(entry)
        else:
            raise ValueError("original asynchronous/synchronization protocol or loop context changed")
    if cursor != len(original_protocol):
        raise ValueError("an original asynchronous/synchronization operation disappeared")
    if observations is not None:
        observations.write_text(json.dumps(dict(passed=False, observed_layouts=observed_layouts,
                                                note="Diagnostic only; must match an independently reviewed layout contract before launch"), indent=2))
    for p in points:
        if observed_layouts[p["id"]] not in p.get("layout_sha256", []):
            raise ValueError(f"{p['id']}: logical element/writer/local-index mapping has no reviewed layout contract")
    return dict(passed=True, automatic_cta_barrier_change=auto_change,
                layout_sha256=observed_layouts,
                added_proxy_fences=added_proxy_fences,
                automatic_cta_barrier_before=baseline["barriers"] if auto_change else [],
                automatic_cta_barrier_after=original if auto_change else [],
                checks=["effective_config", "original_barriers", "unique_monitor_ids", "barrier_domains", "leader", "staging_coverage", "fence_pointer_and_coverage"])


def check_monitor_loops(first, last, printed, point):
    def signature(loops):
        return [(l.loop_var.name, int(l.min), int(l.extent), int(l.kind)) for l in loops]
    required = [(name, 0, extent, 0) for name, extent in zip(point["loop_vars"], point.get("loop_extents", []))]
    if len(required) != len(point["loop_vars"]):
        raise ValueError("incomplete reviewed enclosing-loop contract")
    if signature(first["loops"]) != required or signature(last["loops"]) != required or signature(printed["loops"][:-1]) != required:
        raise ValueError("monitor collective has an unexpected or nonuniform loop context")
    for a, b, c in zip(first["loops"], last["loops"], printed["loops"]):
        if not a.loop_var.same_as(b.loop_var) or not a.loop_var.same_as(c.loop_var):
            raise ValueError("monitor barriers and print belong to different loop instances")


def check_print_read(event, load, variables, point):
    from tvm import tirx
    extra = [l for l in event["loops"] if l.loop_var.name not in point["loop_vars"]]
    if len(extra) != 1:
        raise ValueError("printf requires one complete element loop")
    loop = extra[0]
    if int(loop.min) != 0 or int(loop.extent) != math.prod(point["shape"]):
        raise ValueError("printf element loop is incomplete")
    if len(load.indices) != 1 or not load.indices[0].same_as(loop.loop_var) or not event["args"][-2].same_as(loop.loop_var):
        raise ValueError("printf element identity does not match the actual shared read")
    bit_expr = event["args"][-1]
    width = 16 if point["dtype"] in ("float16", "bfloat16") else 32
    if str(bit_expr.dtype) != "uint32":
        raise ValueError("printf %u requires uint32")
    if width == 16:
        if not isinstance(bit_expr, tirx.Cast):
            raise ValueError("16-bit values must be zero-extended after reinterpret")
        bit_expr = bit_expr.value
    if not isinstance(bit_expr, tirx.Call) or not bit_expr.op.name.endswith("reinterpret") or str(bit_expr.dtype) != f"uint{width}" or len(bit_expr.args) != 1 or not bit_expr.args[0].same_as(load):
        raise ValueError("printf raw bits were converted before reinterpret")


def index_value(expr, values):
    """Small exact integer evaluator for the reviewed, lowered staging layouts."""
    from tvm import tirx
    if isinstance(expr, tirx.IntImm):
        return int(expr)
    if isinstance(expr, tirx.Var):
        return values[expr]
    if isinstance(expr, tirx.Cast):
        return index_value(expr.value, values)
    if isinstance(expr, tirx.Ramp):
        base, step = index_value(expr.base, values), index_value(expr.stride, values)
        return [base + i * step for i in range(int(expr.lanes))]
    operations = {"Add": operator.add, "Sub": operator.sub, "Mul": operator.mul,
                  "FloorDiv": operator.floordiv, "FloorMod": operator.mod}
    if type(expr).__name__ in operations:
        return operations[type(expr).__name__](index_value(expr.a, values), index_value(expr.b, values))
    raise ValueError(f"unproved staging index expression: {expr}")


def check_staging_coverage(store, source_load, allocations, variables, point, writers):
    selected = {"blockIdx.x": point["block"][0], "blockIdx.y": point["block"][1], "blockIdx.z": point["block"][2],
                "threadIdx.y": 0, "threadIdx.z": 0, **dict(zip(point["loop_vars"], point["loop_values"]))}
    values = {v: selected[k] for k, v in variables.items() if k in selected}
    loops = [l for l in store["loops"] if l.loop_var not in values]
    counts = Counter()
    source_counts = Counter()
    mapping = []
    local_size = math.prod(allocations[source_load.buffer.data][1])
    current_thread = None
    def enumerate_loops(depth):
        if depth == len(loops):
            if len(store["expr"].indices) != 1:
                raise ValueError("expected flattened staging index")
            indexes = index_value(store["expr"].indices[0], values)
            indexes = indexes if isinstance(indexes, list) else [indexes]
            counts.update(indexes)
            if len(source_load.indices) != 1:
                raise ValueError("expected flattened local fragment")
            source_indexes = index_value(source_load.indices[0], values)
            source_indexes = source_indexes if isinstance(source_indexes, list) else [source_indexes]
            if len(indexes) != len(source_indexes):
                raise ValueError("staging source/destination vector widths differ")
            source_counts.update((current_thread, i) for i in source_indexes)
            mapping.extend((dst, current_thread, src) for dst, src in zip(indexes, source_indexes))
            return
        loop = loops[depth]
        start, extent = index_value(loop.min, values), index_value(loop.extent, values)
        if not 0 <= extent <= math.prod(point["shape"]):
            raise ValueError("unbounded staging loop")
        for i in range(start, start + extent):
            values[loop.loop_var] = i
            enumerate_loops(depth + 1)
    for tx in writers:
        current_thread = tx
        values[variables["threadIdx.x"]] = tx
        enumerate_loops(0)
    if counts != Counter({i: 1 for i in range(math.prod(point["shape"]))}):
        raise ValueError("staging is not a complete one-write-per-element tile")
    if source_counts != Counter({(tx, i): 1 for tx in writers for i in range(local_size)}):
        raise ValueError("staging source indices are not a complete, in-bounds local fragment")
    return layout_digest(mapping)


def layout_digest(mapping):
    return hashlib.sha256(json.dumps(sorted(mapping), separators=(",", ":")).encode()).hexdigest()
