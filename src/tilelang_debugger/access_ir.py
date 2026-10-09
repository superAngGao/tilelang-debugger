"""Append index-only printf after the original CUDA pipeline, scoped to one compile."""
from contextlib import contextmanager
import importlib
import itertools
import json
from .capture import save_json
from .instrument import digest


def walk(node, visit, loops=(), guards=()):
    from tvm import tirx as T
    if isinstance(node, T.SeqStmt):
        for child in node.seq:
            walk(child, visit, loops, guards)
    elif isinstance(node, T.For):
        walk(node.body, visit, loops + (node,), guards)
    elif isinstance(node, T.IfThenElse):
        walk(node.then_case, visit, loops, guards + (node.condition,))
        if node.else_case is not None:
            walk(node.else_case, visit, loops, guards + (T.Not(node.condition),))
    elif isinstance(node, (T.BufferStore, T.Evaluate, T.Bind, T.AllocBuffer, T.DeclBuffer)):
        visit(node, loops, guards)
    elif isinstance(node, T.AttrStmt):
        walk(node.body, visit, loops, guards)
    else:
        raise ValueError(f"unsupported access statement: {type(node).__name__}")


def evaluate(expr, bindings):
    import tvm
    from tvm import tirx as T
    mapping = {v: T.IntImm(v.dtype, n) for v, n in bindings.items()}
    value = tvm.arith.Analyzer().simplify(T.stmt_functor.substitute(expr, mapping))
    if not isinstance(value, T.IntImm):
        raise ValueError(f"unresolved integer expression: {value}")
    return int(value)


def pure(expr):
    """No data loads, side effects, vector extraction or opaque calls in the logger."""
    from tvm import tirx as T
    allowed = {"IntImm", "Var", "SizeVar", "Cast", "Add", "Sub", "Mul", "FloorDiv", "FloorMod",
               "Div", "Mod", "Min", "Max", "LT", "LE", "GT", "GE", "EQ", "NE", "And", "Or", "Not"}
    def check(n):
        if type(n).__name__ not in allowed:
            raise ValueError(f"unsafe index expression: {type(n).__name__}")
    T.stmt_functor.post_order_visit(expr, check)
    if str(expr.dtype).split("x")[0] not in ("bool", "int32", "uint32", "int64", "uint64"):
        raise ValueError("unsupported index dtype")


def encode_words(values):
    """Preserve original expression width, then sign/zero extend and split for printf."""
    from tvm import tirx as T
    words=[]
    for v in values:
        dtype="uint64" if str(v.dtype).startswith("uint") else "int64"
        bits=T.Cast(dtype,v)
        mask=T.IntImm(dtype,0xffffffff)
        words += [T.Cast("uint32",T.bitwise_and(bits,mask)),
                  T.Cast("uint32",T.bitwise_and(T.shift_right(bits,T.IntImm(dtype,32)),mask))]
    return words


def scalar_indices(access):
    from tvm import tirx as T
    if getattr(access, "predicate", None) is not None or len(access.indices) != 1:
        raise ValueError("expected flattened, predicate-free access")
    x = access.indices[0]
    if isinstance(x, T.Ramp):
        return [x.base + x.stride * i for i in range(int(x.lanes))]
    return [x]


def rewrite(node,replace):
    """Rebuild only statement containers; avoid ir_transform flattening inserted SeqStmt."""
    from tvm import tirx as T
    changed=replace(node)
    if changed is not None:return changed
    if isinstance(node,T.SeqStmt):return T.SeqStmt([rewrite(c,replace) for c in node.seq])
    if isinstance(node,T.For):return T.For(node.loop_var,node.min,node.extent,node.kind,rewrite(node.body,replace),node.thread_binding,node.annotations,node.step)
    if isinstance(node,T.IfThenElse):return T.IfThenElse(node.condition,rewrite(node.then_case,replace),rewrite(node.else_case,replace) if node.else_case is not None else None)
    if isinstance(node,T.AttrStmt):return T.AttrStmt(node.node,node.attr_key,node.value,rewrite(node.body,replace))
    return node


def comparison_view(module):
    """Var-keyed metadata maps fail TVM structural equality even after JSON roundtrip.

    Normalize ONLY map keys in a comparison copy; arguments retain their Var objects,
    and each key must be the same Var as descriptor argument 1. Execution IR is untouched.
    """
    import tvm
    from tvm import tirx as T
    result=tvm.IRModule(dict(module.functions),attrs=module.attrs)
    for gv,f in module.functions.items():
        metadata=f.attrs.get("tma_descriptor_args")
        if metadata is not None:
            normalized={}
            for key,args in metadata.items():
                if not isinstance(key,T.Var) or key.name in normalized or not key.same_as(args[1]):
                    raise ValueError("invalid descriptor map identity")
                normalized[key.name]=args
            result.update_func(gv,f.with_attr("tma_descriptor_args",normalized))
    return result


def erase_codegen_trace(module,points):
    import tvm
    from tvm import tirx as T
    counts={p["site"]:0 for p in points}
    widths={p["site"]:p["vector_width"] for p in points}
    def clean(n):
        if isinstance(n,T.Evaluate) and isinstance(n.value,T.Call) and n.value.op.name=="tirx.call_extern" and n.value.args[0].value=="printf":
            args=n.value.args
            if len(args)!=24 or not isinstance(args[1],T.StringImm):raise ValueError("unrecognized codegen printf")
            fmt=args[1].value
            matches=[site for site in counts if fmt==f"TLACC1|{site}|0|"+"|".join(["%d"]*10+["%u"]*12)+"\n"]
            if len(matches)!=1:raise ValueError("codegen trace format differs")
            if [str(v.dtype) for v in args[2:]]!=["int32"]*10+["uint32"]*12:raise ValueError("codegen printf ABI differs")
            counts[matches[0]]+=1
            return None
        if isinstance(n,T.SeqStmt):
            children=[x for c in n.seq if (x:=clean(c)) is not None]
            return None if not children else children[0] if len(children)==1 else T.SeqStmt(children)
        if isinstance(n,T.IfThenElse):
            then=clean(n.then_case);other=clean(n.else_case) if n.else_case is not None else None
            if then is None and other is None:return None
            if then is None:raise ValueError("trace erasure changed original branch polarity")
            return T.IfThenElse(n.condition,then,other)
        if isinstance(n,T.For):
            body=clean(n.body)
            return T.For(n.loop_var,n.min,n.extent,n.kind,body,n.thread_binding,n.annotations,n.step) if body is not None else None
        if isinstance(n,T.AttrStmt):
            body=clean(n.body)
            return T.AttrStmt(n.node,n.attr_key,n.value,body) if body is not None else None
        return n
    result=tvm.IRModule(dict(module.functions),attrs=module.attrs)
    for gv,f in module.functions.items():result.update_func(gv,f.with_body(clean(f.body)))
    if counts!=widths:raise ValueError(f"codegen trace site counts differ: {counts}")
    return result


def loads(expr, active=None):
    from tvm import tirx as T
    active = T.IntImm("bool", 1) if active is None else active
    if isinstance(expr, T.BufferLoad):
        yield expr, active
    elif isinstance(expr, T.Call):
        if expr.op.name == "tirx.if_then_else":
            yield from loads(expr.args[1], T.And(active, expr.args[0]))
            yield from loads(expr.args[2], T.And(active, T.Not(expr.args[0])))
        else:
            for arg in expr.args:
                yield from loads(arg, active)
    elif isinstance(expr, T.Cast):
        yield from loads(expr.value, active)
    elif hasattr(expr, "a") and hasattr(expr, "b"):
        yield from loads(expr.a, active)
        yield from loads(expr.b, active)


def instrument(func, points):
    import tvm
    from tvm import tirx as T
    threads, binds, allocs, statements = {}, {}, {}, []
    def inventory(n):
        if isinstance(n, T.AttrStmt) and n.attr_key == "thread_extent":
            threads[n.node.thread_tag] = n.node.var
        if isinstance(n, T.Bind):
            if n.var in binds:
                raise ValueError("duplicate scalar binding")
            binds[n.var] = n.value
        if isinstance(n, T.AllocBuffer):
            allocs[n.buffer.data] = n.buffer
    T.stmt_functor.post_order_visit(func.body, inventory)
    walk(func.body, lambda s, l, g: statements.append((s, l, g)))
    def validate_local_state(p,loops,guards):
        if p["case"] not in ("gemm","gqa"):
            if guards:raise ValueError("unexpected ordinary-access outer guard")
            return {}
        locals_by_name={v.name:v for v,b in allocs.items() if b.scope()=="local.var"}
        known={}
        if p["case"]=="gemm":
            var=locals_by_name.get("gi_prod")
            writes=[(s,l,g) for s,l,g in statements if isinstance(s,T.BufferStore) and var is not None and s.buffer.data.same_as(var)]
            if len(writes)!=2:raise ValueError("producer recurrence write count differs")
            init,update=writes
            if init[1] or init[2] or int(init[0].value)!=0 or int(init[0].indices[0])!=0:
                raise ValueError("producer recurrence initialization differs")
            if len(update[1])!=1 or not update[1][0].same_as(loops[0]) or not isinstance(loops[0].body,T.SeqStmt) or not loops[0].body.seq[-1].same_as(update[0]):
                raise ValueError("producer counter update must end each selected outer iteration")
            expected=T.BufferLoad(update[0].buffer,[T.IntImm("int32",0)])+1
            if not tvm.ir.structural_equal(update[0].value,expected) or int(update[0].indices[0])!=0:
                raise ValueError("producer counter recurrence differs")
            known[var]=p["loop_values"][0]
        else:
            # The reviewed source has one producer eff scalar, assigned once before its loop.
            for var in locals_by_name.values():
                if not var.name.startswith("eff"):continue
                writes=[(s,l,g) for s,l,g in statements if isinstance(s,T.BufferStore) and s.buffer.data.same_as(var)]
                if len(writes)!=1 or writes[0][1] or int(writes[0][0].indices[0])!=0:
                    raise ValueError("GQA eff must have one dominating initialization")
                expr=writes[0][0].value
                seq=next((v for v in func.params if v.name=="seq_len_kv"),None)
                if seq is None or not tvm.ir.structural_equal(expr,(seq+127)//128):
                    raise ValueError("GQA dynamic extent is not ceildiv(actual KV shape,128)")
                known[var]=3
            if loops and (not isinstance(loops[0].extent,T.BufferLoad) or loops[0].extent.buffer.data not in known):
                raise ValueError("unproved GQA loop scalar")
        return known
    params = {p.name: p for p in func.params}
    aliases = {v.name: v for v in binds if str(v.dtype) == "handle"}
    def data_var(name):
        v = params.get(name, aliases.get(name))
        if v is None:
            raise ValueError(f"missing unique parameter/alias {name}")
        return v
    def alias_offset(var):
        e = binds[var]
        if not isinstance(e, T.Call) or e.op.name != "tirx.handle_add_byte_offset" or len(e.args) != 2 or e.args[0] not in allocs:
            raise ValueError("unreviewed shared alias")
        allocation = allocs[e.args[0]]
        if allocation.scope() != "shared.dyn" or str(allocation.dtype) != "uint8":
            raise ValueError("unreviewed shared allocation")
        return int(e.args[1]), int(allocation.shape[0])

    pending, manifests = {}, []
    for p in points:
        found = []
        target = data_var(p["data"])
        for stmt, loops, guards in statements:
            if p["operation"] != "transfer" and isinstance(stmt, T.BufferStore):
                candidates = [(stmt, T.IntImm("bool", 1))] if p["operation"] == "write" else list(loads(stmt.value))
                for a, active in candidates:
                    if a.buffer.data.same_as(target):
                        found.append((stmt, loops, guards, a, active))
            elif p["operation"] == "transfer" and isinstance(stmt, T.Evaluate) and isinstance(stmt.value, T.Call):
                call = stmt.value
                if call.op.name in ("tl.tma_load", "tl.tma_store") and call.args[0].same_as(target):
                    ptr = call.args[2 if call.op.name == "tl.tma_load" else 1]
                    if p["shared_offset"] is None or isinstance(ptr.args[2], T.IntImm) and int(ptr.args[2]) == p["shared_offset"]:
                        found.append((stmt, loops, guards, call, T.IntImm("bool", 1)))
        if len(found) != 1:
            raise ValueError(f"{p['site']}: expected one static access, found {len(found)}")
        stmt, loops, guards, a, active = found[0]
        case = p["case"]
        transfer = p["operation"] == "transfer"
        known_local=validate_local_state(p,loops,guards)
        def guard_value(g,bindings):
            def replace(n):
                if isinstance(n,T.Var) and n in binds:
                    return T.stmt_functor.ir_transform(T.Evaluate(binds[n]),replace,None).value
                if isinstance(n,T.BufferLoad):
                    if n.buffer.data not in known_local or len(n.indices)!=1 or int(n.indices[0])!=0:
                        raise ValueError("unknown data-dependent guard")
                    return T.IntImm(n.dtype,known_local[n.buffer.data])
                if isinstance(n,T.Call) and n.op.name.split('.')[-1]=="tl_shuffle_elect":
                    if len(n.args)!=1 or int(n.args[0])!=(32 if p["case"]=="gqa" and p["line"] in (114,115,120,128) else 128):
                        raise ValueError("unexpected election domain")
                    return T.IntImm("bool",1) # Membership below is checked separately; never re-elected on GPU.
                return None
            resolved=T.stmt_functor.ir_transform(T.Evaluate(g),replace,None).value
            pure(resolved)
            return bool(evaluate(resolved,bindings))
        if transfer:
            elections=[]
            for g in guards:T.stmt_functor.post_order_visit(g,lambda n:elections.append(n) if isinstance(n,T.Call) and n.op.name.split('.')[-1]=="tl_shuffle_elect" else None)
            if len(elections)!=1:raise ValueError("TMA site must retain one original election guard")
        expected_loop_extents = {"gelu": [2], "sum": [2, 4], "sum_unpadded": [2] if p["line"] == 47 else [],
                                 "gemm": [4, 3], "gqa": [3] if p["line"] in (120, 128) else []}[case]
        if len(loops) != len(expected_loop_extents) or len(loops) > 4:
            raise ValueError("lowered loop depth differs from reviewed domain")
        for l, extent in zip(loops, expected_loop_extents):
            if int(l.min) != 0:
                raise ValueError("nonzero loop origin")
            if isinstance(l.extent, T.IntImm):
                if int(l.extent) != extent:
                    raise ValueError("loop extent differs")
            elif not (case == "gqa" and isinstance(l.extent, T.BufferLoad) and l.extent.buffer.scope() == "local.var"):
                raise ValueError("unreviewed dynamic loop extent")
        selection = p["loop_values"]
        selected_loops = list(range(len(selection)))
        domain = [([selection[i]] if i in selected_loops else list(range(n))) for i, n in enumerate(expected_loop_extents)]
        manifest = dict(p, lowered_loops=[dict(var=l.loop_var.name, extent=n, selected=domain[i]) for i, (l,n) in enumerate(zip(loops, expected_loop_extents))],
                        guards=[str(g) for g in guards], active_expression=str(active), runtime_fields="IR access operands before access",
                        launch=0, expected_keys=[])
        if transfer:
            load = a.op.name == "tl.tma_load"
            rank = 2 if case == "gemm" else 4
            if len(a.args) != rank + (4 if load else 4) or int(a.args[-1]) != 0 or (not load and int(a.args[-2]) != 0):
                raise ValueError("unreviewed TMA call policy/arity")
            ptr = a.args[2 if load else 1]
            if not isinstance(ptr, T.Call) or ptr.op.name != "tirx.tvm_access_ptr" or int(ptr.args[4]) != (2 if load else 1):
                raise ValueError("unreviewed TMA shared pointer")
            coords = list(a.args[3:3+rank] if load else a.args[2:2+rank])
            barrier = a.args[1] if load else None
            if load and (not isinstance(barrier, T.BufferLoad) or barrier.buffer.scope() != "shared.barrier" or len(barrier.indices) != 1):
                raise ValueError("unreviewed TMA barrier identity")
            values = [ptr.args[2], *coords, *[T.IntImm("int32", 0)]*(4-rank), barrier.indices[0] if load else T.IntImm("int32", -1)]
            indices = [values]
            manifest.update(kind="tma_load" if load else "tma_store", descriptor=target.name, rank=rank,
                            scope="shared.dyn", dtype="float16", shared_buffer=ptr.args[1].name,
                            shared_extent=int(ptr.args[3]), shared_alias_bytes=alias_offset(ptr.args[1]),
                            barrier=barrier.buffer.data.name if load else None,
                            values_expression=[str(v) for v in values], vector_width=1)
            expected_shared = {"a_desc": ("a_smem",0,98304), "b_desc": ("b_smem",49152,98304),
                               "Q_desc": ("Qs",65536,98304), "K_desc": ("Ks",0,98304),
                               "V_desc": ("Vs",32768,98304), "O_desc": ("Os",81920,98304)}[target.name]
            if (ptr.args[1].name, *alias_offset(ptr.args[1])) != expected_shared or int(ptr.args[3]) != (4096 if target.name in ("Q_desc", "O_desc") else 8192):
                raise ValueError("TMA shared allocation mapping differs")
        else:
            idx = scalar_indices(a)
            if len(idx) != p["width"]:
                raise ValueError("access vector width differs")
            indices = [[v, *[T.IntImm("int32", 0)]*5] for v in idx]
            manifest.update(kind=p["operation"], scope=a.buffer.scope(), dtype=str(a.buffer.dtype),
                            shape=[int(n) for n in a.buffer.shape], vector_width=len(idx),
                            values_expression=[str(v) for v in idx])
            if a.buffer.scope() == "shared.dyn":
                if alias_offset(target) != (0,1536):
                    raise ValueError("shared copy alias differs")
                manifest["shared_alias_bytes"] = alias_offset(target)
        for values in indices:
            for v in values:
                pure(v)
        pure(active)
        manifest["signed"] = [not str(v.dtype).startswith("uint") for v in indices[0]]
        # Exhaustively prove the fixed operand domain; the logger still uses original IR expressions.
        for loop_values in itertools.product(*domain):
            for tx in range(*(p["elected"] or [0, 128])):
                bindings = {v: (tx if key == "threadIdx.x" else p["block"]["xyz".index(key[-1])] if key.startswith("blockIdx") else 0) for key,v in threads.items()}
                bindings.update({l.loop_var: value for l,value in zip(loops,loop_values)})
                bindings.update({v:384 if v.name=="seq_len_kv" else 256 for v in func.params if v.name in ("seq_len_kv","seq_len_q")})
                if not all(guard_value(g,bindings) for g in guards):
                    raise ValueError("selected event domain is outside original guards")
                for lane, vals in enumerate(indices):
                    observed = [evaluate(v, bindings) for v in vals]
                    mask = evaluate(active, bindings)
                    bx, by, _ = p["block"]
                    if case == "gelu":
                        expected = bx*2048 + loop_values[0]*1024 + tx*8 + lane
                    elif case == "sum":
                        expected = bx*514 + loop_values[0]*257 + loop_values[1]*128 + tx
                        if mask != int(loop_values[1]*128+tx < 257):
                            raise ValueError("masked load predicate differs")
                    elif case == "sum_unpadded":
                        expected = (bx*512 if p["buffer"] == "x" else (loop_values[0]*256 if loop_values else 0)) + tx*p["width"] + lane
                    else:
                        expected = observed[0]
                        if case == "gemm":
                            expected_values = [selection[1]*8192, selection[0]*64, 0 if target.name == "a_desc" else bx*128, 0, 0, selection[1]]
                        else:
                            k = selection[0] if selection else 0
                            off = p["shared_offset"] if p["shared_offset"] is not None else (k%2)*8192
                            coord1 = bx*128 + (64 if off == 4096 else 0) if target.name in ("Q_desc","O_desc") else k*128
                            # Barrier numbering is preserved and archived; no barrier contents are read.
                            expected_values = [off,0,coord1,by if target.name in ("Q_desc","O_desc") else 0,0,observed[5]]
                        if observed != expected_values:
                            raise ValueError("TMA operand mapping differs")
                    if observed[0] != expected or any(not -(1<<63) <= v < (1<<64) for v in observed):
                        raise ValueError("operand mapping/overflow proof failed")
                    key = [*loop_values, *[0]*(4-len(loop_values)), lane]
                    if not transfer or tx == p["elected"][0]:
                        manifest["expected_keys"].append(dict(loops=key[:4], lane=lane, thread=None if transfer else tx, active=mask))
        logs = []
        for lane, vals in enumerate(indices):
            prefix = [threads.get(f"blockIdx.{axis}", T.IntImm("int32",0)) for axis in "xyz"]
            prefix += [threads["threadIdx.x"], *[l.loop_var for l in loops], *[T.IntImm("int32",0)]*(4-len(loops)), T.IntImm("int32",lane), T.Cast("int32",active)]
            words = encode_words(vals)
            fmt = f"TLACC1|{p['site']}|0|" + "|".join(["%d"]*10+["%u"]*12) + "\n"
            logs.append(T.Evaluate(T.call_extern("int32", "printf", T.StringImm(fmt), *prefix, *words)))
        condition = T.IntImm("bool",1)
        for axis, value in zip("xyz", p["block"]):
            if f"blockIdx.{axis}" in threads:
                condition = T.And(condition, threads[f"blockIdx.{axis}"] == value)
        for i, value in enumerate(selection):
            condition = T.And(condition, loops[i].loop_var == value)
        pending.setdefault(stmt, []).append(T.IfThenElse(condition,logs[0] if len(logs)==1 else T.SeqStmt(logs),None))
        manifests.append(manifest)
    if sum(len(p["expected_keys"]) for p in manifests) > 65536:
        raise ValueError("access trace exceeds event budget")
    inserted = {log for logs in pending.values() for log in logs}
    def append(n):
        if n in pending:
            replacement = T.SeqStmt([*pending[n],n])
            return replacement
        return None
    body = rewrite(func.body,append)
    def erase(n):
        if isinstance(n,T.SeqStmt):
            children=[rewrite(c,erase) for c in n.seq if c not in inserted]
            return children[0] if len(children)==1 else T.SeqStmt(children)
        return None
    stripped = rewrite(body,erase)
    if not tvm.ir.structural_equal(stripped,func.body):
        raise ValueError("removing trace nodes did not restore original device IR")
    return func.with_body(body), manifests


@contextmanager
def session(folder, points, enabled):
    import tvm
    from tvm import tirx as T
    from tilelang.backend.pass_pipeline.pipeline import get_pipeline, register_pipeline, PassPipeline
    original = get_pipeline("cuda")
    lower = importlib.import_module("tilelang.engine.lower")
    prepare = lower._prepare_device_codegen_mod
    state = dict(pipeline_calls=0, codegen_calls=0)
    def export(mod, name):
        (folder/f"{name}.py").write_text(mod.script(show_meta=True),encoding="utf-8")
        (folder/f"{name}.json").write_text(tvm.ir.save_json(mod),encoding="utf-8")
    def pipeline(mod,target):
        state["pipeline_calls"] += 1
        if state["pipeline_calls"] != 1:
            raise ValueError("exactly one CUDA pipeline call required")
        mod = original.lower(mod,target)
        export(mod,"pretrace")
        state["module"] = mod
        if enabled:
            baseline = tvm.ir.load_json((folder.parent/"baseline/pretrace.json").read_text())
            if not tvm.ir.structural_equal(comparison_view(mod),comparison_view(baseline),map_free_vars=True):
                raise ValueError("baseline and pretrace pipeline modules differ")
        devices = [(gv,f) for gv,f in mod.functions.items() if isinstance(f,T.PrimFunc) and int(f.attrs.get("calling_conv",0)) == 2]
        if len(devices) != 1:
            raise ValueError("expected one device function")
        gv, func = devices[0]
        new, manifests = instrument(func,points)
        save_json(folder/"access-points.json",manifests)
        state["points"] = manifests
        if enabled:
            changed = tvm.IRModule(dict(mod.functions),attrs=mod.attrs)
            changed.update_func(gv,new)
            export(changed,"traced")
            return changed
        return mod
    def codegen(mod):
        state["codegen_calls"] += 1
        result = prepare(mod)
        export(result,"codegen")
        if enabled:
            baseline=tvm.ir.load_json((folder.parent/"baseline/codegen.json").read_text())
            stripped=erase_codegen_trace(result,state["points"])
            if not tvm.ir.structural_equal(stripped,baseline,map_free_vars=True):
                raise ValueError("trace removal from actual codegen IR differs from baseline")
        state["codegen_erasure_verified"]=True
        return result
    register_pipeline(PassPipeline("cuda",pipeline))
    lower._prepare_device_codegen_mod = codegen
    try:
        yield state
    finally:
        register_pipeline(original)
        lower._prepare_device_codegen_mod = prepare
        save_json(folder/"pipeline-state.json",dict(pipeline_calls=state["pipeline_calls"],codegen_calls=state["codegen_calls"],
                  restored=get_pipeline("cuda") is original and lower._prepare_device_codegen_mod is prepare))
