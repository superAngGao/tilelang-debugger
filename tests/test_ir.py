"""Real TVM TIR gate tests; require TileLang/TVM but do not launch a GPU kernel."""
import unittest

import tilelang  # Registers the backend's intrinsic Ops.
from tvm import tirx as T

from tilelang_debugger.ir import check_instrumented, layout_digest, protocol, validate_call


def fixture(size=128, bad_source=False, arithmetic=False, bad_read=False, bad_record=False):
    tx, idx = T.Var("tx", "int32"), T.Var("idx", "int32")
    local = T.decl_buffer((1,), "float32", name="fragment", scope="local")
    shared = T.decl_buffer((size,), "float32", name="scratch", scope="shared")
    src = local[999 if bad_source else 0]
    expr = T.BufferStore(shared, src + 1 if arithmetic else src, [tx if size == 128 else tx // 64])
    store = dict(kind="store", expr=expr, guards=() if size == 128 else (tx % 64 == 0,), loops=())
    barrier = dict(kind="named_sync", id=15, count=128, args=[T.StringImm("shared"), T.IntImm("int32",15), T.IntImm("int32",128)], guards=(), loops=())
    loop = T.For(idx, 0, size, T.ForKind.SERIAL, T.Evaluate(0))
    bit_expr = T.reinterpret("uint32", shared[999 if bad_read else idx])
    args = [T.StringImm("TLDBG1|p|0|%d|%d|%d|%d|%d|%d|%u\n"), *[T.IntImm("int32",0) for _ in range(5)], idx+1 if bad_record else idx, bit_expr]
    printed = dict(kind="printf", args=args, expr=T.call_extern("int32", "printf", *args), guards=(tx == 0,), loops=(loop,))
    events = [store, dict(barrier), printed, dict(barrier)]
    p = dict(id="p", shape=[size], dtype="float32", block=[0,0,0], loop_vars=[], loop_values=[], leader=0, threads=128, barrier=15)
    p["layout_sha256"] = [layout_digest([(i, i if size == 128 else i*64, 0) for i in range(size)])]
    baseline = dict(config={"pass_configs": {}}, barriers=[], protocol=[])
    data = dict(config=baseline["config"], barriers=[])
    kwargs = dict(data=data, events=events, variables={"threadIdx.x": tx},
                  allocations={local.data: ("float32", [1]), shared.data: ("float32", [size])},
                  source="", baseline=baseline, points=[p], cta_threads=128, bindings={"p": local})
    return kwargs


class GateTests(unittest.TestCase):
    def test_complete_and_representative_writers(self):
        for size in (128, 2):
            self.assertTrue(check_instrumented(**fixture(size))["passed"])

    def test_bad_source_read_arithmetic_and_identity(self):
        for key in ("bad_source", "arithmetic", "bad_read", "bad_record"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_instrumented(**fixture(**{key: True}))

    def test_barrier_collision_and_partial_participation(self):
        f = fixture()
        f["events"].append(dict(f["events"][1]))
        with self.assertRaises(ValueError):
            check_instrumented(**f)
        f = fixture()
        f["events"][1]["guards"] = (f["variables"]["threadIdx.x"] < 64,)
        with self.assertRaises(ValueError):
            check_instrumented(**f)

    def test_unknown_extern_and_missing_protocol(self):
        with self.assertRaises(ValueError):
            validate_call(T.call_extern("int32", "unknown_barrier"))
        f = fixture()
        f["baseline"]["protocol"] = [dict(kind="tl.wait_wgmma", args=["0"], guards=[], loops=[])]
        with self.assertRaises(ValueError):
            check_instrumented(**f)

    def test_protocol_records_wait_and_loop(self):
        i = T.Var("k", "int32")
        a = dict(kind="tl.wait_wgmma", args=[T.IntImm("int32",0)], guards=(), loops=(T.For(i,0,3,T.ForKind.SERIAL,T.Evaluate(0)),))
        b = dict(a, args=[T.IntImm("int32",1)])
        c = dict(a, loops=(T.For(i,0,2,T.ForKind.SERIAL,T.Evaluate(0)),))
        self.assertNotEqual(protocol([a]), protocol([b]))
        self.assertNotEqual(protocol([a]), protocol([c]))

    def test_complete_but_permuted_mapping_rejected(self):
        f = fixture()
        old = f["events"][0]["expr"]
        f["events"][0]["expr"] = T.BufferStore(old.buffer, old.value, [127-f["variables"]["threadIdx.x"]])
        with self.assertRaises(ValueError):
            check_instrumented(**f)

    def test_thread_dependent_collective_loop_rejected(self):
        f = fixture()
        i = T.Var("j", "int32")
        f["events"][1]["loops"] = (T.For(i,0,f["variables"]["threadIdx.x"]+1,T.ForKind.SERIAL,T.Evaluate(0)),)
        with self.assertRaises((ValueError, TypeError)):
            check_instrumented(**f)


if __name__ == "__main__":
    unittest.main()
