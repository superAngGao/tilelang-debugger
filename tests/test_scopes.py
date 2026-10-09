import ast
import copy
import unittest

from tilelang_debugger.scopes import prepare, inject
from tilelang_debugger.instrument import Unsupported
from tilelang_debugger.sample_records import parse
from tilelang_debugger.sample_analysis import compare_samples


SOURCE = '''import tilelang.language as T
def build():
    @T.prim_func
    def main(x: T.Tensor((8,), "int32")):
        with T.Kernel(1, threads=2):
            tx = T.get_thread_binding()
            for i in T.serial(2, 4):
                if tx == 0:
                    for i in T.serial(3, 5):
                        v = i + 100
'''


def config(source=SOURCE):
    return dict(schema=2, source="chosen.py", points=[dict(id="p", line=next(i for i, l in enumerate(source.splitlines(), 1) if "v =" in l),
                                                       when="after", buffer="v", block=[0, 0, 0], loops=[])])


def contract(parallel=False):
    source = SOURCE.replace("T.serial(2, 4)", "T.Parallel(2)") if parallel else SOURCE
    p = prepare(source, config(source))[0]
    for s in p["scopes"]:
        if s["kind"] != "branch":
            s["actual_bounds"] = [[int(v) for v in b] for b in s["bounds"]]
    p.update(bound=True, dtype="int64", shape=[], elements=1, threads=2, event_budget=100)
    return p


def line(kind, scope, tx, coords=(), index=0, bits=0):
    return "|".join(map(str, ("TLDBG2", "p", kind, scope, 0, 0, 0, 0, tx, *coords, index, bits & 0xffffffff, bits >> 32, "E")))


def log():
    rows = [line("R", -1, tx) for tx in range(2)]
    for tx in range(2):
        for i in range(2, 4):
            rows.append(line("B", 1, tx, (i,), int(tx == 0)))
            if tx == 0:
                for j in range(3, 5):
                    rows.append(line("D", -1, tx, (i, j), bits=9007199254740993 + j))
    return rows


class ScopeTests(unittest.TestCase):
    def test_shadowed_bindings_and_unique_aliases(self):
        points = prepare(SOURCE, config())
        aliases = [s["aliases"][0] for s in points[0]["scopes"] if s["kind"] != "branch"]
        self.assertEqual(len(set(aliases)), 2)
        result = inject(SOURCE, points)
        self.assertIn(f"({aliases[0]}, {aliases[1]})", result)
        ast.parse(result)
        collision = SOURCE.replace("tx =", "__tldbg2_conflict = 1\n            tx =", 1)
        self.assertFalse("__tldbg2_conflict".startswith(prepare(collision, config(collision))[0]["prefix"]))

    def test_condition_evaluated_once_and_implicit_else(self):
        source = SOURCE.replace("if tx == 0:", "if predicate(x[0]):")
        result = inject(source, prepare(source, config(source)))
        self.assertEqual(result.count("predicate(x[0])"), 1)
        self.assertIn("else:", result)

    def test_explicit_if_normalization(self):
        source = SOURCE.replace("if tx == 0:", "with T.If(tx == 0):\n                    with T.Then():").replace("                    for i", "                        for i").replace("                        v =", "                            v =")
        result = inject(source, prepare(source, config(source)))
        self.assertIn("if tx == 0:", result)
        self.assertNotIn("T.If", result)

    def test_unrelated_matrix_call_does_not_block_scalar(self):
        source = SOURCE.replace("            tx =", "            T.gemm(a, b, c)\n            tx =")
        self.assertEqual(len(prepare(source, config(source))), 1)

    def test_sibling_early_exit_is_rejected(self):
        source = SOURCE.replace("                if tx == 0:", "                if tx == 1:\n                    continue\n                if tx == 0:")
        with self.assertRaisesRegex(Unsupported, "early exits"):
            prepare(source, config(source))

    def test_side_effect_and_mutable_loop_bounds_rejected(self):
        for bound in ("extent()", "x[0]", "tx"):
            source = SOURCE.replace("T.serial(2, 4)", f"T.serial({bound})")
            with self.assertRaisesRegex(Unsupported, "pure static"):
                prepare(source, config(source))

    def test_original_loop_selection_and_subscript(self):
        cfg = config(); cfg["points"][0]["loops"] = [dict(line=7, iteration=2)]
        self.assertEqual(prepare(SOURCE, cfg)[0]["scopes"][0]["selection"], 2)
        source = SOURCE.replace("v = i + 100", "x[i + 1] = i + 100")
        cfg["points"][0]["buffer"] = "x[i + 1]"
        self.assertEqual(prepare(source, cfg)[0]["buffer"], "x[i + 1]")
        with self.assertRaises(Unsupported):
            prepare(SOURCE, cfg)
        cfg["points"][0]["buffer"] = "x[next_index()]"
        with self.assertRaises(Unsupported):
            prepare(SOURCE, cfg)

    def test_unordered_round_trip_exact_bits(self):
        records, coverage = parse("\n".join(reversed(log())), [contract()])
        self.assertEqual(len(records), 4)
        self.assertEqual(records[0]["bits"], 9007199254740996)
        self.assertEqual(coverage["p"]["execution_coverage"], "complete")

    def test_missing_root_branch_data_and_entire_instance(self):
        rows = log()
        mutations = [[r for r in rows if r != rows[0]], [r for r in rows if r != rows[2]],
                     [r for r in rows if r != rows[3]], [r for r in rows if "|D|" not in r or "|2|3|" not in r]]
        for mutated in mutations:
            with self.assertRaises(ValueError):
                parse("\n".join(mutated), [contract()])

    def test_duplicate_truncated_and_out_of_domain(self):
        rows = log()
        for extra in (rows[0], rows[3][:-5], line("D", -1, 0, (99, 3), bits=1)):
            with self.assertRaises(ValueError):
                parse("\n".join(rows + [extra]), [contract()])

    def test_parallel_replicas_do_not_prove_execution_coverage(self):
        p = contract(True)
        p["scopes"] = [p["scopes"][0]]
        rows = [line("R", -1, tx) for tx in range(2)] + [line("D", -1, tx, (i,), bits=i) for tx in range(2) for i in range(2)]
        _, coverage = parse("\n".join(rows), [p])
        self.assertEqual(coverage["p"]["logical_coverage"], "complete")
        self.assertEqual(coverage["p"]["execution_coverage"], "unverified")
        missing_replica = [r for r in rows if "|D|" not in r or r.split("|")[8] == "0"]
        self.assertEqual(parse("\n".join(missing_replica), [p])[1]["p"]["execution_coverage"], "unverified")
        missing_coordinate = [r for r in rows if "|D|" not in r or r.split("|")[9] == "0"]
        with self.assertRaises(ValueError):
            parse("\n".join(missing_coordinate), [p])

    def test_parallel_branch_without_data_is_not_not_executed(self):
        p = contract(True)
        rows = [line("R", -1, tx) for tx in range(2)]
        c = parse("\n".join(rows), [p])[1]["p"]
        self.assertEqual(c["point_execution"], "unverified")
        self.assertEqual(c["logical_coverage"], "unverified")

    def test_reference_int64_one_bit_difference_is_visible(self):
        records, _ = parse("\n".join(log()), [contract()])
        samples = [dict(thread=0, coordinates=[i, j], index=0, value=9007199254740993 + j) for i in range(2, 4) for j in range(3, 5)]
        spec = dict(schema=2, key="execution", dtype="int64", samples=samples, atol=0, rtol=0)
        self.assertTrue(compare_samples(records, contract(), spec)[0]["matched"])
        samples[0]["value"] += 1
        report, rows = compare_samples(records, contract(), spec)
        self.assertFalse(report["matched"])
        self.assertEqual(rows[0]["abs_error"], 1)

    def test_reference_domain_cannot_shrink_to_received_records(self):
        records, _ = parse("\n".join(log()), [contract()])
        spec = dict(schema=2, key="execution", dtype="int64", samples=[], atol=0, rtol=0)
        self.assertFalse(compare_samples(records, contract(), spec)[0]["matched"])


if __name__ == "__main__":
    unittest.main()
