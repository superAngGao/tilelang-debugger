import ast
import copy
import json
import math
from pathlib import Path
import struct
import unittest

from tilelang_debugger.instrument import Unsupported, inject, locate, prepare
from tilelang_debugger.records import parse, value

ROOT = Path(__file__).resolve().parents[1]


class SourceTests(unittest.TestCase):
    def fixture(self, name):
        folder = ROOT / "examples" / name
        return (folder / "kernel.py").read_text(encoding="utf-8"), json.loads((folder / "monitor.json").read_text()), (folder / "run.py").read_bytes()

    def test_four_contracts_and_source_injection(self):
        for name in ("gelu", "sum", "gemm", "gqa"):
            with self.subTest(name=name):
                source, cfg, driver = self.fixture(name)
                _, points = prepare(source, cfg, driver)
                result = ast.parse(inject(source, points))
                calls = [n for n in ast.walk(result) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id.startswith("__tldbg_capture")]
                self.assertEqual(len(calls), len(points))
                self.assertEqual({n.args[1].value for n in calls}, {p["id"] for p in points})

    def test_modified_source_driver_unsafe_point_rejected(self):
        source, cfg, driver = self.fixture("gqa")
        for changed_source, changed_cfg, changed_driver in ((source + "\n#changed", cfg, driver), (source, cfg, driver+b"\n")):
            with self.assertRaises(Unsupported):
                prepare(changed_source, changed_cfg, changed_driver)
        cfg["points"][0]["line"] += 1
        with self.assertRaises(Unsupported):
            prepare(source, cfg, driver)

    def test_nested_ordinals_and_ambiguous_line(self):
        source = "for k in range(3, 12, 2):\n    for j in T.serial(2):\n        T.copy(A, B)\n"
        p = dict(line=3, loops=[dict(line=1, iteration=2), dict(line=2, iteration=1)])
        _, _, loops, values = locate(source, p)
        self.assertEqual(values, [5, 0])
        p["loops"] = p["loops"][:1]
        with self.assertRaises(Unsupported):
            locate(source, p)
        with self.assertRaises(Unsupported):
            locate("a=1; b=2", dict(line=1, loops=[]))

    def test_parallel_and_pipeline_interior_rejected(self):
        for op in ("Parallel", "Pipelined", "unroll"):
            with self.assertRaises(Unsupported):
                locate(f"for i in T.{op}(4):\n    T.copy(A, B)\n", dict(line=2, loops=[]))

    def test_duplicate_ids_and_bad_iterations(self):
        source, cfg, driver = self.fixture("gemm")
        for iteration in (0, 5, True):
            c = copy.deepcopy(cfg)
            c["points"][0]["loops"][0]["iteration"] = iteration
            with self.assertRaises(Unsupported):
                prepare(source, c, driver)
        cfg["points"] *= 2
        with self.assertRaises(Unsupported):
            prepare(source, cfg, driver)


class RecordTests(unittest.TestCase):
    points = [dict(id="p", shape=[2], dtype="float32", block=[1, 0, 0], loop_values=[3])]
    lines = ["TLDBG1|p|0|1|0|0|3|0|0|2147483648", "TLDBG1|p|0|1|0|0|3|0|1|2143289345"]

    def test_unordered_bits_and_signed_zero(self):
        records = parse("ordinary log\n" + "\n".join(reversed(self.lines)), self.points)
        self.assertEqual([r["bits"] for r in records], [0x80000000, 0x7fc00001])
        self.assertEqual(math.copysign(1, value(records[0]["bits"], "float32")), -1)
        self.assertTrue(math.isnan(value(records[1]["bits"], "float32")))
        self.assertEqual(value(0xffffffff, "int32"), -1)
        self.assertEqual(value(0x3f80, "bfloat16"), 1)
        self.assertEqual(value(0x3c00, "float16"), 1)

    def test_missing_duplicate_identity_width_truncation(self):
        good = "\n".join(self.lines)
        for bad in (self.lines[0], good+"\n"+self.lines[0], good.replace("|p|", "|q|"),
                    good.replace("|1|0|0|3|", "|2|0|0|3|"), good.replace("|3|0|", "|4|0|"),
                    good.replace("2147483648", "4294967296"), good[:-5]+"x", good+"\nTLDBG1|p"):
            with self.subTest(log=bad), self.assertRaises(ValueError):
                parse(bad, self.points)


if __name__ == "__main__":
    unittest.main()
