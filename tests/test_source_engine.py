import ast
import importlib
import json
import linecache
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tilelang_debugger.source_instrument import prepare, inject, source_loader
from tilelang_debugger.source_capture import output_indices, no_storage_alias
from tilelang_debugger.instrument import Unsupported
from tilelang_debugger.evidence import reference_status
from tilelang_debugger import cli


SOURCE = '''import tilelang.language as T
def build():
    @T.prim_func
    def main(x: T.Tensor((8,), "float32"), y: T.Tensor((8,), "float32")):
        with T.Kernel(2, threads=128) as bx:
            acc = T.alloc_fragment((8,), "float32")
            for k in T.serial(3, 12, 2):
                T.fill(acc, k)
            T.copy(acc, y)
    return main
'''


def config(line=8):
    return dict(source="/chosen/kernel_impl.py", points=[dict(id="acc", line=line, when="after", buffer="acc", block=[1, 0, 0], loops=[dict(line=7, iteration=2)] if line == 8 else [])])


class SourceEngineTests(unittest.TestCase):
    def test_source_selection_without_hash_or_rename(self):
        points = prepare(SOURCE, config())
        self.assertEqual(points[0]["loop_values"], [5])
        self.assertEqual(points[0]["loop_bounds"], [["3", "12", "2"]])
        emitted = inject(SOURCE, points)
        self.assertIn("__tldbg_source_capture(acc, 'acc', (k,), [(3, 12, 2)])", emitted)
        self.assertNotIn("tldbg_buffer", emitted)
        changed = SOURCE.replace("acc", "renamed") + "\n# arbitrary source edit\n"
        cfg = config(); cfg["points"][0]["buffer"] = "renamed"
        self.assertEqual(prepare(changed, cfg)[0]["buffer"], "renamed")
        ast.parse(emitted)

    def test_divergent_scopes_and_early_exit_rejected(self):
        for source in (SOURCE.replace("for k in T.serial(3, 12, 2):", "for k in T.Parallel(8):"),
                       SOURCE.replace("T.fill(acc, k)", "break"),
                       SOURCE.replace("for k in T.serial(3, 12, 2):", "if bx == 1:")):
            with self.assertRaises(Unsupported):
                prepare(source, config())
        with self.assertRaises(Unsupported):
            prepare(SOURCE.replace("T.fill(acc, k)", "T.gemm(x, x, acc)"), config())

    def test_readonly_global_source_domain(self):
        src = '''def main(x, y):
    with T.Kernel(2, threads=128) as bx:
        for i in T.Parallel(8):
            y[i] = x[i]
'''
        cfg = dict(source="x.py", points=[dict(id="input", line=3, when="before", buffer="x", block=[0,0,0], loops=[])])
        self.assertTrue(prepare(src, cfg)[0]["global_readonly_syntax"])
        for statement in ("y[i] = opaque(x)", "x[i] = y[i]", "a = x", "y[i] = opaque(x[i])"):
            altered = src.replace("y[i] = x[i]", statement)
            self.assertFalse(prepare(altered, cfg)[0]["global_readonly_syntax"])

    def test_real_package_relative_import_and_restoration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); package = root / "tldbg_test_package"; package.mkdir()
            (package / "__init__.py").write_text("")
            (package / "sibling.py").write_text("VALUE = 7\n")
            chosen = package / "custom_name.py"
            chosen.write_text("raise RuntimeError('original must not run')\n")
            original = chosen.read_bytes()
            cache, meta = dict(linecache.cache), list(sys.meta_path)
            sys.path.insert(0, temp)
            try:
                with source_loader(chosen, "from .sibling import VALUE\nRESULT = VALUE + 2\n") as state:
                    module = importlib.import_module("tldbg_test_package.custom_name")
                    self.assertEqual(module.RESULT, 9)
                    self.assertEqual(Path(module.__file__), chosen)
                    self.assertEqual(module.__package__, "tldbg_test_package")
                    self.assertEqual(state["loads"], 1)
                self.assertTrue(state["restored"])
                self.assertEqual(cache, linecache.cache)
                self.assertEqual(meta, sys.meta_path)
                self.assertEqual(original, chosen.read_bytes())
            finally:
                sys.path.remove(temp)
                for name in list(sys.modules):
                    if name.startswith("tldbg_test_package"):
                        del sys.modules[name]

    def test_hook_restoration_on_exception_and_missing_import(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "unused.py"; file.write_text("x=1")
            with self.assertRaisesRegex(RuntimeError, "driver failed"):
                with source_loader(file, "x=2") as state:
                    raise RuntimeError("driver failed")
            self.assertTrue(state["restored"])
            self.assertEqual(state["loads"], 0)

    def test_output_index_normalization(self):
        self.assertEqual(output_indices([-1, 0], 3), [2, 0])
        for indices in ([], [0, -3], [True], [3], None):
            with self.assertRaises(ValueError):
                output_indices(indices, 3)

    def test_global_storage_alias(self):
        class Tensor:
            def __init__(self, address): self.address = address
            def untyped_storage(self): return self
            def data_ptr(self): return self.address
            def nbytes(self): return 32
        no_storage_alias([Tensor(64), Tensor(128)], [0])
        with self.assertRaises(ValueError):
            no_storage_alias([Tensor(64), Tensor(80)], [0])

    def test_reference_status_does_not_weaken_legacy(self):
        missing = {"status": "not_provided", "passed": None}
        self.assertEqual(reference_status([missing, missing], "source-capture-v1"), "not_checked")
        self.assertEqual(reference_status([missing, {"passed": False}], "source-capture-v1"), "failed")
        for schema, value in [(1, missing), ("source-capture-v1", {}), ("source-capture-v1", {"passed": 1})]:
            with self.assertRaises(ValueError): reference_status([value], schema)

    def test_cli_driver_arguments_after_separator(self):
        with patch("sys.argv", ["tldbg", "run", "driver.py", "--monitor", "m.json", "--output", "cap", "--", "--output", "driver-output"]), \
             patch("tilelang_debugger.source_capture.run", return_value={"numerical_status": "not_checked"}) as run:
            cli.main()
        self.assertEqual(run.call_args.kwargs["driver_args"], ["--output", "driver-output"])


if __name__ == "__main__":
    unittest.main()
