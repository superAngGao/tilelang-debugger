"""CPU checks for source identity and truthful acceptance, no TileLang needed."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "tileops"))
from common import cases, load_upstream, locate

spec = importlib.util.spec_from_file_location("tileops_validation", ROOT / "tests" / "validate_tileops.py")
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


class IntegrationTests(unittest.TestCase):
    def test_branch_and_line_identity_survives_blank_lines(self):
        source = "def factory():\n    if op == 'softmax':\n        T.reduce_sum(x, y)\n    else:\n        T.reduce_sum(x, y)\n"
        point = dict(id="sum", factory="factory", branch="op == 'softmax'", statement="T.reduce_sum(x, y)")
        self.assertEqual(locate(source, point)["line"], 3)
        self.assertEqual(locate("\n\n" + source, point)["line"], 5)
        with self.assertRaisesRegex(ValueError, "uniquely"):
            locate(source, {k:v for k,v in point.items() if k != "branch"})

    def test_renamed_factory_and_buffer_work_without_hashes(self):
        point = dict(id="p", factory="renamed", statement="T.copy(new_name, output)")
        self.assertEqual(locate("def renamed():\n    T.copy(new_name, output)\n", point)["line"], 2)
        with self.assertRaises(ValueError):
            locate("def renamed():\n    T.copy(wrong, output)\n", point)

    def test_loop_metadata_preserves_source_not_lowered_order(self):
        source = "def f():\n    for k in T.Serial(3):\n        for i, j in T.Parallel(4, 8):\n            x[i, j] = y[k, i, j]\n"
        p = locate(source, dict(id="x", factory="f", statement="x[i,j] = y[k,i,j]"))
        self.assertEqual([x["line"] for x in p["enclosing_loops"]], [2, 3])

    def test_wrong_installed_package_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "src/tileops").mkdir(parents=True)
            (root / "src/tileops/__init__.py").touch()
            original = sys.modules.get("tileops")
            fake = types.ModuleType("tileops")
            fake.__file__ = str(root / "wrong/tileops/__init__.py")
            sys.modules["tileops"] = fake
            try:
                with self.assertRaisesRegex(ValueError, "already imported"):
                    load_upstream(root, "tileops")
            finally:
                if original is None:
                    del sys.modules["tileops"]
                else:
                    sys.modules["tileops"] = original

    def test_failure_classification_does_not_hide_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            driver = Path(temp) / "run.py"
            missing = driver.resolve().with_name("kernel.py")
            message = ('Traceback (most recent call last):\n  File "capture.py", line 1\n'
                       '    source = source_file.read_text(encoding="utf-8")\n'
                       f"FileNotFoundError: [Errno 2] No such file or directory: {str(missing)!r}\n")
            failed = dict(returncode=1, timed_out=False)
            self.assertEqual(validation.debugger_status(failed, message, driver, "run"), "unsupported")
            trace_message = message.replace('capture.py', 'access.py').replace('source_file.read_text', 'driver.with_name("kernel.py").read_text')
            self.assertEqual(validation.debugger_status(failed, trace_message, driver, "trace"), "unsupported")
            self.assertEqual(validation.debugger_status(failed, message, driver, "trace"), "failed")
            for changed in (message.replace("kernel.py'", "config.json'"), "ModuleNotFoundError: torch", "invalid config"):
                self.assertEqual(validation.debugger_status(failed, changed, driver, "run"), "failed")
            for process in (dict(returncode=-11, timed_out=False), dict(returncode=1, timed_out=True)):
                self.assertEqual(validation.debugger_status(process, message, driver, "run"), "failed")
            self.assertEqual(validation.debugger_status(dict(returncode=0, timed_out=False), "", driver, "run"), "unverified")

    def test_strict_acceptance_fails_on_unsupported(self):
        base, probes = [dict(passed=True)], [dict(status="unsupported")]
        self.assertFalse(validation.suite_status(base, probes, False)["delivery_passed"])
        self.assertEqual(validation.suite_status(base, probes, False)["exitcode"], 0)
        self.assertEqual(validation.suite_status(base, probes, True)["exitcode"], 1)
        self.assertEqual(validation.suite_status(base, [dict(status="failed")], False)["exitcode"], 1)
        self.assertEqual(validation.suite_status([dict(passed=False)], probes, False)["exitcode"], 1)
        self.assertEqual(validation.suite_status(base, [], False)["exitcode"], 1)

    def test_incomplete_evidence_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "result").mkdir()
            (folder / "result/files.json").write_text("{}")
            (folder / "result/result.json").write_text(json.dumps(dict(status="passed", padding_zero=True, instrumented=False)))
            self.assertFalse(validation.baseline_status(folder, dict(returncode=0, timed_out=False), None))

    def test_matrix_has_all_three_dtypes_and_paths(self):
        matrix = cases()
        self.assertEqual(len(matrix), 21)
        self.assertEqual(len({c["id"] for c in matrix}), 21)
        self.assertEqual({c["dtype"] for c in matrix}, {"float16", "bfloat16", "float32"})


if __name__ == "__main__":
    unittest.main()
