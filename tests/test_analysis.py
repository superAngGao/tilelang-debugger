"""CPU-only tests with real PyTorch; synthetic fixtures are not GPU evidence."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import test_evidence
from test_evidence import write
from tilelang_debugger.analysis import analyze, manifest
from tilelang_debugger.cli import main
from tilelang_debugger.numerics import check_output, tensor_data

PROVIDER = '''import torch
def reference(inputs, points):
    return {"points": {p["id"]: {"tensor": inputs[0].clone(), "atol": 0, "rtol": 0} for p in points},
            "outputs": [{"tensor": inputs[0].clone(), "atol": 0, "rtol": 0}]}
'''


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        fixture = test_evidence.EvidenceTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.capture = fixture.root
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.provider = self.root / "reference.py"
        self.provider.write_text(PROVIDER)
        run = json.loads((self.capture / "run.json").read_text())
        run["run_id"] = "synthetic-unit-test-not-gpu-evidence"
        write(self.capture / "run.json", run)
        points = json.loads((self.capture / "points.json").read_text())
        points[0].update(line=42, when="after", buffer="x", loops=[], source_sha256="synthetic")
        for p in (self.capture / "points.json", self.capture / "instrumented/points.json"):
            write(p, points)

    def run_analysis(self, name="report"):
        return analyze(self.capture, self.provider, self.root / name)

    def test_report_and_capture_immutability_no_tilelang(self):
        before = manifest(self.capture)
        result = self.run_analysis()
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["matched"])
        self.assertEqual(before, manifest(self.capture))
        self.assertEqual(result["comparisons"][0]["source"]["line"], 42)
        self.assertTrue((self.root / "report/report.md").exists())
        import sys
        self.assertNotIn("tilelang", sys.modules)
        self.assertFalse(torch.cuda.is_initialized())

    def test_mismatch_cli_exit_two_and_precise_coordinates(self):
        self.provider.write_text(PROVIDER.replace('inputs[0].clone()', 'inputs[0].clone() + 1'))
        args = ["tldbg", "analyze", str(self.capture), "--reference", str(self.provider), "--output", str(self.root / "report")]
        with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exc:
            main()
        self.assertEqual(exc.exception.code, 2)
        result = json.loads((self.root / "report/analysis.json").read_text())
        self.assertEqual(result["status"], "completed")
        row = result["comparisons"][0]["first_mismatches"][0]
        self.assertEqual((row["coordinates"], row["actual"], row["expected"], row["abs_error"]), ([0], 0, 1, 1))

    def test_provider_mutation_isolated(self):
        self.provider.write_text(PROVIDER.replace('    return', '    inputs[0].add_(1)\n    points[0]["line"] = -9\n    return'))
        result = self.run_analysis()
        self.assertFalse(result["matched"])
        self.assertEqual(result["comparisons"][0]["source"]["line"], 42)
        self.assertEqual(result["comparisons"][0]["first_mismatches"][0]["actual"], 0)

    def test_provider_errors_are_failed_not_mismatch(self):
        sources = [
            'raise RuntimeError("provider crash")',
            'reference = 3',
            PROVIDER.replace('for p in points', 'for p in []'),
            PROVIDER.replace('p["id"]:', '"extra":'),
            PROVIDER.replace('inputs[0].clone()', 'torch.zeros(2)'),
            PROVIDER.replace('"atol": 0', '"atol": True'),
            PROVIDER.replace('inputs[0].clone()', 'inputs[0].to(torch.int32)'),
            PROVIDER.replace('inputs[0].clone()', 'inputs[0].to(torch.complex64)'),
            PROVIDER.replace('inputs[0].clone()', 'inputs[0].to_sparse()'),
        ]
        for i, source in enumerate(sources):
            self.provider.write_text(source)
            with self.subTest(i=i), self.assertRaises(Exception):
                self.run_analysis(str(i))
            result = json.loads((self.root / str(i) / "analysis.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertNotIn("matched", result)

    def test_numeric_failure_allowed_but_bad_evidence_rejected(self):
        write(self.capture / "baseline/reference.json", dict(passed=False))
        result = self.run_analysis("numeric")
        self.assertTrue(result["matched"])
        self.assertEqual(result["capture_numerical_status"], "failed")
        for i, (path, bad) in enumerate((("baseline/launch-gate.json", dict(passed=False)),
                                       ("instrumented/process.json", dict(returncode=86, timeout=False)))):
            p = self.capture / path
            old = p.read_bytes()
            write(p, bad)
            with self.assertRaises(ValueError):
                self.run_analysis(str(i))
            p.write_bytes(old)
        (self.capture / "instrumented/stdout.log").write_text("")
        with self.assertRaises(ValueError):
            self.run_analysis("truncated")

    def test_helper_continues_only_numeric_mismatch(self):
        with patch.dict(os.environ, {"TLDBG_OUTPUT": str(self.root)}):
            self.assertFalse(check_output(torch.zeros(1), torch.ones(1), atol=0, rtol=0)["passed"])
            self.assertFalse(json.loads((self.root / "reference.json").read_text())["passed"])
            with self.assertRaises(ValueError):
                check_output(torch.zeros(1), torch.ones(2), atol=0, rtol=0)
        with patch.dict(os.environ):
            os.environ.pop("TLDBG_OUTPUT", None)
            with self.assertRaises(AssertionError):
                check_output(torch.zeros(1), torch.ones(1), atol=0, rtol=0)

    def test_raw_signed_zero_nan_payload_scalar_and_ref_float64(self):
        data = tensor_data(torch.tensor([0, -2147483648, 2143289345], dtype=torch.int32).view(torch.float32))
        self.assertEqual(data["bits"], [0, 2147483648, 2143289345])
        self.assertEqual(tensor_data(torch.tensor(3.0))["shape"], [])
        self.provider.write_text(PROVIDER.replace('inputs[0].clone()', 'inputs[0].double()'))
        self.assertTrue(self.run_analysis()["matched"])

    def test_dataclass_annotations_module_registration_and_cleanup(self):
        import sys
        before = {name for name in sys.modules if name.startswith("_tldbg_user_reference_")}
        prefix = 'from __future__ import annotations\nfrom dataclasses import dataclass\n@dataclass\nclass Config:\n    atol: float = 0.0\n'
        self.provider.write_text(prefix + PROVIDER.replace('"atol": 0', '"atol": Config().atol'))
        self.assertTrue(self.run_analysis("dataclass")["matched"])
        self.assertEqual(before, {name for name in sys.modules if name.startswith("_tldbg_user_reference_")})
        self.provider.write_text(prefix + 'raise RuntimeError("failed import")')
        with self.assertRaises(RuntimeError):
            self.run_analysis("import-failure")
        self.assertEqual(before, {name for name in sys.modules if name.startswith("_tldbg_user_reference_")})


if __name__ == "__main__":
    unittest.main()
