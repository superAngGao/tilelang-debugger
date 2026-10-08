import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tilelang_debugger.evidence import verify_capture, verify_success
from tilelang_debugger.records import parse


def write(path, data):
    path.write_text(json.dumps(data))


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        points = [dict(id="p", shape=[1], dtype="float32", block=[0,0,0], loop_values=[])]
        log = "TLDBG1|p|0|0|0|0|0|0|0|0\n"
        write(self.root / "run.json", dict(status="passed", sanitizer="racecheck", records=1, inputs_equal=True, outputs_bitwise_equal=True))
        write(self.root / "points.json", points)
        (self.root / "records.jsonl").write_text(json.dumps(parse(log, points)[0])+"\n")
        for mode in ("baseline", "instrumented"):
            f = self.root / mode
            f.mkdir()
            write(f / "execution.json", dict(compiles=1, launches=1, restored=True))
            write(f / "launch-gate.json", dict(passed=True))
            write(f / "process.json", dict(returncode=0, timeout=False))
            write(f / "reference.json", dict(passed=True))
            write(f / "command.json", ["compute-sanitizer", "--tool", "racecheck"])
            (f / "racecheck.log").write_text("RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)")
            for prefix in ("inputs", "outputs"):
                (f / f"{prefix}-0.bin").write_bytes(bytes(4))
                write(f / f"{prefix}.json", [dict(file=f"{prefix}-0.bin", bytes=4, shape=[1], stride=[1], dtype="float32", sha256=hashlib.sha256(bytes(4)).hexdigest())])
        write(self.root / "instrumented/points.json", points)
        (self.root / "instrumented/stdout.log").write_text(log)

    def test_success_and_wrong_requested_mode(self):
        self.assertEqual(len(verify_success(self.root, "racecheck")), 1)
        with self.assertRaises(ValueError):
            verify_success(self.root, None)

    def test_failed_status_reference_gate_process(self):
        for path, replacement in (("run.json", {"status": "failed"}),
                                  ("instrumented/reference.json", {"passed": False}),
                                  ("instrumented/launch-gate.json", {"passed": False}),
                                  ("baseline/process.json", {"returncode": 86, "timeout": False})):
            f = self.root / path
            old = f.read_text()
            write(f, replacement)
            with self.subTest(path=path), self.assertRaises(ValueError):
                verify_success(self.root, "racecheck")
            f.write_text(old)

    def test_tampered_bytes_and_truncated_stdout(self):
        f = self.root / "instrumented/outputs-0.bin"
        f.write_bytes(b"bad!")
        with self.assertRaises(ValueError):
            verify_success(self.root, "racecheck")
        f.write_bytes(bytes(4))
        (self.root / "instrumented/stdout.log").write_text("")
        with self.assertRaises(ValueError):
            verify_success(self.root, "racecheck")

    def test_numeric_mismatch_is_complete_capture_not_acceptance(self):
        write(self.root / "baseline/reference.json", dict(passed=False))
        self.assertEqual(len(verify_capture(self.root, "racecheck")), 1)
        with self.assertRaises(ValueError):
            verify_success(self.root, "racecheck")

    def test_invalid_reference_boolean_and_summary(self):
        for passed in (None, 0, 1, "false"):
            write(self.root / "baseline/reference.json", dict(passed=passed))
            with self.subTest(passed=passed), self.assertRaises(ValueError):
                verify_capture(self.root, "racecheck")
        write(self.root / "baseline/reference.json", dict(passed=False))
        run = json.loads((self.root / "run.json").read_text())
        run["numerical_status"] = "passed"
        write(self.root / "run.json", run)
        with self.assertRaises(ValueError):
            verify_capture(self.root, "racecheck")


if __name__ == "__main__":
    unittest.main()
