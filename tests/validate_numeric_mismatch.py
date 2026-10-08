"""Real H200 negative fixture. No product contract bypass; only reference is shifted."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch

from tilelang_debugger import instrument
from tilelang_debugger.analysis import analyze, save
from tilelang_debugger.cli import main as cli_main
from tilelang_debugger.evidence import read, verify_capture, verify_success

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    save(output / "validation.json", dict(status="running"))
    try:
        original = (ROOT / "examples/gelu/run.py").read_text()
        assert original.count("check_output(out, ref,") == 1
        modified = original.replace("check_output(out, ref,", "check_output(out, ref + 100,")
        (output / "fixture-original.py").write_text(original)
        (output / "fixture-modified.py").write_text(modified)
        with tempfile.TemporaryDirectory() as temporary:
            driver = Path(temporary) / "run.py"
            driver.write_text(modified)
            shutil.copyfile(ROOT / "examples/gelu/kernel.py", driver.with_name("kernel.py"))
            contracts = instrument.contracts()
            contracts["gelu"]["driver_sha256"] = [instrument.digest(driver.read_bytes())]
            # This only authorizes the reviewed test driver's reference expression in this process.
            # Kernel, compile config, gate, monitor and worker behaviour are untouched.
            argv = ["tldbg", "run", str(driver), "--monitor", str(ROOT / "examples/gelu/monitor.json"),
                    "--output", str(output / "capture"), "--sanitizer", "racecheck", "--timeout", "360"]
            stdout = io.StringIO()
            with patch.object(instrument, "contracts", return_value=contracts), patch("sys.argv", argv), contextlib.redirect_stdout(stdout):
                try:
                    cli_main()
                except SystemExit as exc:
                    exit_code = exc.code
                else:
                    exit_code = 0
            (output / "cli.stdout.log").write_text(stdout.getvalue())
        assert exit_code == 2
        capture = output / "capture"
        points = verify_capture(capture, "racecheck")
        assert len(points) == 2
        assert read(capture / "run.json")["numerical_status"] == "failed"
        assert all(read(capture / m / "reference.json")["passed"] is False for m in ("baseline", "instrumented"))
        try:
            verify_success(capture, "racecheck")
        except ValueError as exc:
            assert "reference failed" in str(exc)
        else:
            raise AssertionError("strict acceptance allowed a wrong reference")
        # A correct independent reference can diagnose the bad reference, without recapturing.
        report = analyze(capture, ROOT / "examples/gelu/reference.py", output / "analysis")
        assert report["matched"] is True and report["capture_numerical_status"] == "failed"
        save(output / "validation.json", dict(status="passed", run_exit_code=exit_code,
                  capture_status="passed", numerical_status="failed", records=read(capture / "run.json")["records"],
                  strict_acceptance_rejected=True, independent_reference_matched=True))
    except Exception as exc:
        save(output / "validation.json", dict(status="failed", error=str(exc)))
        raise


if __name__ == "__main__":
    main()
