import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="TileLang H200 source-selected capture")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("driver")
    run.add_argument("--monitor", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--timeout", type=int, default=240)
    run.add_argument("--sanitizer", choices=("racecheck", "synccheck"))
    analyze = commands.add_parser("analyze", help="Compare saved capture with a CPU reference")
    analyze.add_argument("capture")
    analyze.add_argument("--reference", required=True)
    analyze.add_argument("--output", required=True)
    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("folder")
    args = parser.parse_args()
    if args.command == "analyze":
        from .analysis import analyze as analyze_capture
        result = analyze_capture(args.capture, args.reference, args.output)
        print(json.dumps(dict(status=result["status"], matched=result["matched"], output=args.output)))
        if not result["matched"]:
            raise SystemExit(2)
        return
    from . import capture
    if args.command == "_worker":
        capture.worker(args.folder)
    else:
        result = capture.run(args.driver, args.monitor, args.output, args.timeout, args.sanitizer)
        print(json.dumps(result, indent=2))
        if result["numerical_status"] == "failed":
            raise SystemExit(2)
