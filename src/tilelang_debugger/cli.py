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
    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("folder")
    args = parser.parse_args()
    from . import capture
    if args.command == "_worker":
        capture.worker(args.folder)
    else:
        print(json.dumps(capture.run(args.driver, args.monitor, args.output, args.timeout, args.sanitizer), indent=2))
