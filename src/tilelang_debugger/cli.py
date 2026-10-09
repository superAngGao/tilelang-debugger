import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="TileLang H200 source-selected capture")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("driver")
    run.add_argument("--engine", choices=("source", "reviewed"), default="source", help="source: Python source instrumentation; reviewed: legacy fixed contracts")
    run.add_argument("--source", help="Kernel source file (relative to cwd); defaults to config source relative to config directory")
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
    source_worker = commands.add_parser("_source_worker", help=argparse.SUPPRESS)
    source_worker.add_argument("folder")
    unified_worker = commands.add_parser('_unified_worker', help=argparse.SUPPRESS)
    unified_worker.add_argument('folder')
    trace = commands.add_parser("trace", help="Trace runtime access operands at reviewed source lines")
    trace.add_argument("driver")
    trace.add_argument("--source", help="Kernel source file (relative to cwd); defaults to config source relative to config directory")
    trace.add_argument("--access", required=True)
    trace.add_argument("--output", required=True)
    trace.add_argument("--timeout", type=int, default=240)
    trace.add_argument("--sanitizer", choices=("racecheck", "synccheck", "memcheck"))
    access_worker = commands.add_parser("_access_worker", help=argparse.SUPPRESS)
    access_worker.add_argument("folder")
    argv = sys.argv[1:]
    separator = argv.index("--") if "--" in argv else len(argv)
    driver_args = argv[separator + 1:]
    args = parser.parse_args(argv[:separator])
    if driver_args and (args.command != "run" or args.engine != "source"):
        parser.error("driver arguments after -- require run --engine source")
    if args.command == "_source_worker":
        from .source_capture import worker as source_worker_run
        source_worker_run(args.folder)
        return
    if args.command == '_unified_worker':
        from .runtime.unified_capture import worker as unified_worker_run
        unified_worker_run(args.folder)
        return
    if args.command in ("trace", "_access_worker"):
        from . import access
        if args.command == "_access_worker":
            access.worker(args.folder)
        else:
            result = access.run(args.driver, args.access, args.output, args.timeout, args.sanitizer, source_path=args.source)
            print(json.dumps(result, indent=2))
            if result["numerical_status"] == "failed":
                raise SystemExit(2)
        return
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
        if args.engine == "source":
            from .source_capture import run as source_run
            result = source_run(args.driver, args.monitor, args.output, args.timeout, args.sanitizer, source_path=args.source, driver_args=driver_args)
        else:
            result = capture.run(args.driver, args.monitor, args.output, args.timeout, args.sanitizer, source_path=args.source)
        print(json.dumps(result, indent=2))
        if result.get('status') == 'partial':
            raise SystemExit(3)
        if result["numerical_status"] == "failed":
            raise SystemExit(2)
