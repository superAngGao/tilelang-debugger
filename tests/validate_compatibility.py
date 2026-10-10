"""Run one installed TileLang version through the public capture matrix."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected-version', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tileops', required=True)
    args = parser.parse_args()
    import tilelang
    import tilelang_debugger
    assert tilelang.__version__ == args.expected_version, (tilelang.__version__, tilelang.__file__)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(tilelang=tilelang.__version__, tilelang_path=tilelang.__file__, debugger_path=tilelang_debugger.__file__, python=sys.executable)
    (output / 'environment.json').write_text(json.dumps(environment, indent=2))
    suites = [('cpu', 'run_cpu.py', []),
              ('contexts', 'validate_access_contexts.py', []),
              ('expressions', 'validate_access_expressions.py', []),
              ('runtime', 'validate_runtime_expansion.py', []),
              ('access', 'validate_source_access.py', ['--sanitizer', 'racecheck']),
              ('values', 'validate_unified.py', ['--sanitizer', 'synccheck', '--case', 'pipeline', '--case', 'group', '--case', 'fragment', '--case', 'shared', '--case', 'dynamic']),
              ('tileops', 'validate_source_access_tileops.py', ['--tileops', args.tileops, '--sanitizer', 'synccheck'])]
    results = []
    for name, script, options in suites:
        command = [sys.executable, str(Path(__file__).with_name(script)), '--output', str(output / name), *options]
        with (output / (name + '.log')).open('w') as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=2400).returncode
        results.append(dict(suite=name, returncode=code, passed=code == 0, command=command))
        (output / 'summary.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(results[-1]), flush=True)
        if code:
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
