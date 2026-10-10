"""Run each CPU suite in its own process (analysis must not import TileLang)."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args();folder=Path(args.output);folder.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    results=[]
    for file in ('test_cpu.py','test_evidence.py','test_numerics.py','test_analysis.py','test_ir.py',
                 'test_source_path.py','test_tileops_integration.py','test_source_engine.py','test_scopes.py','test_components.py','test_unified.py','test_source_access.py','test_frontend_contracts.py','test_runtime_parameters.py','test_reporting.py'):
        command=[sys.executable,'-m','unittest','discover','-s','tests/reporting' if file=='test_reporting.py' else 'tests','-p',file,'-v']
        result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=120)
        (folder/(file+'.log')).write_text(result.stdout+result.stderr)
        match=re.search(r'Ran (\d+) tests?',result.stderr)
        tests=int(match[1]) if match else 0
        passed=result.returncode==0 and tests>0 and 'skipped' not in result.stderr
        results.append(dict(suite=file,command=command,returncode=result.returncode,tests=tests,passed=passed))
        (folder/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(results[-1]),flush=True)
        if not passed:raise SystemExit(1)


if __name__=='__main__':main()
