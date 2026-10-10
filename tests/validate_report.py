"""Offline integration against previously accepted H200 captures; no GPU launch."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

from tilelang_debugger.analysis import manifest
from tilelang_debugger.reporting.render import generate
from tilelang_debugger.reporting.load import load


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    root,output=Path(args.inputs).resolve(),Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    cases=[('mixed','access/shift-0','access/analysis-0',8),
           ('fragment','values/fragment','values/fragment-analysis',2),
           ('strided','runtime/strided','runtime/strided-analysis',6),
           ('no-reference','expressions/logic',None,7)]
    results=[]
    for name,capture,analysis,count in cases:
        capture=root/capture;analysis=root/analysis if analysis else None
        before=manifest(capture)
        result=generate(capture,output/name,analysis)
        model=json.loads((output/name/'report.json').read_text())
        assert result['points']==count
        assert model['run']['status']=='passed'
        assert model['analysis']['status']==('linked' if analysis else 'not_provided')
        assert len(model['compiler'])==2
        assert manifest(capture)==before==manifest(output/name/'evidence/capture')
        if analysis: assert manifest(analysis)==manifest(output/name/'evidence/analysis')
        # Entire report directory remains self-contained after copying elsewhere.
        with tempfile.TemporaryDirectory() as temp:
            moved=Path(temp)/'moved';shutil.copytree(output/name,moved)
            load(moved/'evidence/capture',moved/'evidence/analysis' if analysis else None)
            evidence=json.loads((moved/'manifest.json').read_text())
            actual=manifest(moved);actual.pop('manifest.json')
            assert evidence['files']==actual
        results.append(dict(case=name,passed=True,points=count,html_bytes=(output/name/'report.html').stat().st_size))
    # Corruption must fail instead of producing a green report.
    with tempfile.TemporaryDirectory() as temp:
        copied=Path(temp)/'capture';shutil.copytree(root/'access/shift-0',copied)
        log=copied/'records.jsonl';log.write_bytes(log.read_bytes()+b'{}\n')
        try:generate(copied,Path(temp)/'report')
        except ValueError:pass
        else:raise AssertionError('corrupt capture accepted')
    (output/'summary.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results))


if __name__=='__main__':main()
