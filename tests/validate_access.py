"""H200 acceptance: actual indices, masks, TMA regions, output equality and sanitizers."""
import argparse
import json
from pathlib import Path
from tilelang_debugger.access import run
from tilelang_debugger.capture import save_json


def verify(folder,case):
    data=json.loads((folder/'run.json').read_text())
    if data['status']!='passed' or data['numerical_status']!='passed':raise AssertionError(data)
    points={p['site']:p for p in json.loads((folder/'access-points.json').read_text())}
    records=[json.loads(l) for l in (folder/'access-records.jsonl').read_text().splitlines()]
    if len(records)!={'gelu':4096,'sum':512,'sum_unpadded':1280,'gemm':2,'gqa':6}[case]:raise AssertionError('event count')
    for r in records:
        p=points[r['site']];bx,by,bz=r['block'];tx=r['thread'];lane=r['vector_lane'];loops=r['loops']
        if case=='gelu':expected=bx*2048+loops[0]*1024+tx*8+lane
        elif case=='sum':
            expected=bx*514+loops[0]*257+loops[1]*128+tx
            assert r['active']==(loops[1]*128+tx<257)
        elif case=='sum_unpadded':expected=(bx*512 if p['buffer']=='x' else loops[0]*256 if p['line']==47 else 0)+tx*p['width']+lane
        elif case=='gemm':
            assert r['coordinates']==[64,0 if p['buffer']=='a' else bx*128]
            assert r['shared_element_offset']==8192 and r['barrier_index']==1
            continue
        else:
            if p['line'] in (120,128):assert r['coordinates']==[0,128,0,0] and r['shared_element_offset']==8192
            else:assert r['coordinates']==[0,bx*128+(64 if p['line'] in (115,455) else 0),by,0]
            continue
        assert r['element_offset']==expected
    if case=='sum':assert sum(r['active'] for r in records)==257
    analysis=json.loads((folder/'access-analysis.json').read_text())
    assert analysis['complete']
    if case in ('gemm','gqa'):
        assert all(p['global_bounds']=='in_bounds' for p in analysis['points'])
        if case=='gqa':assert next(p for p in analysis['points'] if p['line']==455)['region_in_tensor_axes']==[[0,1],[192,256],[0,1],[0,64]]
    for mode in ('baseline','instrumented'):
        gate=json.loads((folder/mode/'launch-gate.json').read_text());assert gate['codegen_erasure_verified']
    return dict(case=case,records=len(records),passed=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--sanitizer',choices=['racecheck','synccheck','memcheck'],required=True)
    parser.add_argument('--cases',nargs='+',default=['gelu','sum','sum_unpadded','gemm','gqa'])
    args=parser.parse_args();root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=False)
    results=[]
    for case in args.cases:
        base=Path('examples')/('sum' if case=='sum_unpadded' else case)
        driver=base/('run_unpadded.py' if case=='sum_unpadded' else 'run.py')
        config=base/('access_unpadded.json' if case=='sum_unpadded' else 'access.json')
        run(driver,config,root/case,600,args.sanitizer)
        results.append(verify(root/case,case));save_json(root/'summary.json',results)
        print(json.dumps(results[-1]),flush=True)


if __name__=='__main__':main()
