"""H200: maximum 22-argument FIFO budget, signed/unsigned ABI, wrong-index fidelity.

Dedicated test kernel; not an authorization escape hatch for product contracts.
Parent records process exit independently of child validation.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def worker(folder):
    os.environ['TILELANG_DISABLE_CACHE']='1'
    import tilelang
    import tilelang.language as L
    import tvm
    from tvm import tirx as T
    from tilelang.backend.pass_pipeline.pipeline import get_pipeline,register_pipeline,PassPipeline
    from tilelang_debugger.capture import configure_runtime,save_json
    from tilelang_debugger.access_ir import rewrite,encode_words,scalar_indices
    torch,_=configure_runtime(folder)
    @L.prim_func
    def probe(A:L.Tensor((65536,),"int32"),O:L.Tensor((65536,),"int32")):
        with L.Kernel(512,threads=128) as bx:
            tx=L.get_thread_binding()
            i=bx*128+tx
            O[i]=A[(i+1)%65536]
    original=get_pipeline('cuda')
    count=0
    def pipeline(mod,target):
        nonlocal count
        mod=original.lower(mod,target)
        changed=tvm.IRModule(dict(mod.functions),attrs=mod.attrs)
        for gv,f in mod.functions.items():
            if int(f.attrs.get('calling_conv',0))!=2:continue
            def add(n):
                nonlocal count
                if not isinstance(n,T.BufferStore):return None
                if not isinstance(n.value,T.BufferLoad):raise ValueError('probe must retain direct load')
                count+=1
                index=scalar_indices(n.value)[0]
                ident=scalar_indices(n)[0]
                values=[T.IntImm('int64',-(1<<63)),T.IntImm('int64',-1),T.IntImm('int64',(1<<31)+7),
                        T.IntImm('int64',(1<<32)+9),T.const((1<<64)-1,'uint64'),index]
                fmt='TLACC_PROBE|'+('S'*64)+'|'+'|'.join(['%d']*10+['%u']*12)+'\n'
                args=[ident,*[T.IntImm('int32',-2147483647)]*9,*encode_words(values)]
                return T.SeqStmt([T.Evaluate(T.call_extern('int32','printf',fmt,*args)),n])
            changed.update_func(gv,f.with_body(rewrite(f.body,add)))
        return changed
    register_pipeline(PassPipeline('cuda',pipeline))
    try:
        kernel=tilelang.compile(probe,out_idx=[1],target='cuda')
        if count!=1:raise ValueError('probe access count differs')
        (folder/'kernel.cu').write_text(kernel.get_kernel_source())
        x=torch.arange(65536,device='cuda',dtype=torch.int32)
        out=kernel(x);torch.cuda.synchronize()
        if not torch.equal(out,torch.roll(x,-1)):raise ValueError('wrong-index fixture behavior changed')
    finally:register_pipeline(original)
    save_json(folder/'worker.json',dict(restored=get_pipeline('cuda') is original,records=65536))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--worker',action='store_true');args=p.parse_args()
    folder=Path(args.output).resolve()
    if args.worker:return worker(folder)
    folder.mkdir(parents=True,exist_ok=False)
    from tilelang_debugger.capture import save_json
    from tilelang_debugger.access_records import join_words
    command=[sys.executable,str(Path(__file__).resolve()),'--worker','--output',str(folder)]
    save_json(folder/'command.json',command)
    with (folder/'stdout.log').open('w') as out,(folder/'stderr.log').open('w') as err:
        result=subprocess.run(command,stdout=out,stderr=err,timeout=240)
    save_json(folder/'process.json',dict(returncode=result.returncode,timeout=False))
    if result.returncode:raise RuntimeError('probe worker failed; see stderr.log')
    seen=set()
    for line in (folder/'stdout.log').read_text().splitlines():
        if not line.startswith('TLACC_PROBE|'):continue
        parts=line.split('|');values=list(map(int,parts[2:]))
        if len(values)!=22:raise ValueError('truncated protocol probe')
        i=values[0]
        if not 0<=i<65536 or i in seen or values[1:10]!=[-2147483647]*9:raise ValueError('duplicate/unknown probe identity')
        seen.add(i)
        observed=[join_words(values[10+2*j],values[11+2*j],j!=4) for j in range(6)]
        if observed!=[-2**63,-1,2**31+7,2**32+9,2**64-1,(i+1)%65536]:raise ValueError('GPU ABI/index fidelity failure')
    if len(seen)!=65536:raise ValueError(f'FIFO loss: {len(seen)}/65536')
    save_json(folder/'validation.json',dict(passed=True,records=len(seen),max_numeric_args=22,wrong_index_preserved=True))
    print('PASS: 65536 records, signed/unsigned ABI and original shifted load index')


if __name__=='__main__':main()
