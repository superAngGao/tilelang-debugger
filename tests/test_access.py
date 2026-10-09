"""CPU protocol/config tests. Synthetic records are not GPU validation evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from tilelang_debugger.access_contracts import prepare
from tilelang_debugger.access_records import parse,join_words,report,validate_workers

ROOT=Path(__file__).resolve().parents[1]


class AccessTests(unittest.TestCase):
    def test_source_selection(self):
        for case in ("gelu","sum","sum_unpadded","gemm","gqa"):
            folder=ROOT/"examples"/("sum" if case=="sum_unpadded" else case)
            config=json.loads((folder/("access_unpadded.json" if case=="sum_unpadded" else "access.json")).read_text())
            source=(folder/"kernel.py").read_text(encoding="utf-8")
            driver=(folder/("run_unpadded.py" if case=="sum_unpadded" else "run.py")).read_bytes()
            self.assertEqual(prepare(source,driver,config)[0],case)
            for key,value in (("line",999),("block",[99,0,0]),("operation","anything"),("loops",[{"line":0,"iteration":0}])):
                invalid=copy.deepcopy(config);invalid["accesses"][0][key]=value
                with self.assertRaises(ValueError):prepare(source,driver,invalid)
            with self.assertRaises(ValueError):prepare(source+"\n# change",driver,config)
            with self.assertRaises(ValueError):prepare(source,driver+b"\n",config)

    def test_words(self):
        for signed,values in ((True,[-2**63,-1,0,2**31,2**32+17,2**63-1]),(False,[0,2**32,2**64-1])):
            for v in values:
                bits=v%2**64
                self.assertEqual(join_words(bits%2**32,bits>>32,signed),v)
        for pair in ((-1,0),(2**32,0),(0,2**32)):
            with self.assertRaises(ValueError):join_words(*pair,True)

    def fixture(self):
        point=dict(site="sum_L35_read_x",id="x",block=[0,0,0],elected=None,signed=[True]*6,
                   shape=[10],line=35,operation="read",buffer="x",expected_keys=[dict(thread=i,loops=[0]*4,lane=0,active=int(i==0)) for i in (0,1)])
        rows=[]
        for i in (0,1):
            fields=[0,0,0,i,0,0,0,0,0,int(i==0),123,0,*[0]*10]
            rows.append("TLACC1|sum_L35_read_x|0|"+"|".join(map(str,fields)))
        return point,rows

    def test_coverage_and_wrong_offset_retained(self):
        p,rows=self.fixture()
        records=parse("\n".join(reversed(rows)),[p],"run")
        self.assertEqual(records[0]["element_offset"],123)
        self.assertEqual(records[0]["bounds"],"out_of_bounds")
        self.assertEqual(records[1]["bounds"],"masked")
        for invalid in ([rows[0]],rows+[rows[0]],[rows[0],rows[1][:-2]],[rows[0].replace('|sum_L35','|unknown_L35'),rows[1]]):
            with self.assertRaises(ValueError):parse("\n".join(invalid),[p],"run")
        invalid=rows.copy();fields=invalid[1].split('|');fields[12]='1';invalid[1]='|'.join(fields)
        with self.assertRaises(ValueError):parse('\n'.join(invalid),[p],'run')

    def test_tma_region_uses_recorded_coordinate(self):
        p=dict(site="gqa_L455_transfer_Os",id="out",line=455,operation="transfer",buffer="Os",descriptor="O_desc",rank=4,kind="tma_store",shared_extent=4096,shared_alias_bytes=[81920,98304])
        r=dict(site=p['site'],active=True,coordinates=[0,192,0,0],shared_element_offset=4096,barrier_index=None)
        d=dict(name="O_desc",rank=4,settings=[0,3,2,0],global_dim=[64,256,2,1],box_dim=[64,64,1,1],mode_to_tensor_axis=[3,1,2,0])
        analysis,_=report([r],[p],[d]);item=analysis['points'][0]
        self.assertEqual(item['region_in_tensor_axes'],[[0,1],[192,256],[0,1],[0,64]])
        self.assertEqual(item['global_bounds'],'in_bounds')
        r['coordinates'][1]=240
        self.assertEqual(report([r],[p],[d])[0]['points'][0]['global_bounds'],'store_oob_discard')
        r['coordinates'][1]=-8
        self.assertEqual(report([r],[p],[d])[0]['points'][0]['global_bounds'],'invalid_store_origin')
        with self.assertRaises(ValueError):report([r],[p],[])

    def test_sanitizer_formats_and_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for mode in ('baseline','instrumented'):
                p=root/mode;p.mkdir()
                for name,data in {'process':dict(returncode=0,timeout=False),'pipeline-state':dict(pipeline_calls=1,codegen_calls=1,restored=True),'launch-gate':dict(passed=True),'command':['compute-sanitizer','--tool','racecheck','--error-exitcode','86']}.items():
                    (p/f'{name}.json').write_text(json.dumps(data))
                (p/'racecheck.log').write_text('========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n')
            validate_workers(root,'racecheck')
            (root/'instrumented/process.json').write_text(json.dumps(dict(returncode=1,timeout=False)))
            with self.assertRaises(ValueError):validate_workers(root,'racecheck')


if __name__=='__main__':unittest.main()
