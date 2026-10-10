import copy
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tilelang_debugger.diagnostics.values import samples, summarize, identity
from tilelang_debugger.diagnostics.operations import describe, checks
from tilelang_debugger.reporting.model import portable
from tilelang_debugger.reporting.render import html, generate
from tilelang_debugger.reporting.load import load
from tilelang_debugger.reporting.build import build


def record(number, dtype='float32', **extra):
    bits = struct.unpack('<I', struct.pack('<f', number))[0] if dtype == 'float32' else number % 2**64
    return dict(launch=0, compile=0, point='p', block=[0,0,0], thread=0, visit=0, coordinates=[], ordinals=[], index=0, dtype=dtype, bits=bits, **extra)


class ReportingTests(unittest.TestCase):
    def test_partial_and_unlaunched_keep_missing_values(self):
        point=dict(id='p',line=1,when='after',buffer='x',shape=[],dtype='float32',scope='local.var')
        data=dict(records=[],accesses=[],analysis=None,comparisons={},
                  contracts=[dict(launch=0,compile=0,points=[point])],
                  run=dict(status='partial',source_path='kernel.py',unlaunched_points=['later'],
                           coverage={'0':{'p':dict(capture_integrity='partial',truncated=True)}}),
                  monitor=dict(points=[point,dict(point,id='later')]),
                  execution=dict(compiles=[]),source='x = 1',environment={})
        model=build(data)
        self.assertEqual(model['analysis']['status'],'not_provided')
        self.assertEqual(model['points'][0]['coverage']['capture_integrity'],'partial')
        self.assertEqual(model['points'][1]['coverage']['capture_integrity'],'unlaunched')
        self.assertEqual(model['points'][1]['values'],[])
        self.assertEqual(model['numeric_summary']['samples'],0)
        self.assertIsNone(model['numeric_summary']['minimum'])

    def test_exact_int64_and_special_values(self):
        r=record(2**63-1,'int64')
        row=samples([r],{identity(r):dict(expected=2**63-2,matched=False)})[0]
        self.assertEqual(row['abs_error'],1)
        self.assertEqual(portable(row)['actual'],str(2**63-1))
        rows=samples([record(x) for x in (math.nan,math.inf,-math.inf,2)],{})
        summary=summarize(rows)
        self.assertEqual((summary['nan'],summary['posinf'],summary['neginf'],summary['finite']),(1,1,1,1))
        self.assertEqual(summary['minimum'],2)

    def test_reference_zero_and_nonfinite_comparison(self):
        for actual, expected, absolute, relative in [(1,0,1,'+Inf'), (0,0,0,0), (math.inf,'+Inf',0,0), (math.nan,'NaN',None,None)]:
            r=record(actual)
            row=samples([r],{identity(r):dict(expected=expected,matched=False)})[0]
            self.assertEqual((row['abs_error'],row['relative_error']),(absolute,relative))

    def test_operation_checks_require_explicit_operand_observations(self):
        op=describe('y = a / denominator',1)
        point=dict(launch=0,line=1,mode='value',when='before',buffer='denominator',shape=[],operation=op,dtype='float32',values=samples([record(0)],{}))
        result=checks([point],1e-8)
        self.assertEqual((result[0]['status'],result[0]['signals']),('observed',1))
        point['when']='after'
        self.assertEqual(checks([point],1e-8)[0]['status'],'not_collected')
        point.update(when='before',operation=describe('y = T.sqrt(denominator)',1),values=samples([record(-1)],{}))
        self.assertEqual(checks([point],1e-8)[0]['signals'],1)

    def test_conversion_pairing_does_not_cross_visits(self):
        operation=describe("y = T.cast(x, 'float16')",1)
        a=dict(launch=0,line=1,mode='value',when='before',buffer='x',shape=[],operation=operation,dtype='float32',values=samples([record(1.0001)],{}))
        b=dict(a,when='after',buffer='y',dtype='float16',values=samples([record(1)],{}))
        result=checks([a,b],1e-8)[0]
        self.assertEqual(result['samples'],1)
        self.assertGreater(result['max_abs_change'],0)
        b['values'][0]['visit']=1
        self.assertEqual(checks([a,b],1e-8)[0]['status'],'not_collected')

    def test_script_injection_and_large_integer_serialization(self):
        payload={'source': '</script><script>window.injected=true</script>', 'n':2**64-1}
        document=html(portable(payload))
        self.assertNotIn(payload['source'],document)
        self.assertIn('\\u003c/script\\u003e',document)
        self.assertIn('"18446744073709551615"',document)
        self.assertNotIn('src="https://',document)

    def test_output_separation_and_thresholds(self):
        with tempfile.TemporaryDirectory() as temp:
            capture=Path(temp)/'capture';capture.mkdir()
            for output in (capture,capture/'report',Path(temp)):
                with self.assertRaises(ValueError):generate(capture,output)
            with self.assertRaises(ValueError):generate(capture,Path(temp)/'report',near_zero=-1)

    def test_no_tilelang_or_torch_imported(self):
        subprocess.run([sys.executable,'-c',"from tilelang_debugger.reporting.render import generate; import sys; assert 'torch' not in sys.modules and 'tilelang' not in sys.modules and 'tvm' not in sys.modules"],check=True)

    def test_analysis_from_another_capture_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            capture=Path(temp)/'capture';analysis=Path(temp)/'analysis';capture.mkdir();analysis.mkdir()
            (capture/'run.json').write_text(json.dumps(dict(schema='source-unified-v3',run_id='a',sanitizer=None)))
            (capture/'records.jsonl').write_text('')
            (analysis/'analysis.json').write_text(json.dumps(dict(schema='unified-analysis-v3',status='completed',run_id='b')))
            with patch('tilelang_debugger.reporting.load.verify',return_value=[]):
                with self.assertRaisesRegex(ValueError,'belong'):load(capture,analysis)


if __name__=='__main__': unittest.main()
