"""CPU/TIR regression tests against saved real modules; set TLACC_FIXTURE to GQA trace."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get('TLACC_FIXTURE'),'requires saved H200 compilation artifacts, no GPU execution')
class AccessIRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tilelang
        import tvm
        from tvm import tirx as T
        cls.tvm=tvm;cls.T=T
        p=Path(os.environ['TLACC_FIXTURE'])/'baseline'
        cls.module=tvm.ir.load_json((p/'pretrace.json').read_text())
        cls.frontend=tvm.ir.load_json((p/'frontend.json').read_text())
        cls.points=json.loads((p/'request.json').read_text())['points']
        cls.host_gv=next(gv for gv,f in cls.module.functions.items() if int(f.attrs.get('calling_conv',0))!=2)
        cls.device_gv=next(gv for gv,f in cls.module.functions.items() if int(f.attrs.get('calling_conv',0))==2)

    def host(self,module):
        from tilelang_debugger.access_host import HostBindings
        return HostBindings(module,self.frontend,[3],'gqa')

    def mutate_host(self,callback):
        T=self.T;m=self.tvm.IRModule(dict(self.module.functions),attrs=self.module.attrs)
        f=m[self.host_gv];body=T.stmt_functor.ir_transform(f.body,None,callback)
        m.update_func(self.host_gv,f.with_body(body));return m

    def test_original_and_comparison_roundtrip(self):
        from tilelang_debugger.access_ir import comparison_view
        self.host(self.module)
        view=comparison_view(self.module)
        self.assertTrue(self.tvm.ir.structural_equal(view,comparison_view(self.tvm.ir.load_json(self.tvm.ir.save_json(self.module))),map_free_vars=True))

    def test_hidden_constructor_or_repeated_launch(self):
        T=self.T
        for name,extent in [('__tvm_tensormap_create_tiled',0),(self.device_gv.name_hint,2)]:
            def mutate(n):
                if isinstance(n,T.Evaluate) and isinstance(n.value,T.Call) and n.value.op.name=='tirx.tvm_call_packed' and n.value.args[0].value==name:
                    return T.For(T.Var('injected','int32'),0,extent,T.ForKind.SERIAL,n)
            with self.assertRaises(ValueError):self.host(self.mutate_host(mutate))

    def test_wrong_packed_discriminator_and_scalar_cast(self):
        T=self.T
        def wrong_condition(n):
            if isinstance(n,T.Bind) and n.var.name=='Q_handle':
                return T.Bind(n.var,T.Select(T.IntImm('bool',0),n.value.true_value,n.value.false_value))
        with self.assertRaises(ValueError):self.host(self.mutate_host(wrong_condition))
        def wrong_cast(n):
            if isinstance(n,T.Bind) and n.var.name=='seq_len_q':return T.Bind(n.var,T.Cast('int32',T.Cast('int8',n.value.value)))
        host=self.host(self.mutate_host(wrong_cast))
        # scalar() operates on shape metadata; CPU fake tensors suffice and never launch.
        from types import SimpleNamespace
        host.tensors={0:SimpleNamespace(shape=[1,256,2,64])}
        with self.assertRaises(ValueError):host.scalar(next(v for v in host.binds if v.name=='seq_len_q'))

    def test_bad_descriptor_and_guard(self):
        T=self.T
        def bad_ctor(n):
            if isinstance(n,T.Evaluate) and isinstance(n.value,T.Call) and n.value.op.name=='tirx.tvm_call_packed' and n.value.args[0].value=='__tvm_tensormap_create_tiled':
                args=list(n.value.args);args[5]=T.IntImm('int32',65)
                return T.Evaluate(T.Call(n.value.dtype,n.value.op,args))
        with self.assertRaises(ValueError):self.host(self.mutate_host(bad_ctor))
        from tilelang_debugger.access_ir import instrument
        f=self.module[self.device_gv]
        instrument(f,self.points)
        def wrong_elect(n):
            if isinstance(n,T.Call) and n.op.name.split('.')[-1]=='tl_shuffle_elect' and int(n.args[0])==32:
                return T.Call(n.dtype,n.op,[T.IntImm('int32',64)])
        f=f.with_body(T.stmt_functor.ir_transform(f.body,None,wrong_elect))
        with self.assertRaises(ValueError):instrument(f,self.points)

    def test_pipeline_restored_on_exception(self):
        import importlib
        from tilelang.backend.pass_pipeline.pipeline import get_pipeline
        from tilelang_debugger.access_ir import session
        lower=importlib.import_module('tilelang.engine.lower')
        original=get_pipeline('cuda');prepare=lower._prepare_device_codegen_mod
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'deliberate'):
                with session(Path(tmp),self.points,True):raise RuntimeError('deliberate')
            self.assertIs(get_pipeline('cuda'),original)
            self.assertIs(lower._prepare_device_codegen_mod,prepare)


if __name__=='__main__':unittest.main()
