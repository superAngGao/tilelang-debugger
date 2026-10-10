"""Real eager frontend contracts; no lowering or GPU launch needed."""
from collections import Counter
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import tilelang
from tvm import tirx, ir

from tilelang_debugger.frontend.access import operands
from tilelang_debugger.protocols.access import operand_fields
from tilelang_debugger.source_analysis.control_flow import prepare
from tilelang_debugger.instrumentation.unified import inject
from tilelang_debugger.runtime.builds import session


SOURCE = '''import tilelang.language as T
n = T.symbolic('n')
stride = T.symbolic('stride')
events = []
def dimension():
    events.append('dimension')
    return n
def step():
    events.append('step')
    return 1
def build():
    @T.prim_func
    def main(x: T.StridedTensor((n,), (stride,), 'int32')):
        with T.Kernel(1, threads=32):
            for i in T.serial(0, stop=dimension(), step=step()):
                value = x[i]
    return main
'''


class FrontendContracts(unittest.TestCase):
    def test_frontend_capabilities(self):
        from tilelang_debugger.frontend.compatibility import check
        result = check()
        self.assertEqual(result['status'], 'compatible_interfaces')
        self.assertEqual(result['tilelang'], tilelang.__version__)
        self.assertIn(result['region_op'], {'tl.tileop.region', 'tl.region'})
        with patch('tilelang.language.copy_op._normalize_copy_regions', return_value=[]):
            with self.assertRaisesRegex(RuntimeError, 'incompatible copy region normalization'):
                check()

    def test_scalar_encoding_rejects_lossy_or_vector_operands(self):
        x = tirx.decl_buffer((32,), 'int32')
        spec = dict(kind='element', operation='read', predicate='p', expression='x[0]')
        for dtype in ('float32', 'bfloat16', 'uint64', 'int32x2'):
            with self.assertRaisesRegex(ValueError, 'scalar integers'):
                operands(x[0], None, tirx.Var('p', dtype), spec)

    def test_effectful_macro_prefix_is_rejected(self):
        source = '''import tilelang.language as T
def change(v):
    @T.macro
    def bump():
        T.buffer_store(v.buffer, v + 1, 0)
    bump()
    return T.int32(0)
def build():
    @T.prim_func
    def main(x: T.Tensor((32,), 'int32'), y: T.Tensor((32,), 'int32')):
        with T.Kernel(1, threads=32):
            j = T.alloc_var('int32', init=0)
            value = change(j) + x[j]
            y[0] = value
    return main
'''
        for statement in ('value = change(j) + x[j]', 'j = x[j] = 1', 'j, x[j] = 1, 9'):
            text = source.replace('value = change(j) + x[j]', statement).replace('y[0] = value', 'y[0] = j')
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'effect_fixture.py'; path.write_text(text)
                spec = importlib.util.spec_from_file_location('effect_fixture', path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                function = module.build()
                stores = []
                tirx.stmt_functor.post_order_visit(function.body, lambda n: stores.append(n) if isinstance(n, tirx.BufferStore) and n.buffer.scope() == 'local.var' else None)
                self.assertGreaterEqual(len(stores), 2)
            cfg = dict(schema=3, source='kernel.py', points=[dict(id='p', line=next(i for i, s in enumerate(text.splitlines(), 1) if statement in s), when='before', buffer='x', mode='access', block=[0,0,0], loops=[])])
            with self.assertRaisesRegex(ValueError, 'evaluation prefix|multiple assignment'):
                prepare(text, cfg)
        # A later macro emits its store before the final deferred BufferLoad.
        # Only mutable operands are affected; a constant index remains valid.
        for index in ('j', '0'):
            text = source.replace('change(j) + x[j]', f'x[{index}] + change(j)')
            cfg = dict(schema=3, source='kernel.py', points=[dict(id='p', line=13, when='before', buffer='x', mode='access', block=[0,0,0], loops=[])])
            points = prepare(text, cfg)
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'suffix_fixture.py'; path.write_text(inject(text, points))
                spec = importlib.util.spec_from_file_location('suffix_fixture', path)
                module = importlib.util.module_from_spec(spec)
                with session(points, True):
                    spec.loader.exec_module(module)
                    if index == 'j':
                        with self.assertRaisesRegex(ValueError, 'mutable access operands'):
                            module.build()
                    else:
                        module.build()

    def build(self, source, enabled):
        config = dict(schema=3, source='kernel.py', points=[dict(id='p', line=next(i for i,s in enumerate(source.splitlines(), 1) if 'value = x' in s), when='before', buffer='x', mode='access', thread=0, block=[0,0,0], loops=[])])
        points = prepare(source, config)
        staged = inject(source, points, enabled=enabled)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'kernel.py'
            path.write_text(staged)
            spec = importlib.util.spec_from_file_location('binding_fixture', path)
            module = importlib.util.module_from_spec(spec)
            with session(points, enabled) as registry:
                spec.loader.exec_module(module)
                function = module.build()
            result = list(function.buffer_map.values())[0]
            self.assertEqual(module.n.name, 'n')
            self.assertEqual(module.stride.name, 'stride')
            self.assertTrue(result.shape[0].same_as(module.n))
            self.assertTrue(result.strides[0].same_as(module.stride))
            self.assertEqual(str(result.name), 'x')
            return module.events, function, registry

    def test_binding_preserves_names_identity_and_evaluation_count(self):
        for source in (SOURCE, SOURCE.replace('stop=dimension()', 'stop=n')):
            baseline, before, _ = self.build(source, False)
            instrumented, after, registry = self.build(source, True)
            self.assertEqual(instrumented, baseline)
            self.assertEqual(baseline, ['dimension', 'step'] if 'stop=dimension()' in source else ['step'])
            self.assertEqual(registry['builds'][0][0]['access']['fields'], operand_fields(1))
            # Actual frontend buffer reads are unchanged by access observation.
            def loads(function):
                found = Counter()
                def visit(node):
                    if isinstance(node, tirx.BufferLoad) and node.buffer.scope() == 'global':
                        found[str(node.buffer.name)] += 1
                tirx.stmt_functor.post_order_visit(function.body, visit)
                return found
            self.assertEqual(loads(before), loads(after))

    def test_copy_regions_and_memory_scope_are_frontend_facts(self):
        x = tirx.decl_buffer((4, 49), 'int32', name='x')
        shared = tirx.decl_buffer((1, 32), 'int32', name='scratch', scope='shared.dyn')
        spec = dict(kind='copy', side=0, operation='read', predicate='True', expression='x[1, 32]')
        meta, values = operands(x[1, 32], shared, True, spec)
        self.assertEqual(meta['memory_scope'], 'global')
        self.assertEqual(meta['fields'], operand_fields(2))
        self.assertEqual([int(v) for v in values[1:5]], [1, 32, 1, 32])
        self.assertEqual([int(v) for v in values[5:9]], [4, 49, 49, 1])
        meta, values = operands(shared, x[1, 32], True, dict(spec, side=1, operation='write'))
        self.assertEqual(meta['memory_scope'], 'shared.dyn')
        self.assertEqual([int(v) for v in values[1:5]], [0, 0, 1, 32])
        sliced = tirx.BufferRegion(x, [ir.Range.from_min_extent(1, 1), ir.Range.from_min_extent(3, 8)])
        other = tirx.decl_buffer((1, 8), 'int32')
        _, values = operands(sliced, other, True, spec)
        self.assertEqual([int(v) for v in values[1:5]], [1, 3, 1, 8])
        scalar = dict(spec, kind='element')
        _, values = operands(x[2, 7], None, True, scalar)
        self.assertEqual([int(v) for v in values[1:5]], [2, 7, 1, 1])

    def test_adapter_rejects_hidden_memory_reads(self):
        x = tirx.decl_buffer((32,), 'int32')
        index = tirx.decl_buffer((1,), 'int32')
        spec = dict(kind='element', operation='read', predicate='True', expression='x[index[0]]')
        with self.assertRaisesRegex(ValueError, 'memory read'):
            operands(x[index[0]], None, True, spec)


if __name__ == '__main__':
    unittest.main()
