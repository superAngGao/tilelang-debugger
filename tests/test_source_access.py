import ast
import copy
import unittest

from tilelang_debugger.source_analysis.control_flow import prepare
from tilelang_debugger.instrumentation.unified import inject
from tilelang_debugger.instrument import Unsupported
from tilelang_debugger.protocols.access import report


def source(statement):
    return 'import tilelang.language as T\ndef f():\n    with T.Kernel(1, threads=32):\n        ' + statement + '\n'


def config(**options):
    return dict(schema=3, source='kernel.py', points=[dict(id='p', line=4, when='before', mode='access', buffer='x', block=[0, 0, 0], loops=[], **options)])


class SourceAccessTests(unittest.TestCase):
    def test_read_write_and_augmented_assignment(self):
        for statement, role in [('v = x[i]', 'read'), ('x[i] = v', 'write')]:
            p = prepare(source(statement), config())[0]
            self.assertEqual(p['access_spec']['operation'], role)
        with self.assertRaisesRegex(Unsupported, 'multiple'):
            prepare(source('x[i] += 1'), config())
        for role in ('read', 'write'):
            self.assertEqual(prepare(source('x[i] += 1'), config(operation=role))[0]['access_spec']['operation'], role)

    def test_multiple_sites_need_explicit_selection(self):
        text = source('y[i] = x[i] + x[j]')
        with self.assertRaisesRegex(Unsupported, 'multiple'):
            prepare(text, config())
        self.assertEqual(prepare(text, config(occurrence=1))[0]['access_spec']['expression'], 'x[j]')
        c = config(); c['points'][0]['buffer'] = 'x[j]'
        self.assertEqual(prepare(text, c)[0]['access_spec']['expression'], 'x[j]')

    def test_masks_and_original_statement_preserved(self):
        text = source('v = T.if_then_else(i < n, x[i], 0)')
        p = prepare(text, config())
        self.assertEqual(p[0]['access_spec']['predicate'], 'i < n')
        tree = ast.parse(inject(text, p))
        self.assertEqual(sum(isinstance(n, ast.Call) and ast.unparse(n.func) == 'T.if_then_else' for n in ast.walk(tree)), 1)
        self.assertIn('_observe_access(x[i], None, i < n,', ast.unparse(tree))

    def test_no_additional_memory_reads_or_calls_in_operands(self):
        for statement in ('v = x[index[i]]', 'v = x[next_index()]', 'v = T.if_then_else(mask[i], x[i], 0)'):
            with self.assertRaisesRegex(Unsupported, 'scalar binding'):
                prepare(source(statement), config())
        with self.assertRaisesRegex(Unsupported, 'undefined inactive index'):
            prepare(source('v = T.if_then_else(n != 0, x[i // n], 0)'), config())

    def test_copy_regions_are_source_operands(self):
        p = prepare(source('T.copy(x[k * 32], shared)'), config())[0]
        self.assertEqual(p['access_spec']['kind'], 'copy')
        self.assertEqual(p['access_spec']['destination'], 'shared')
        p = prepare(source('T.copy(x[k:k+8, 0:16], shared)'), config())[0]
        self.assertEqual(p['access_spec']['kind'], 'copy')
        self.assertIn('k:k + 8', inject(source('T.copy(x[k:k+8, 0:16], shared)'), [p]))

    def test_buffer_calls_are_not_mislabeled_as_scalar_reads(self):
        for statement in ('T.fill(x[0:4], 0)', 'custom_macro(x[i])'):
            with self.assertRaisesRegex(Unsupported, 'semantics need an adapter'):
                prepare(source(statement), config())

    def test_selection_precedes_replay_validation(self):
        for statement in ('v = x[i] + x[index[j]]', 'v = x[i] and x[j]', 'v = x[i] + custom(x[j])',
                          'v = x[i] + T.atomic_add(**kwargs)', 'v = x[i] + T.copy(**kwargs)'):
            self.assertEqual(prepare(source(statement), config(occurrence=0))[0]['access_spec']['expression'], 'x[i]')
        for statement in ('v = custom() + x[i]', 'x[i] = custom()', 'v = T.atomic_add(y[0], 1) + x[i]',
                          'v = custom()[0] + x[i]', 'v = factory()() + x[i]',
                          'v = T.address_of(custom()) + x[i]', 'custom()[0] += x[i]'):
            with self.assertRaisesRegex(Unsupported, 'evaluation prefix'):
                prepare(source(statement), config())
        for statement in ('j = x[j] = 1', 'j, x[j] = 1, 9'):
            with self.assertRaisesRegex(Unsupported, 'multiple assignment'):
                prepare(source(statement), config())

    def test_operation_arguments_keep_source_evaluation_order(self):
        for statement in ('T.atomic_add(value=custom(), dst=x[i])',
                          'T.atomic_add(dst=y[custom()], value=x[i])',
                          'v = T.address_of(y[custom()]) + x[i]'):
            with self.assertRaisesRegex(Unsupported, 'evaluation prefix'):
                prepare(source(statement), config())

    def test_conditional_paths(self):
        for statement, predicate in [('v = T.And(i < n, x[i] > 0)', 'i < n'),
                                     ('v = T.Or(i >= n, x[i] > 0)', 'not i >= n'),
                                     ('v = i < n < x[i]', 'i < n'),
                                     ('v = x[i] if i < n else 0', 'i < n'),
                                     ('v = 0 if i < n else x[i]', 'not i < n')]:
            self.assertEqual(prepare(source(statement), config())[0]['access_spec']['predicate'], predicate)

    def test_speculative_domain_checks_include_predicates(self):
        for statement in ('v = x[1 << -1] if False else 0',
                          'v = x[i << shift] if i < n else 0',
                          'v = (n != 0 and i // n > 0) and x[i]',
                          'v = x[i // -1] if i < n else 0',
                          'T.copy(x, y[1 // 0]) if False else T.copy(z, y)'):
            with self.assertRaisesRegex(Unsupported, 'undefined inactive'):
                prepare(source(statement), config())

    def test_explicit_access_selector_uses_shared_expression_rules(self):
        for statement, selector in [('v = x[i & 31]', 'x[i & 31]'),
                                    ('v = x[T.cast(i, T.int32)]', 'x[T.cast(i, T.int32)]'),
                                    ('T.copy(x[i:i+8], shared)', 'x[i:i+8]')]:
            plain = prepare(source(statement), config())[0]['access_spec']
            c = config(); c['points'][0]['buffer'] = selector
            self.assertEqual(prepare(source(statement), c)[0]['access_spec'], plain)

    def test_operation_adapters(self):
        for statement in ('T.copy(src=x[i:i+8], dst=shared)', 'T.copy(x[i:i+8], dst=shared, disable_tma=True)'):
            self.assertEqual(prepare(source(statement), config())[0]['access_spec']['kind'], 'copy')
        for name in ('atomic_add', 'atomic_min', 'atomic_max'):
            p = prepare(source(f'T.{name}(dst=x[i], value=1)'), config(operation='read_write'))[0]
            self.assertEqual(p['access_spec']['kind'], 'atomic')
        p = prepare(source('v = T.address_of(x[i])'), config(operation='address'))[0]
        self.assertEqual(p['access_spec']['kind'], 'address')
        with self.assertRaisesRegex(Unsupported, 'single element'):
            prepare(source('T.atomic_add(x[0:4], shared)'), config())

    def test_access_is_before_without_collective_reads(self):
        for change in ({'when': 'after'}, {'collective': {'ready': True}}, {'region': [[0, 1, 1]]}):
            c = config(); c['points'][0].update(change)
            with self.assertRaises(Unsupported):
                prepare(source('v = x[i]'), c)

    def test_branch_local_points_do_not_invent_masked_events(self):
        text = source('if i < n:\n            v = x[i]')
        c = config(); c['points'][0]['line'] = 5
        p = prepare(text, c)[0]
        self.assertEqual(p['scopes'][0]['condition'], 'i < n')
        self.assertEqual(p['access_spec']['predicate'], 'True')

    def test_report_bounds_offsets_mask_and_completeness(self):
        p = prepare(source('v = x[i]'), config())[0]
        p['access'] = dict(rank=1, fields=['active', 'origin.0', 'extent.0', 'shape.0', 'stride.0', 'elem_offset'], operation='read', kind='element', buffer='x', memory_scope='global', dtype='int32', predicate='i < n', branches=[])
        contracts = [dict(launch=0, points=[p])]
        coverage = {'0': {'p': dict(capture_integrity='complete')}}
        def records(values):
            return [dict(launch=0, compile=0, point='p', thread=0, visit=0, block=[0,0,0], coordinates=[], ordinals=[], index=i, dtype='int64', bits=v % 2**64) for i, v in enumerate(values)]
        for payload, expected in [([1, 2, 1, 4, 2, 1], 'in_bounds'), ([0, -1, 1, 4, 2, 1], 'masked'), ([1, 3, 4, 4, 2, 1], 'partial'), ([1, 5, 1, 4, 2, 1], 'out_of_bounds')]:
            summary, rows = report(records(payload), contracts, coverage)
            self.assertEqual(rows[0]['status'], expected)
            self.assertEqual(rows[0]['derived_byte_offset'], (payload[1] * 2 + 1) * 4)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            report(records([1, 2, 1, 4, 2]), contracts, coverage)


if __name__ == '__main__':
    unittest.main()
