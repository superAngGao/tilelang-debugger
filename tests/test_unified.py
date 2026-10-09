import ast
import copy
import unittest

from tilelang_debugger.source_analysis.control_flow import prepare
from tilelang_debugger.instrumentation.unified import inject
from tilelang_debugger.protocols.unified import parse, value
from tilelang_debugger.unified_analysis import compare_samples
from tilelang_debugger.runtime.unified_capture import split_launches
from tilelang_debugger.instrument import Unsupported
from tilelang_debugger.emitters.unified import frontend_thread


SOURCE = '''import tilelang.language as T
def build():
    @T.prim_func
    def main(x: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32):
            for k in T.serial(2, 8, step=2):
                v = k + 1
    return main
'''


def configuration(source=SOURCE):
    return dict(schema=3, source='kernel.py', points=[dict(id='p', line=next(i for i, line in enumerate(source.splitlines(), 1) if 'v =' in line), when='after', buffer='v', block=[0, 0, 0], loops=[])])


def contract():
    p = prepare(SOURCE, configuration())[0]
    p.update(bound=True, thread_extents=[2, 1, 1], threads=2, dtype='int64', indices=[0, 1], elements=2,
             shape=[2], scope='local', capacity=3, reader=None)
    return p


def packet(kind, tid=0, visit=0, index=0, bits=0, coords=(), ordinals=()):
    words = []
    for coordinate in (*coords, *ordinals):
        bits64 = coordinate % 2**64
        words += [bits64 & 0xffffffff, bits64 >> 32]
    return '|'.join(map(str, ('TLDBG3', 'p', kind, 0, 0, 0, tid, 0, 0, visit, index, bits & 0xffffffff, bits >> 32, *words, 'Z')))


def log(count=2):
    return [packet('D', 0, v, i, 9007199254740993 + i, [2 + v * 2], [v + 1]) for v in range(min(count, 3)) for i in range(2)] + [packet('E', 0, count, min(count, 3)), packet('E', 1)]


class UnifiedTests(unittest.TestCase):
    def test_frontend_thread_selection_normalization(self):
        self.assertEqual(frontend_thread([1, 2, 1], [16, 4, 2]), 97)
        self.assertEqual(frontend_thread(97, [16, 4, 2]), 97)
        self.assertIsNone(frontend_thread(None, [16, 4, 2]))
        for choice in (True, -1, 128, [16, 0, 0], [0, 4, 0], [0, 0, 2], [1, 2]):
            with self.assertRaises(ValueError):
                frontend_thread(choice, [16, 4, 2])

    def test_keyword_loop_and_original_evaluation_once(self):
        source = SOURCE.replace('T.serial(2, 8, step=2)', 'T.serial(start(), stop(), step=step())')
        points = prepare(source, configuration(source))
        text = inject(source, points)
        for call in ('start()', 'stop()', 'step()'):
            self.assertEqual(text.count(call), 1)
        self.assertIn('_kw0', points[0]['scopes'][0]['ordinal'])
        ast.parse(text)

    def test_pipeline_keyword_stop_and_stage_preserved(self):
        source = SOURCE.replace('T.serial(2, 8, step=2)', 'T.Pipelined(2, stop=8, num_stages=3)')
        p = prepare(source, configuration(source))[0]
        self.assertIn('_kw0', p['scopes'][0]['ordinal'])
        self.assertEqual(p['scopes'][0]['step'], '1')
        self.assertIn('num_stages=', inject(source, [p]))

    def test_manual_pipeline_diagnostic(self):
        source = SOURCE.replace('T.serial(2, 8, step=2)', 'T.Pipelined(8, order=[0], stage=[0])')
        with self.assertRaisesRegex(Unsupported, 'manual Pipelined'):
            prepare(source, configuration(source))

    def test_dynamic_bounds_and_transfers_preserved(self):
        source = SOURCE.replace('T.serial(2, 8, step=2)', 'T.serial(x[0])').replace('                v =', '                if k == 1:\n                    continue\n                v =')
        p = prepare(source, configuration(source))
        self.assertEqual(p[0]['transfers'][0]['target'], 6)
        text = inject(source, p)
        self.assertEqual(text.count('x[0]'), 1)
        self.assertIn('continue', text)

    def test_conditional_return_diagnostic(self):
        source = SOURCE.replace('                v =', '                if k == 1:\n                    return\n                v =')
        with self.assertRaisesRegex(Unsupported, 'frontend construction'):
            prepare(source, configuration(source))

    def test_direct_element_before_read_allowed(self):
        cfg = configuration(); cfg['points'][0].update(buffer='x[k]', when='before')
        self.assertEqual(prepare(SOURCE, cfg)[0]['buffer'], 'x[k]')

    def test_loop_header_and_inner_point_share_rewrite(self):
        cfg = configuration()
        cfg['points'].append(dict(id='header', line=6, when='before', buffer='x', block=[0, 0, 0], loops=[]))
        points = prepare(SOURCE, cfg)
        self.assertEqual(points[1]['coordinate_count'], 0)
        ast.parse(inject(SOURCE, points))

    def test_unordered_exact_roundtrip(self):
        rows, coverage = parse('\n'.join(reversed(log())), [contract()])
        self.assertEqual(rows[0]['bits'], 9007199254740993)
        self.assertEqual(coverage['p']['capture_integrity'], 'complete')

    def test_zero_attempts_are_legitimate(self):
        _, coverage = parse('\n'.join(log(0)), [contract()])
        self.assertEqual(coverage['p']['point_execution'], 'no_selected_visit')

    def test_budget_truncation_distinct_from_missing_data(self):
        rows, coverage = parse('\n'.join(log(7)), [contract()])
        self.assertEqual(len(rows), 6)
        self.assertEqual(coverage['p']['capture_integrity'], 'truncated')
        with self.assertRaisesRegex(ValueError, 'missing/unexpected'):
            parse('\n'.join(log(7)[:4] + log(7)[5:]), [contract()])

    def test_missing_entire_thread_or_visit_fails(self):
        rows = log()
        for changed in ([r for r in rows if r not in (packet('R', 1), packet('E', 1))], [r for r in rows if not ('|D|' in r and '|4|0|2|0|Z' in r)]):
            with self.assertRaises(ValueError):
                parse('\n'.join(changed), [contract()])

    def test_duplicate_truncated_and_bad_counter(self):
        rows = log()
        for changed in (rows + [rows[0]], rows[:-1], rows[:-1] + [rows[-1][:-2]], rows[:-2] + [packet('E', 0, 3, 3), packet('E', 1)]):
            with self.assertRaises(ValueError):
                parse('\n'.join(changed), [contract()])

    def test_parallel_ordinal_and_word_width_checked(self):
        p = contract(); p['scopes'][0]['kind'] = 'parallel'
        with self.assertRaisesRegex(ValueError, 'Parallel'):
            parse('\n'.join(log()), [p])
        rows = log(); fields = rows[2].split('|'); fields[-2] = str(2**32); rows[2] = '|'.join(fields)
        with self.assertRaisesRegex(ValueError, 'word'):
            parse('\n'.join(rows), [contract()])

    def test_total_launch_budget_includes_inactive_roots(self):
        p = contract(); p.update(bound=False, capacity=0, budget=3, indices=[])
        q = dict(p, id='q')
        rows = log(0)
        with self.assertRaisesRegex(ValueError, 'budget'):
            parse('\n'.join(rows + [r.replace('|p|', '|q|') for r in rows]), [p, q])

    def test_lost_invalid_read_packet_still_fails_at_end(self):
        rows = [packet('E', 0, 1, 0, 2), packet('E', 1)]
        with self.assertRaisesRegex(ValueError, 'out-of-bounds'):
            parse('\n'.join(rows), [contract()])

    def test_raw_types(self):
        for dtype, bits, expected in [('uint64', 2**64 - 1, 2**64 - 1), ('int8', 255, -1), ('uint16', 65535, 65535), ('float64', 0x3ff0000000000000, 1.0)]:
            self.assertEqual(value(bits, dtype), expected)

    def test_reference_is_independent_and_integer_exact(self):
        records, _ = parse('\n'.join(log()), [contract()])
        samples = [dict(thread=0, visit=v, coordinates=[2 + v * 2], ordinals=[v + 1], index=i, value=9007199254740993 + i) for v in range(2) for i in range(2)]
        spec = dict(schema=3, key='execution', dtype='int64', samples=samples, atol=0, rtol=0)
        self.assertTrue(compare_samples(records, contract(), spec)[0]['matched'])
        samples[0]['value'] += 1
        self.assertFalse(compare_samples(records, contract(), spec)[0]['matched'])

    def test_launch_boundaries_are_exact(self):
        launches = [dict(launch=0, compile=0), dict(launch=1, compile=0)]
        lines = ['TLHOST3|0|B|0|Z', packet('R'), 'TLHOST3|0|E|0|Z', 'TLHOST3|1|B|0|Z', 'TLHOST3|1|E|0|Z']
        self.assertEqual(split_launches('\n'.join(lines), launches)[0], packet('R'))
        for changed in (lines[:-1], lines + [packet('R')], lines + [lines[-1]], [lines[3], *lines[:3], lines[4]]):
            with self.assertRaises(ValueError):
                split_launches('\n'.join(changed), launches)


if __name__ == '__main__':
    unittest.main()
