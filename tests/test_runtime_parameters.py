"""Frontend arithmetic and logical storage reconstruction contracts."""
import tempfile
from pathlib import Path
import unittest

import tilelang
from tvm import tirx, arith
from tilelang_debugger.runtime.parameters import expression, evaluate
from tilelang_debugger.runtime.tensor_views import restore_storage


class ParameterTests(unittest.TestCase):
    def test_frontend_integer_semantics(self):
        n = tirx.Var('n', 'int64')
        two = tirx.IntImm('int64', 2)
        expressions = [tirx.Div(n, two), tirx.Mod(n, two), tirx.FloorDiv(n, two), tirx.FloorMod(n, two),
                       3 + tirx.Div(n, two)]
        for expr in expressions:
            for value in (-257, -3, 0, 257, 2**60 + 1):
                expected = arith.Analyzer().simplify(tirx.stmt_functor.substitute(expr, {n: tirx.IntImm('int64', value)}))
                self.assertEqual(evaluate(expression(expr), {'n': value}), int(expected), (expr, value))

    def test_narrowing_cast_against_independent_byte_interpretation(self):
        n = tirx.Var('n', 'int64')
        for value in (-257, -3, 0, 257, 2**60 + 1):
            raw = bytes([value & 255])
            for dtype in ('int8', 'uint8'):
                expected = int.from_bytes(raw, 'little', signed=dtype == 'int8')
                self.assertEqual(evaluate(expression(tirx.Cast(dtype, n)), {'n': value}), expected)
            self.assertEqual(evaluate(expression(tirx.Cast('bool', n)), {'n': value}), int(value != 0))

    def test_undefined_arithmetic_rejected(self):
        n = tirx.Var('n', 'int32')
        for expr, value in [(tirx.Div(n, -1), -(2**31)), (n + 1, 2**31 - 1), (tirx.Div(1, n), 0)]:
            with self.assertRaisesRegex(ValueError, 'overflow|division by zero'):
                evaluate(expression(expr), {'n': value})
        with self.assertRaises(ValueError):
            evaluate(expression(n), {'n': 2**31})

    def test_unsigned_wrap_and_cast(self):
        n = tirx.Var('n', 'uint32')
        self.assertEqual(evaluate(expression(n + 1), {'n': 2**32 - 1}), 0)
        with self.assertRaises(ValueError):
            evaluate(expression(n), {'n': -1})

    def test_overlapping_views_cross_dtype_and_conflicts(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / 'a.bin').write_bytes(bytes(range(8)))
            (folder / 'b.bin').write_bytes(bytes(range(4, 8)))
            a = dict(kind='tensor', dtype='int32', shape=[2], stride=[1], storage_offset=0, alias_group=0, storage_bytes=8, file='a.bin')
            b = dict(a, dtype='uint8', shape=[4], storage_offset=4, file='b.bin')
            self.assertEqual(restore_storage([a, b], folder)[0], bytes(range(8)))
            (folder / 'b.bin').write_bytes(b'xxxx')
            with self.assertRaisesRegex(ValueError, 'inconsistent bytes'):
                restore_storage([a, b], folder)


if __name__ == '__main__':
    unittest.main()
