import json
import math
import unittest

from tilelang_debugger.numerics import compare, coordinates, finite_tolerance, scalar_error


def data(values, shape=None, dtype="float32"):
    return dict(values=values, bits=[0]*len(values), shape=shape if shape is not None else [len(values)], dtype=dtype)


class NumericTests(unittest.TestCase):
    def test_reference_relative_boundary(self):
        self.assertTrue(scalar_error(1.125, 1.0, 0, 0.125)[0])
        self.assertFalse(scalar_error(math.nextafter(1.125, math.inf), 1.0, 0, 0.125)[0])
        self.assertFalse(scalar_error(2.0, 1.0, 0, 0.5)[0])

    def test_zero_special_values(self):
        s, rows = compare(data([0., -0., 1., math.nan, math.inf, -math.inf, math.inf]),
                          data([-0., 0., 0., math.nan, math.inf, math.inf, 1.]), 0, 0)
        self.assertEqual(s["mismatches"], 4)
        self.assertEqual([r["matched"] for r in rows], [True, True, False, False, True, False, False])
        self.assertEqual(rows[2]["relative_error"], "+Inf")
        self.assertIsNone(rows[3]["abs_error"])
        self.assertEqual(s["special_values"]["actual"], dict(nan=1, posinf=2, neginf=1))
        json.loads(json.dumps([s, rows], allow_nan=False))

    def test_int32_extremes_and_coordinates(self):
        s, rows = compare(data([-2147483648, 2147483647, 1, 2], [2, 2], "int32"),
                          data([2147483647, 2147483647, 1, 3], [2, 2], "int32"), 0, 0)
        self.assertEqual(s["max_abs_error"], 4294967295)
        self.assertEqual([r["coordinates"] for r in s["first_mismatches"]], [[0, 0], [1, 1]])
        self.assertEqual(coordinates(5, [2, 3]), [1, 2])
        self.assertEqual(coordinates(0, []), [])

    def test_invalid_tolerances_shape_and_dtype(self):
        for v in (-1, math.inf, math.nan, True, "0"):
            with self.subTest(v=v), self.assertRaises(ValueError):
                finite_tolerance(v)
        for a, e, atol in ((data([1], [1]), data([1], [1,1]), 0),
                            (data([1], dtype="int32"), data([1]), 0),
                            (data([1], dtype="int32"), data([1], dtype="int32"), 1),
                            (data([1]), data([1], dtype="int32"), 0)):
            with self.assertRaises(ValueError):
                compare(a, e, atol, 0)

    def test_failure_preview_is_bounded_not_count(self):
        s, rows = compare(data(list(range(30))), data([-1]*30), 0, 0)
        self.assertEqual(s["mismatches"], 30)
        self.assertEqual(len(s["first_mismatches"]), 20)
        self.assertEqual(len(rows), 30)


if __name__ == "__main__":
    unittest.main()
