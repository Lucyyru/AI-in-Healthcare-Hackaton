import unittest
from wheelchair_check import compare_width


class WheelchairCheckTests(unittest.TestCase):
    def test_strict_comparison(self):
        for channel, wheelchair, expected in [
            (1.2, 0.65, "Can pass!"),
            (1.2, 1.2, "Cannot pass"),
            (1.2, 1.3, "Cannot pass"),
        ]:
            with self.subTest(channel=channel, wheelchair=wheelchair):
                self.assertEqual(compare_width(channel, wheelchair), expected)

    def test_missing_or_invalid_measurement_is_unknown(self):
        for channel in (None, float("nan"), float("inf"), 0):
            self.assertEqual(compare_width(channel, 0.65), "Unable to determine: no valid passage width.")

    def test_invalid_wheelchair_width(self):
        for wheelchair in (0, -0.5, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                compare_width(1.2, wheelchair)
