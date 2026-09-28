import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'idr-benchmark/src'))
from reference_adapter import run_reference


class ReferenceAdapterValidationTest(unittest.TestCase):
    def test_duplicate_and_large_gap_rejected(self):
        for time in ([0, 0], [0, 1]):
            with self.assertRaises(ValueError):
                run_reference(time, np.zeros((2, 6)), [0]*3, [0]*3)

    def test_nonfinite_input_rejected(self):
        imu = np.zeros((2, 6)); imu[1, 0] = np.nan
        with self.assertRaises(ValueError):
            run_reference([0, .1], imu, [0]*3, [0]*3)

    def test_reference_trajectory_cannot_be_passed_as_seed(self):
        with self.assertRaises(ValueError):
            run_reference([0, .1], np.zeros((2, 6)), np.zeros((2, 3)), [0]*3)
