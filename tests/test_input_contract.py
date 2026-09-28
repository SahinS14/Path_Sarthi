import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'idr-benchmark/src'))
from input_contract import align_labels, convert_imu, proper_rotation, speed_mps


class InputContractTest(unittest.TestCase):
    def test_no_reference_based_unit_guess(self):
        np.testing.assert_allclose(speed_mps([36], 'km/h'), [10])
        np.testing.assert_allclose(speed_mps([36], 'm/s'), [36])
        with self.assertRaises(ValueError):
            speed_mps([36], 'auto')

    def test_separate_sensor_frames(self):
        r = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        a, w = convert_imu([[1, 0, 9.81]], [[1, 0, 0]], acceleration_to_body=r,
                          gyro_to_body=np.eye(3), acceleration_kind='specific_force')
        np.testing.assert_allclose(a, [[0, 1, 9.81]])
        np.testing.assert_allclose(w, [[1, 0, 0]])
        with self.assertRaises(ValueError):
            proper_rotation(np.diag([1, 1, -1]))

    def test_linear_acceleration_restores_gravity(self):
        a, _ = convert_imu([[0, 0, 0]], [[0, 0, 0]], acceleration_to_body=np.eye(3),
                           gyro_to_body=np.eye(3), acceleration_kind='linear_acceleration',
                           gravity=[[0, 0, 9.81]])
        np.testing.assert_allclose(a, [[0, 0, 9.81]])

    def test_offset_gap_and_no_extrapolation(self):
        y, valid = align_labels([0, 1, 1.5, 2, 3, 4, 5], [0, 1, 3], [[0], [10], [30]],
                                reference_offset_s=1, max_gap_s=1)
        np.testing.assert_array_equal(valid, [False, True, True, True, False, True, False])
        np.testing.assert_allclose(y[valid, 0], [0, 5, 10, 30])
        self.assertTrue(np.isnan(y[~valid]).all())

    def test_duplicate_clock_rejected(self):
        with self.assertRaises(ValueError):
            align_labels([0, 1], [0, 0], [[1], [2]], reference_offset_s=0, max_gap_s=1)

    def test_nonfinite_labels_do_not_spread_across_exact_samples(self):
        y, valid = align_labels([0, .5, 1, 2], [0, 1, 2], [[1], [np.nan], [3]],
                                reference_offset_s=0, max_gap_s=1)
        np.testing.assert_array_equal(valid, [True, False, False, True])
