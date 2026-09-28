import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'idr-benchmark' / 'src'))
from speed_tcn_contract import FEATURES, SAMPLE_HZ, WINDOW_SAMPLES, android_features, io_vnbd_features


class SpeedTcnContractTest(unittest.TestCase):
    def test_android_legacy_order_is_explicit(self):
        row = android_features([1, 2, 3], [4, 5, 6], [7, 8, 9])
        np.testing.assert_array_equal(row, np.array([1,2,3,4,5,6,9,8,7], dtype=np.float32))

    def test_io_feature_order_matches_export_contract(self):
        source = {name: index for index, name in enumerate(FEATURES)}
        np.testing.assert_array_equal(io_vnbd_features(source), np.arange(9, dtype=np.float32))
        self.assertEqual((SAMPLE_HZ, WINDOW_SAMPLES), (10, 20))

    def test_nonfinite_sensor_value_is_rejected(self):
        with self.assertRaises(ValueError): android_features([0, 0, np.nan], [0, 0, 9.8], [0, 0, 0])


if __name__ == '__main__': unittest.main()
