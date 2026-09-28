import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'idr-benchmark/src'))
from train_speed_model import FEATURES, windows


class WindowIntegrityTests(unittest.TestCase):
    def frame(self):
        data = {key: np.ones(50) for key in FEATURES}
        data.update(elapsed_s=np.arange(50) / 10, ref_speed_kmh=np.arange(50) * 3.6)
        return pd.DataFrame(data)

    def run_windows(self, frame):
        return windows(frame, np.zeros(9), np.ones(9), 20, 1, 'delta')

    def test_endpoint_target(self):
        x, y, _, _ = self.run_windows(self.frame())
        self.assertEqual(x.shape, (31, 20, 9))
        np.testing.assert_allclose(y, 19, atol=1e-5)

    def test_invalid_row_does_not_join_segments(self):
        frame = self.frame()
        frame.loc[25, FEATURES[0]] = np.nan
        x, _, _, _ = self.run_windows(frame)
        self.assertEqual(len(x), 11)
        self.assertTrue(np.isfinite(x).all())

    def test_time_gap_does_not_join_segments(self):
        frame = self.frame()
        frame.loc[25:, 'elapsed_s'] += 1
        x, _, _, _ = self.run_windows(frame)
        self.assertEqual(len(x), 12)

    def test_short_drive_is_empty(self):
        x, y, _, _ = self.run_windows(self.frame().iloc[:10])
        self.assertEqual(x.shape, (0, 20, 9))
        self.assertEqual(len(y), 0)


if __name__ == '__main__':
    unittest.main()
