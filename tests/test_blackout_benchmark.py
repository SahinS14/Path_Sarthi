import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'idr-benchmark/src'))
from benchmark_blackouts import summarize


class BlackoutSummaryTest(unittest.TestCase):
    def test_groups_by_duration_and_counts_threshold(self):
        rows=[dict(outage_seconds=10, reference_distance_m=20, filter_drift_percent=5, constant_velocity_drift_percent=15),
              dict(outage_seconds=10, reference_distance_m=30, filter_drift_percent=15, constant_velocity_drift_percent=5),
              dict(outage_seconds=30, reference_distance_m=.5, filter_drift_percent=None, constant_velocity_drift_percent=None)]
        summary=summarize(rows)
        self.assertEqual(summary['10']['cases'], 2)
        self.assertEqual(summary['10']['filter_under_10_percent'], 1)
        self.assertEqual(summary['30']['cases'], 0)
