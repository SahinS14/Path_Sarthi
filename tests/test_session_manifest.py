import sys
import unittest
import io
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'idr-benchmark/src'))
from session_manifest import assign_groups, recording_dates


def record(name, dates, digest=None):
    return dict(drive=name, dates=dates, phone_sha256=digest or name,
                vehicle_sha256='vehicle-'+name)


class SessionGroupsTest(unittest.TestCase):
    def test_dates_validate_all_rows(self):
        text = ' DATE (YYYY-MO-DD HH-MI-SS_SSS)\n2019-01-01 23:59:59:900\n2019-01-02 00:00:00:000\n'
        with patch.object(Path, 'open', return_value=io.StringIO(text)):
            self.assertEqual(recording_dates(Path('unused')), ['2019-01-01', '2019-01-02'])
        with patch.object(Path, 'open', return_value=io.StringIO(text + 'invalid\n')):
            with self.assertRaises(ValueError):
                recording_dates(Path('unused'))

    def test_same_day_and_midnight_transitivity(self):
        rows = [record('A', ['2019-01-01']), record('B', ['2019-01-01', '2019-01-02']),
                record('C', ['2019-01-02']), record('D', ['2019-01-03'])]
        assign_groups(rows)
        self.assertEqual(len({r['session_group'] for r in rows[:3]}), 1)
        self.assertNotEqual(rows[0]['session_group'], rows[3]['session_group'])

    def test_subdivisions_and_duplicates(self):
        rows = [record('Vw14a', ['1']), record('Vw14b', ['2'], 'shared'), record('Z', ['3'], 'shared')]
        assign_groups(rows)
        self.assertEqual(len({r['session_group'] for r in rows}), 1)

    def test_order_independence(self):
        rows = [record('A', ['1']), record('B', ['1']), record('C', ['2'])]
        expected = {r['drive']: r['session_group'] for r in assign_groups(rows)}
        self.assertEqual(expected, {r['drive']: r['session_group'] for r in assign_groups(rows[::-1])})
