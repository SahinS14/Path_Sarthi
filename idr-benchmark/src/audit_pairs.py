"""Check paired rows and clocks before starting expensive training."""
import argparse
import json
from pathlib import Path

import numpy as np

from io_vnbd import find_pairs, _read, _select, PHONE_COLUMNS, VEHICLE_COLUMNS


def audit(pair):
    phone = _select(_read(pair.phone_csv), PHONE_COLUMNS)
    vehicle = _select(_read(pair.vehicle_csv), VEHICLE_COLUMNS)
    result = dict(drive=pair.name, phone_rows=len(phone), vehicle_rows=len(vehicle),
                  unequal_rows=len(phone) != len(vehicle))
    for name, frame, clock, scale in [('phone', phone, 'phone_time_ms', .001),
                                       ('vehicle', vehicle, 'vehicle_time_s', 1.)]:
        t = frame[clock].to_numpy() * scale
        dt = np.diff(t)
        result[name] = dict(nonfinite_values=int((~np.isfinite(frame.to_numpy())).sum()),
                            nonincreasing_intervals=int((dt <= 0).sum()),
                            gaps_over_150ms=int((dt > .15).sum()),
                            median_dt_s=float(np.nanmedian(dt)),
                            duration_s=float(t[-1] - t[0]))
    # Compare elapsed clocks; this diagnoses rate/offset drift, not absolute alignment.
    n = min(len(phone), len(vehicle))
    tp = phone.phone_time_ms.to_numpy()[:n] * .001
    tv = vehicle.vehicle_time_s.to_numpy()[:n]
    result['relative_clock_disagreement_max_s'] = float(np.nanmax(np.abs((tp-tp[0])-(tv-tv[0]))))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    pairs = sorted(find_pairs(args.dataset), key=lambda p: p.name)
    if not pairs:
        parser.error('No paired drives found')
    results = []
    for pair in pairs:
        try:
            results.append(audit(pair))
        except Exception as error:
            results.append(dict(drive=pair.name, error=str(error)))
    report = dict(pair_count=len(pairs), drives=results,
                  note='Clock agreement does not prove trajectory/reference alignment.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(pairs=len(pairs), errors=sum('error' in r for r in results),
                         unequal_rows=sum(r.get('unequal_rows', False) for r in results),
                         clock_disagreement_over_1s=sum(r.get('relative_clock_disagreement_max_s', 0)>1 for r in results)), indent=2))
