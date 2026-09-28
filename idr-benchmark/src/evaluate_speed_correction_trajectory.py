"""Replay held-out prepared trajectories using a causal IMU speed-correction model.

This is a narrow development gate: each prepared window starts with a trusted
GNSS position, speed and heading.  From then on, position is propagated solely
from phone IMU yaw and either held GNSS speed or learned IMU speed correction.
The reference trajectory is used only after replay to score errors.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_speed_model import SpeedTCN


def wrap(value): return np.arctan2(np.sin(value), np.cos(value))


def replay(imu, t, reference, initial_velocity, initial_rpy, model, mean, std):
    # In production the last fresh GNSS velocity supplies the outage-start
    # heading.  Do not substitute an attitude-derived yaw here: it confounds
    # mounting alignment with the speed-correction experiment.
    anchor = float(np.linalg.norm(initial_velocity)); heading = float(np.arctan2(initial_velocity[0], initial_velocity[1]))
    predicted_pos = np.zeros(2); held_pos = np.zeros(2); predicted = [predicted_pos.copy()]; held = [held_pos.copy()]
    for i in range(1, len(t)):
        dt = min(.25, max(.001, float(t[i]-t[i-1])))
        # Prepared input is body-frame [gyro xyz, specific-force xyz].
        heading = float(wrap(heading + imu[i, 2] * dt))
        speed = anchor
        if i >= 19:
            extra = np.array([anchor, t[i]], dtype=np.float32)
            features = np.concatenate((imu[i-19:i+1], np.repeat(extra[None, :], 20, axis=0)), axis=1)
            with torch.no_grad():
                correction = float(model(torch.from_numpy(((features-mean)/std)[None].astype(np.float32))).item())
            speed = max(0., anchor + correction)
        direction = np.array([np.sin(heading), np.cos(heading)])
        predicted_pos = predicted_pos + speed * direction * dt
        held_pos = held_pos + anchor * direction * dt
        predicted.append(predicted_pos.copy()); held.append(held_pos.copy())
    target = reference[:, :2] - reference[0, :2]
    model_error = np.linalg.norm(np.asarray(predicted)-target, axis=1)
    held_error = np.linalg.norm(np.asarray(held)-target, axis=1)
    distance = float(np.linalg.norm(np.diff(target, axis=0), axis=1).sum())
    return distance, float(model_error[-1]), float(held_error[-1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--model', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--split', choices=('train','validation'), default='validation')
    a = p.parse_args()
    if a.out.exists(): p.error('Use a new output filename')
    artifact = torch.load(a.model, map_location='cpu', weights_only=False)
    if artifact.get('window') != 20 or len(artifact.get('features', [])) != 8: raise ValueError('Expected speed-correction model')
    model = SpeedTCN(8); model.load_state_dict(artifact['state_dict']); model.eval()
    mean = np.asarray(artifact['mean'], np.float32); std = np.asarray(artifact['std'], np.float32)
    manifest = json.loads((a.data/'manifest.json').read_text()); rows = []
    for record in manifest['records']:
        if record['split'] != a.split: continue
        with np.load(a.data/record['file'], allow_pickle=False) as z:
            d, learned, hold = replay(z['imu'].astype(np.float32), z['t'].astype(float), z['reference_position'].astype(float), z['initial_velocity'].astype(float), z['initial_rpy'].astype(float), model, mean, std)
        if d >= 1: rows.append({'drive': record['drive'], 'distance_m': d, 'learned_endpoint_error_m': learned, 'gnss_hold_endpoint_error_m': hold, 'learned_drift_percent': 100*learned/d, 'gnss_hold_drift_percent': 100*hold/d})
    if not rows: raise ValueError('No usable replay windows')
    drives = sorted({r['drive'] for r in rows})
    average = lambda key: float(np.mean([np.mean([r[key] for r in rows if r['drive'] == d]) for d in drives]))
    report = {'scope': 'Prepared IO-VNBD replay; IMU yaw and speed-only propagation. No map matching or GNSS inside outage.', 'split': a.split,
              'macro_by_drive': {'learned_endpoint_error_m': average('learned_endpoint_error_m'), 'gnss_hold_endpoint_error_m': average('gnss_hold_endpoint_error_m'), 'learned_drift_percent': average('learned_drift_percent'), 'gnss_hold_drift_percent': average('gnss_hold_drift_percent')},
              'windows': len(rows), 'drives': drives, 'limitations': ['Provisional frame/time contracts.', 'Ten-second prepared windows only.', 'Does not establish PS compliance.'], 'records': rows}
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['macro_by_drive'], indent=2))


if __name__ == '__main__': main()
