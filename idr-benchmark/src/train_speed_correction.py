"""Train a causal, GNSS-anchored IMU speed-correction model on IO-VNBD.

During a GNSS blackout the last trusted GNSS velocity is available at its start.
This model receives that scalar anchor and elapsed outage time alongside a trailing
two-second phone-IMU window, and predicts the correction to the anchor.  Vehicle
reference positions are labels only.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from train_speed_model import SpeedTCN


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root, stride):
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('schema') != 'idr-trajectory-v1' or manifest.get('status') not in ('provisional', 'verified'):
        raise ValueError('Expected prepared IO-VNBD trajectory data')
    result = {'train': [], 'validation': []}
    for row in manifest['records']:
        if sha(root / row['file']) != row['sha256']: raise ValueError('Prepared-window hash mismatch')
        with np.load(root / row['file'], allow_pickle=False) as z:
            imu = z['imu'].astype(np.float32); t = z['t'].astype(float)
            p = z['reference_position'].astype(float); initial = z['initial_velocity'].astype(float)
        indexes = np.arange(19, len(imu) - 1, stride)
        speed = np.linalg.norm((p[indexes + 1] - p[indexes - 1]) /
                               (t[indexes + 1, None] - t[indexes - 1, None]), axis=1)
        valid = np.isfinite(imu).all(1) & np.isfinite(p).all(1)
        keep = np.array([valid[i-19:i+2].all() for i in indexes]) & np.isfinite(speed)
        if not keep.any(): continue
        indexes, speed = indexes[keep], speed[keep]
        anchor = float(np.linalg.norm(initial))
        # Extra features are physically available at blackout start, not labels:
        # last GNSS speed and seconds since blackout.
        extra = np.column_stack((np.full(len(indexes), anchor), t[indexes])).astype(np.float32)
        x = np.concatenate((np.stack([imu[i-19:i+1] for i in indexes]),
                            np.repeat(extra[:, None, :], 20, axis=1)), axis=2)
        result[row['split']].append((row['drive'], x, (speed-anchor).astype(np.float32), anchor))
    if not result['train'] or not result['validation']: raise ValueError('No usable split')
    return result


def normalise(chunks):
    values = np.concatenate([x.reshape(-1, 8) for _, x, _, _ in chunks])
    return values.mean(0).astype(np.float32), np.maximum(values.std(0), 1e-3).astype(np.float32)


def score(model, chunks, mean, std, device):
    rows = []; model.eval()
    with torch.no_grad():
        for drive, x, correction, anchor in chunks:
            predicted = model(torch.from_numpy((x-mean)/std).to(device)).cpu().numpy() + anchor
            target = correction + anchor
            rows.append((drive, float(np.abs(predicted-target).mean()), float(np.abs(anchor-target).mean())))
    drives = sorted({r[0] for r in rows})
    macro = lambda n: float(np.mean([np.mean([r[n] for r in rows if r[0] == d]) for d in drives]))
    return {'macro_mae_mps': macro(1), 'macro_gnss_hold_baseline_mae_mps': macro(2),
            'drives': [{'drive': d,
                        'mae_mps': float(np.mean([r[1] for r in rows if r[0] == d])),
                        'gnss_hold_baseline_mae_mps': float(np.mean([r[2] for r in rows if r[0] == d])),
                        'windows': sum(r[0] == d for r in rows)} for d in drives]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=60); p.add_argument('--patience', type=int, default=10)
    p.add_argument('--stride', type=int, default=5); p.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    a = p.parse_args()
    if min(a.epochs, a.patience, a.stride) < 1: p.error('epochs, patience, and stride must be positive')
    if a.device == 'mps' and not torch.backends.mps.is_available(): p.error('MPS unavailable')
    if a.out.exists(): raise ValueError('Use a new output folder')
    chunks = build(a.data, a.stride); mean, std = normalise(chunks['train'])
    x = np.concatenate([part for _, part, _, _ in chunks['train']]); y = np.concatenate([part for _, _, part, _ in chunks['train']])
    torch.manual_seed(26168); random.seed(26168); torch.set_num_threads(2); device = torch.device(a.device)
    model = SpeedTCN(8).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.from_numpy(((x-mean)/std).astype(np.float32)), torch.from_numpy(y)), batch_size=512, shuffle=True)
    a.out.mkdir(parents=True); best = float('inf'); stale = 0; history = []
    for epoch in range(1, a.epochs+1):
        model.train(); losses = []
        for bx, by in loader:
            optimizer.zero_grad(); loss = torch.nn.functional.smooth_l1_loss(model(bx.to(device)), by.to(device)); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); losses.append(loss.item())
        validation = score(model, chunks['validation'], mean, std, device); value = validation['macro_mae_mps']
        improved = value < best; best = value if improved else best; stale = 0 if improved else stale + 1
        payload = {'state_dict': model.state_dict(), 'mean': mean.tolist(), 'std': std.tolist(), 'window': 20,
                   'features': ['gyro_x','gyro_y','gyro_z','specific_force_x','specific_force_y','specific_force_z','last_gnss_speed_mps','outage_elapsed_s'],
                   'target': 'correction to last GNSS speed m/s', 'epoch': epoch, 'validation': validation}
        if improved:
            torch.save(payload, a.out/'best.pt'); (a.out/'best-validation.json').write_text(json.dumps(validation, indent=2)+'\n')
        history.append({'epoch': epoch, 'train_huber': float(np.mean(losses)), 'validation': validation})
        (a.out/'history.json').write_text(json.dumps(history, indent=2)+'\n')
        print(f"epoch {epoch}: validation MAE {value:.4f} m/s, GNSS-hold {validation['macro_gnss_hold_baseline_mae_mps']:.4f}", flush=True)
        if stale >= a.patience: print('Validation early stopping'); break


if __name__ == '__main__': main()
