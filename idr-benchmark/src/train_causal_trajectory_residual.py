"""Directly train a causal IMU trajectory residual on prepared IO-VNBD windows.

The candidate starts from the last trusted GNSS velocity/heading, consumes only
IMU samples and elapsed outage time, and is trained against the later reference
trajectory.  It is a development gate, not an Android deployment artifact.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('schema') != 'idr-trajectory-v1' or manifest.get('status') not in ('provisional', 'verified'):
        raise ValueError('Requires provisional or verified canonical trajectory windows')
    values = {'train': [], 'validation': []}
    for row in manifest['records']:
        path = root / row['file']
        if digest(path) != row['sha256']:
            raise ValueError('Prepared window hash mismatch: ' + row['file'])
        with np.load(path, allow_pickle=False) as data:
            t = data['t'].astype(np.float32)
            imu = data['imu'].astype(np.float32)
            position = data['reference_position'][:, :2].astype(np.float32)
            velocity = data['initial_velocity'].astype(np.float32)
        if t.ndim != 1 or imu.shape != (len(t), 6) or position.shape != (len(t), 2):
            raise ValueError('Invalid window shape: ' + row['file'])
        values[row['split']].append((row['drive'], t, imu, position, velocity))
    if not values['train'] or not values['validation']:
        raise ValueError('Both partitions are required')
    return manifest, values


def normalisation(rows):
    imu = np.concatenate([x[2] for x in rows])
    return imu.mean(0), np.maximum(imu.std(0), 1e-3)


class Windows(Dataset):
    def __init__(self, rows, mean, std):
        self.rows = rows; self.mean = mean; self.std = std
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        drive, t, imu, position, velocity = self.rows[index]
        # No reference enters the model input.  The first GNSS velocity and
        # elapsed outage time are available at a real GNSS-loss transition.
        features = np.concatenate(((imu-self.mean)/self.std,
                                   np.full((len(t), 1), np.linalg.norm(velocity), np.float32),
                                   t[:, None]), axis=1)
        return drive, torch.from_numpy(features), torch.from_numpy(t), torch.from_numpy(position), torch.from_numpy(velocity)


class CausalTrajectoryResidual(nn.Module):
    """A compact causal GRU with bounded physical correction outputs."""
    def __init__(self, hidden=48):
        super().__init__()
        self.gru = nn.GRU(8, hidden, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2))
    def forward(self, x):
        state, _ = self.gru(x)
        # [speed residual m/s, yaw-rate residual rad/s]
        return self.head(state)


def propagate(outputs, t, features, initial_velocity):
    dt = torch.clamp(t[:, 1:] - t[:, :-1], .001, .25)
    anchor_speed = torch.linalg.vector_norm(initial_velocity, dim=1)
    heading0 = torch.atan2(initial_velocity[:, 0], initial_velocity[:, 1])
    # gyro Z is the current provisional yaw convention; the learned term is
    # bounded to prevent a network from inventing an implausible vehicle turn.
    yaw_rate = features[:, 1:, 2] + .50 * torch.tanh(outputs[:, 1:, 1])
    heading = heading0[:, None] + torch.cumsum(yaw_rate * dt, dim=1)
    speed = torch.relu(anchor_speed[:, None] + 8.0 * torch.tanh(outputs[:, 1:, 0]))
    step = torch.stack((speed * torch.sin(heading) * dt, speed * torch.cos(heading) * dt), dim=-1)
    zero = torch.zeros((len(t), 1, 2), dtype=step.dtype, device=step.device)
    return torch.cat((zero, torch.cumsum(step, dim=1)), dim=1)


def held_baseline(t, features, initial_velocity):
    return propagate(torch.zeros((len(t), len(t[0]), 2), dtype=features.dtype, device=features.device), t, features, initial_velocity)


def evaluate(model, dataset, device):
    model.eval(); rows = []
    with torch.no_grad():
        for index in range(len(dataset)):
            drive, features, t, target, velocity = dataset[index]
            features, t, target, velocity = (v.unsqueeze(0).to(device) for v in (features, t, target, velocity))
            predicted = propagate(model(features), t, features, velocity)[0].cpu().numpy()
            held = held_baseline(t, features, velocity)[0].cpu().numpy()
            truth = target[0].cpu().numpy()
            distance = np.linalg.norm(np.diff(truth, axis=0), axis=1).sum()
            rows.append(dict(drive=drive, distance_m=float(distance),
                             learned_endpoint_error_m=float(np.linalg.norm(predicted[-1]-truth[-1])),
                             hold_endpoint_error_m=float(np.linalg.norm(held[-1]-truth[-1]))))
    drives = sorted({r['drive'] for r in rows})
    def macro(key): return float(np.mean([np.mean([r[key] for r in rows if r['drive'] == d]) for d in drives]))
    return dict(macro_drive_endpoint_error_m=macro('learned_endpoint_error_m'),
                macro_drive_hold_endpoint_error_m=macro('hold_endpoint_error_m'), records=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=80); parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--lr', type=float, default=3e-4); parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu'); args = parser.parse_args()
    if args.out.exists(): raise ValueError('Use a new output directory')
    if args.device == 'mps' and not torch.backends.mps.is_available(): raise ValueError('MPS unavailable')
    manifest, rows = load(args.data); mean, std = normalisation(rows['train'])
    torch.manual_seed(26168); random.seed(26168); torch.set_num_threads(4)
    device = torch.device(args.device); model = CausalTrajectoryResidual().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_data, valid_data = Windows(rows['train'], mean, std), Windows(rows['validation'], mean, std)
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    args.out.mkdir(parents=True); best = float('inf'); stale = 0; history = []
    print(f'parameters={sum(p.numel() for p in model.parameters())}, train_windows={len(train_data)}, validation_windows={len(valid_data)}', flush=True)
    for epoch in range(1, args.epochs+1):
        model.train(); losses = []
        for _, features, t, target, velocity in loader:
            features, t, target, velocity = (v.to(device) for v in (features, t, target, velocity))
            optimizer.zero_grad(); predicted = propagate(model(features), t, features, velocity)
            loss = nn.functional.smooth_l1_loss(predicted, target)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); losses.append(loss.item())
        report = evaluate(model, valid_data, device); score = report['macro_drive_endpoint_error_m']
        improved = score < best; best = min(best, score); stale = 0 if improved else stale + 1
        history.append(dict(epoch=epoch, train_huber=float(np.mean(losses)), validation=report))
        payload = dict(format='navrakshak-causal-trajectory-residual-v1', state_dict=model.state_dict(), mean=mean.tolist(), std=std.tolist(),
                       status=manifest['status'], parameters=sum(p.numel() for p in model.parameters()), epoch=epoch, validation=report,
                       limits=['Provisional IO-VNBD contracts.', 'Development validation only; no independent test.', 'No Android deployment without blackout-drift acceptance.'])
        torch.save(payload, args.out/'last.pt')
        if improved:
            torch.save(payload, args.out/'best.pt'); (args.out/'best-validation.json').write_text(json.dumps(report, indent=2)+'\n')
        (args.out/'history.json').write_text(json.dumps(history, indent=2)+'\n')
        print(f"epoch {epoch}: endpoint {score:.3f} m, GNSS-hold {report['macro_drive_hold_endpoint_error_m']:.3f} m", flush=True)
        if stale >= args.patience: print('Validation early stopping'); break


if __name__ == '__main__': main()
