"""Train a gravity-removed recurrent speed-increment candidate on IO-VNBD.

This is a comparison experiment, deliberately kept separate from the deployed
TCN.  It uses the exact session split recorded in a prepared-v1 manifest but
rebuilds features from the original paired CSV files.  Therefore it must not
be described as a pristine final test or as phone-orientation independent.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from io_vnbd import find_pairs, load_drive

FEATURES = (
    'linear_acc_x_mps2', 'linear_acc_y_mps2', 'linear_acc_z_mps2',
    'gyro_yaw_rps', 'gyro_pitch_rps', 'gyro_roll_rps',
)
WINDOW = 20


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def vehicle_motion_features(frame):
    """Specific force plus gyro in IO-VNBD's documented mounted-phone axes."""
    acceleration = frame[['acc_x_mps2', 'acc_y_mps2', 'acc_z_mps2']].to_numpy(float)
    gravity = frame[['gravity_x_mps2', 'gravity_y_mps2', 'gravity_z_mps2']].to_numpy(float)
    gyro = frame[['gyro_yaw_rps', 'gyro_pitch_rps', 'gyro_roll_rps']].to_numpy(float)
    return np.column_stack((acceleration - gravity, gyro))


def make_windows(frame, mean=None, std=None, stride=5):
    values = vehicle_motion_features(frame)
    speed = frame['phone_speed_kmh'].to_numpy(float) / 3.6
    elapsed = frame['elapsed_s'].to_numpy(float)
    finite = np.isfinite(values).all(axis=1) & np.isfinite(speed) & np.isfinite(elapsed)
    result_x, result_y = [], []
    for endpoint in range(WINDOW - 1, len(frame), stride):
        start = endpoint - WINDOW + 1
        times = elapsed[start:endpoint + 1]
        if (not finite[start:endpoint + 1].all() or np.any(np.diff(times) <= 0)
                or np.any(np.diff(times) > 0.25)):
            continue
        x = values[start:endpoint + 1]
        if mean is not None:
            x = (x - mean) / std
        result_x.append(x.astype('float32'))
        result_y.append(np.float32(speed[endpoint] - speed[start]))
    if not result_x:
        return np.empty((0, WINDOW, len(FEATURES)), np.float32), np.empty((0,), np.float32)
    return np.stack(result_x), np.asarray(result_y, dtype=np.float32)


class VehicleSpeedGRU(nn.Module):
    def __init__(self, features=len(FEATURES), hidden=32):
        super().__init__()
        self.gru = nn.GRU(features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x):
        output, _ = self.gru(x)
        return self.head(output[:, -1]).squeeze(1)


def parameter_count(model):
    return sum(value.numel() for value in model.parameters())


def source_records(raw_root, manifest):
    pairs = {pair.name: pair for pair in find_pairs(raw_root)}
    result = []
    for record in manifest['records']:
        pair = pairs.get(record['drive'])
        if pair is None:
            raise ValueError('Raw pair absent: ' + record['drive'])
        if digest(pair.phone_csv) != record['phone_sha256'] or digest(pair.vehicle_csv) != record['vehicle_sha256']:
            raise ValueError('Raw input hash mismatch: ' + record['drive'])
        result.append((record, pair))
    return result


def build_data(raw_root, split_manifest):
    manifest = json.loads((split_manifest / 'manifest.json').read_text())
    records = source_records(raw_root, manifest)
    training_values = []
    loaded = []
    for record, pair in records:
        frame = load_drive(pair)
        features = vehicle_motion_features(frame)
        if record['split'] == 'train':
            training_values.append(features[np.isfinite(features).all(axis=1)])
        loaded.append((record, frame))
    raw_train = np.concatenate(training_values, axis=0)
    mean = raw_train.mean(axis=0)
    # A floor avoids turning near-static channels into enormous normalized values.
    std = np.maximum(raw_train.std(axis=0), 0.05)
    partitions = {split: [] for split in ('train', 'validation', 'test')}
    for record, frame in loaded:
        x, y = make_windows(frame, mean, std)
        if len(y):
            partitions[record['split']].append((record['drive'], x, y))
    return manifest, mean, std, partitions


def score(model, records, device):
    model.eval(); report = []
    with torch.no_grad():
        for drive, x, y in records:
            prediction = []
            for start in range(0, len(y), 512):
                prediction.append(model(torch.from_numpy(x[start:start + 512]).to(device)).cpu().numpy())
            error = np.concatenate(prediction) - y
            report.append(dict(drive=drive, count=int(len(y)), mae=float(np.mean(np.abs(error))),
                               rmse=float(np.sqrt(np.mean(error ** 2))),
                               zero_change_mae=float(np.mean(np.abs(y)))))
    return dict(macro_mae=float(np.mean([r['mae'] for r in report])),
                macro_zero_change_mae=float(np.mean([r['zero_change_mae'] for r in report])),
                drives=report)


def save(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def train(args):
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS requested but unavailable')
    if args.out.exists() and not args.resume:
        raise ValueError('Use a new output directory or --resume')
    torch.manual_seed(26168); random.seed(26168); torch.set_num_threads(4)
    manifest, mean, std, partitions = build_data(args.raw, args.split)
    fingerprint = digest(args.split / 'manifest.json')
    device = torch.device(args.device)
    model = VehicleSpeedGRU().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    start = 0; best = float('inf'); stale = 0; history = []
    if args.resume:
        saved = torch.load(args.out / 'last.pt', map_location='cpu', weights_only=False)
        if saved['split_manifest_sha256'] != fingerprint:
            raise ValueError('Checkpoint split manifest mismatch')
        model.load_state_dict(saved['state_dict']); optimizer.load_state_dict(saved['optimizer'])
        start, best, stale, history = saved['epoch'], saved['best'], saved['stale'], saved['history']
    args.out.mkdir(parents=True, exist_ok=True)
    x_train = np.concatenate([x for _, x, _ in partitions['train']])
    y_train = np.concatenate([y for _, _, y in partitions['train']])
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                        batch_size=args.batch_size, shuffle=True)
    print(f'GRU parameters: {parameter_count(model)}; train windows: {len(y_train)}', flush=True)
    for epoch in range(start + 1, args.epochs + 1):
        model.train(); loss_sum = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            loss = nn.functional.smooth_l1_loss(model(x.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            loss_sum += loss.item() * len(y)
        validation = score(model, partitions['validation'], device)
        improved = validation['macro_mae'] < best
        stale = 0 if improved else stale + 1
        if improved: best = validation['macro_mae']
        history.append(dict(epoch=epoch, train_huber=loss_sum / len(y_train), validation=validation))
        payload = dict(format='navrakshak-vehicle-speed-gru-v1', state_dict=model.state_dict(),
                       optimizer=optimizer.state_dict(), epoch=epoch, best=best, stale=stale,
                       history=history, split_manifest_sha256=fingerprint, features=FEATURES,
                       window=WINDOW, mean=mean.tolist(), std=std.tolist(), hidden=32,
                       target='speed_delta_mps_over_1.9s', limitations=[
                           'Uses IO-VNBD mounted-phone axes, not arbitrary phone orientation.',
                           'Historical development split, not an independent final test.',
                           'Speed-increment accuracy does not establish trajectory-drift accuracy.'])
        save(args.out / 'last.pt', payload)
        if improved: save(args.out / 'best.pt', payload)
        (args.out / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        print(f"epoch {epoch}: validation MAE {validation['macro_mae']:.4f}, zero-change {validation['macro_zero_change_mae']:.4f}", flush=True)
        if stale >= args.patience:
            print('Validation early stopping'); break
    best_payload = torch.load(args.out / 'best.pt', map_location='cpu', weights_only=False)
    # A run may train on MPS, but the final saved-model report is deliberately
    # generated on CPU so it can be reproduced on machines without Metal.
    model = VehicleSpeedGRU()
    model.load_state_dict(best_payload['state_dict'])
    test = score(model, partitions['test'], torch.device('cpu'))
    result = dict(parameters=parameter_count(model), test=test,
                  acceptance=('PASS' if test['macro_mae'] < test['macro_zero_change_mae'] else 'REJECT'),
                  scope='Historical IO-VNBD speed-increment comparison only; not a production navigation acceptance result.')
    (args.out / 'test.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        parser.error('epochs, patience, and batch-size must be positive')
    train(args)
