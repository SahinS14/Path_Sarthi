"""Resumable CPU trajectory training for the causal covariance head.

Consumes explicit, verified canonical windows, not the legacy speed-model data.
Synthetic manifests are supported only for pipeline self-tests, never acceptance.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch_filter_compat import compatible_filter
from reference_adapter import run_reference

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'engine/src'))
from causal_covariance import CausalCovariance


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def checkpoint(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary); temporary.replace(path)


def load_data(root):
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest.get('schema') != 'idr-trajectory-v1' or manifest.get('status') not in ('verified', 'provisional', 'synthetic'):
        raise ValueError('Requires canonical trajectory data with an explicit status')
    records = manifest['records']
    if not isinstance(manifest.get('sample_hz'), (int, float)) or not 0 < manifest['sample_hz'] <= 200:
        raise ValueError('Invalid canonical sample rate')
    if not records or {r['split'] for r in records} != {'train', 'validation'}:
        raise ValueError('Nonempty training and validation partitions required')
    groups = {}; seen = {}; windows = []
    for record in records:
        group = record['session_group']; split = record['split']
        if group in groups and groups[group] != split:
            raise ValueError('Session leakage across partitions')
        groups[group] = split
        path = (root/record['file']).resolve()
        if root.resolve() not in path.parents:
            raise ValueError('Window path escapes prepared directory')
        if digest(path) != record['sha256']:
            raise ValueError('Window fingerprint changed: '+record['file'])
        if record['sha256'] in seen and seen[record['sha256']] != split:
            raise ValueError('Duplicate window content across partitions')
        seen[record['sha256']] = split
        with np.load(path, allow_pickle=False) as archive:
            window = {k: archive[k].astype(np.float64) for k in ('t', 'imu', 'reference_position', 'initial_velocity', 'initial_rpy')}
        n = len(window['t'])
        shapes = dict(t=(n,), imu=(n, 6), reference_position=(n, 3), initial_velocity=(3,), initial_rpy=(3,))
        if n < 2 or any(window[k].shape != shape or not np.isfinite(window[k]).all() for k, shape in shapes.items()):
            raise ValueError('Invalid window shapes or nonfinite values')
        dt = np.diff(window['t'])
        if np.any(dt <= 0) or not np.allclose(dt, 1/manifest['sample_hz'], atol=.01, rtol=0):
            raise ValueError('Unexpected sample rate or gap')
        windows.append((record, window))
    return manifest, windows


def normalization(windows):
    values = np.concatenate([w['imu'] for r, w in windows if r['split'] == 'train'])
    return values.mean(0), np.maximum(values.std(0), 1e-3)


def make_filter(source):
    sys.path.insert(0, str(source.resolve()/'src'))
    from utils_torch_filter import TORCHIEKF
    f = compatible_filter(TORCHIEKF)()
    f.g = torch.as_tensor(f.g, dtype=torch.float64)
    for p in f.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        for p in f.initprocesscov_net.parameters():
            p.zero_()
    f.set_Q()
    return f


def trajectory(f, model, window, mean, std):
    u = torch.from_numpy(window['imu'])
    normalized = ((u-mean)/std).T.unsqueeze(0)
    covariance = model(normalized)[0]
    velocity = torch.zeros((len(u), 3), dtype=torch.float64)
    velocity[0] = torch.from_numpy(window['initial_velocity'])
    _, _, position, *_ = f.run(torch.from_numpy(window['t']-window['t'][0]), u, covariance,
                               velocity, torch.zeros_like(velocity), len(u),
                               torch.from_numpy(window['initial_rpy']))
    return position


def metrics(position, window):
    truth = window['reference_position']-window['reference_position'][0]
    error = np.linalg.norm((position-truth)[:, :2], axis=1)
    distance = float(np.linalg.norm(np.diff(truth[:, :2], axis=0), axis=1).sum())
    return dict(endpoint_error_m=float(error[-1]), rmse_m=float(np.sqrt(np.mean(error**2))),
                distance_m=distance, drift_fraction=float(error[-1]/distance) if distance >= 1 else None)


def evaluate(model, windows, mean, std, source, max_windows=None):
    results = []; model.eval()
    selected = [(record, window) for record, window in windows if record['split'] == 'validation']
    if max_windows is not None:
        selected = selected[:max_windows]
    for record, window in selected:
        with torch.no_grad():
            cov = model(torch.from_numpy(((window['imu']-mean.numpy())/std.numpy()).T).unsqueeze(0))[0].numpy()
        shared = dict(initial_velocity=window['initial_velocity'], initial_rpy=window['initial_rpy'], source=source)
        adaptive = run_reference(window['t'], window['imu'], covariance=cov, **shared)['position']
        fixed = run_reference(window['t'], window['imu'], **shared)['position']
        constant = (window['t']-window['t'][0])[:, None]*window['initial_velocity']
        results.append(dict(drive=record['drive'], window=record['file'],
                            adaptive=metrics(adaptive, window), fixed=metrics(fixed, window),
                            constant_velocity=metrics(constant, window)))
    # Equal weight per drive, so many overlapping windows cannot dominate selection.
    drives = sorted({r['drive'] for r in results})
    scores = {mode: float(np.mean([np.mean([r[mode]['rmse_m'] for r in results if r['drive']==drive])
                                  for drive in drives])) for mode in ('adaptive', 'fixed', 'constant_velocity')}
    return dict(selection_metric='macro_drive_horizontal_rmse_m', scores=scores, windows=results)


def fingerprints(source):
    paths = [Path(__file__), PROJECT/'engine/src/causal_covariance.py',
             Path(__file__).with_name('torch_filter_compat.py'), Path(__file__).with_name('reference_adapter.py'),
             source/'src/utils_numpy_filter.py', source/'src/utils_torch_filter.py']
    return {str(p.resolve()): digest(p) for p in paths}


def train(args):
    manifest, windows = load_data(args.data)
    torch.set_num_threads(2); torch.manual_seed(26168); random.seed(26168)
    model = CausalCovariance().double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    f = make_filter(args.source)
    mean, std = [torch.from_numpy(x) for x in normalization(windows)]
    max_train = getattr(args, 'max_train_windows', None)
    max_validation = getattr(args, 'max_validation_windows', None)
    identity = dict(data=digest(args.data/'manifest.json'), code=fingerprints(args.source),
                    torch=torch.__version__, lr=args.lr, patience=args.patience,
                    max_train_windows=max_train, max_validation_windows=max_validation)
    start = 0; best = float('inf'); stale = 0; history = []
    if args.resume:
        saved = torch.load(args.out/'last.pt', weights_only=False, map_location='cpu')
        if saved['identity'] != identity:
            raise ValueError('Resume rejected: code, data, runtime or optimizer configuration changed')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        start, best, stale, history = saved['epoch'], saved['best'], saved['stale'], saved['history']
        random.setstate(saved['python_rng']); torch.set_rng_state(saved['torch_rng'])
    elif args.out.exists():
        raise ValueError('Use a new run directory or --resume')
    if start >= args.epochs or stale >= args.patience:
        print('Run already reached its epoch limit or early-stopping condition'); return
    args.out.mkdir(parents=True, exist_ok=True)
    train_windows = [(r, w) for r, w in windows if r['split'] == 'train']
    if max_train is not None:
        train_windows = train_windows[:max_train]
    for epoch in range(start+1, args.epochs+1):
        tick = time.perf_counter(); model.train(); losses = []
        order = list(range(len(train_windows))); random.shuffle(order)
        for index in order:
            _, window = train_windows[index]
            optimizer.zero_grad()
            position = trajectory(f, model, window, mean, std)
            truth = torch.from_numpy(window['reference_position']-window['reference_position'][0])
            loss = torch.nn.functional.smooth_l1_loss(position[:, :2], truth[:, :2])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite trajectory loss; last completed epoch remains resumable')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); losses.append(float(loss.detach()))
        report = evaluate(model, windows, mean, std, args.source, max_validation)
        score = report['scores']['adaptive']; improved = score < best
        stale = 0 if improved else stale+1
        if improved: best = score
        history.append(dict(epoch=epoch, train_huber=float(np.mean(losses)), validation=report['scores'],
                            seconds=time.perf_counter()-tick))
        payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                       best=best, stale=stale, history=history, identity=identity,
                       mean=mean, std=std, sample_hz=manifest['sample_hz'], dataset_status=manifest['status'],
                       python_rng=random.getstate(), torch_rng=torch.get_rng_state())
        if improved:
            checkpoint(args.out/'best.pt', payload)
            atomic_json(args.out/'best-validation.json', report)
        checkpoint(args.out/'last.pt', payload)
        atomic_json(args.out/'history.json', history)
        print(json.dumps(history[-1]), flush=True)
        if stale >= args.patience:
            print('Validation early stopping'); break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=PROJECT/'work/reference/ai-imu-dr')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-train-windows', type=int, help='Deterministic pilot limit; omit for full training')
    parser.add_argument('--max-validation-windows', type=int, help='Deterministic pilot limit; omit for full validation')
    parser.add_argument('--evaluate', type=Path, help='Evaluate a saved checkpoint; do not train')
    parser.add_argument('--check-only', action='store_true', help='Validate prepared data without training')
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('epochs, patience and learning rate must be positive')
    if any(v is not None and v < 1 for v in (args.max_train_windows, args.max_validation_windows)):
        parser.error('window limits must be positive')
    if args.evaluate and (args.resume or args.check_only):
        parser.error('Evaluation cannot be combined with resume or check-only')
    if args.check_only:
        manifest, windows = load_data(args.data)
        print(json.dumps(dict(status=manifest['status'], windows=len(windows),
                              partitions={s: sum(r['split']==s for r, _ in windows) for s in ('train', 'validation')},
                              scope='Data-schema check only, not accuracy or production approval'), indent=2))
    elif args.evaluate:
        manifest, windows = load_data(args.data)
        saved = torch.load(args.evaluate, weights_only=False, map_location='cpu')
        if saved['identity']['data'] != digest(args.data/'manifest.json') or saved['identity']['code'] != fingerprints(args.source):
            raise ValueError('Evaluation code or dataset does not match checkpoint')
        model = CausalCovariance().double(); model.load_state_dict(saved['model'])
        report = evaluate(model, windows, saved['mean'], saved['std'], args.source, args.max_validation_windows)
        report.update(dataset_status=manifest['status'], scope='Development validation, not pristine test or PS certification')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
        print(json.dumps(report['scores'], indent=2))
    else:
        train(args)


if __name__ == '__main__':
    main()
