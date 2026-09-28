"""Phase-3: train a small IMU temporal model for vehicle-speed estimation.

Split is by drive, never random rows, preventing adjacent samples from leaking
between training and evaluation.  The model uses only phone IMU-derived inputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from io_vnbd import find_pairs, load_drive
from speed_tcn_contract import FEATURES as CONTRACT_FEATURES


FEATURES = list(CONTRACT_FEATURES)
TRAIN_DRIVES = ("Vw14b", "Vta01a", "Vta29", "Vtb01")
VALIDATION_DRIVE = "Vw14c"
TEST_DRIVE = "Vta10"


class SpeedTCN(nn.Module):
    """A compact depthwise temporal CNN operating on a trailing IMU window."""
    def __init__(self, features: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(features, 24, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(24, 24, kernel_size=5, padding=2, groups=24), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(24, 1),
        )

    def forward(self, inputs):
        return self.network(inputs.transpose(1, 2)).squeeze(1)


def pair_by_name(root: Path) -> dict[str, object]:
    return {pair.name: pair for pair in find_pairs(root)}


def windows(frame, mean: np.ndarray | None, std: np.ndarray | None, window: int, stride: int, target_mode: str):
    values = frame[FEATURES].to_numpy(dtype=np.float32)
    targets = (frame.ref_speed_kmh.to_numpy(dtype=np.float32) / 3.6)
    valid = np.isfinite(values).all(axis=1) & np.isfinite(targets)
    if window < 2 or stride < 1:
        raise ValueError("window must be >= 2 and stride >= 1")
    if mean is None:
        if not valid.any():
            raise ValueError("No finite training samples")
        mean = values[valid].mean(axis=0); std = values[valid].std(axis=0).clip(1e-4)
    values = (values - mean) / std
    indexes = np.arange(window - 1, len(values), stride)
    # Never compress invalid rows: doing so fabricates continuity across gaps.
    # IO-VNBD is nominally 10 Hz; reject backward, duplicated or missing times.
    times = frame.elapsed_s.to_numpy(dtype=np.float64)
    intervals = np.diff(times)
    good_interval = np.isfinite(intervals) & (intervals >= .05) & (intervals <= .15)
    indexes = np.asarray([i for i in indexes
                          if valid[i-window+1:i+1].all()
                          and good_interval[i-window+1:i].all()], dtype=int)
    sequences = (np.stack([values[i - window + 1:i + 1] for i in indexes])
                 if len(indexes) else np.empty((0, window, len(FEATURES)), dtype=np.float32))
    if target_mode == "delta":
        # A 20-sample 10 Hz trailing window spans 1.9 seconds between endpoints.
        # The raw-axis model still requires mounting calibration/augmentation.
        labels = targets[indexes] - targets[indexes - window + 1]
    else:
        labels = targets[indexes]
    return sequences, labels, mean, std


def assemble(pairs: dict, names, mean=None, std=None, window=20, stride=5, target_mode="delta"):
    frames = []
    for name in names:
        if name not in pairs:
            raise ValueError(f"Required drive {name} not found")
        frames.append(load_drive(pairs[name]))
    # Compute feature normalization over every training drive, rather than
    # accidentally using only the first drive in the list.
    if mean is None:
        all_values = np.concatenate([frame[FEATURES].to_numpy(dtype=np.float32) for frame in frames])
        valid = np.isfinite(all_values).all(axis=1)
        mean = all_values[valid].mean(axis=0); std = all_values[valid].std(axis=0).clip(1e-4)
    xs, ys = [], []
    for frame in frames:
        x, y, _, _ = windows(frame, mean, std, window, stride, target_mode)
        xs.append(x); ys.append(y)
    return np.concatenate(xs), np.concatenate(ys), mean, std


def mae(model, loader, device):
    model.eval(); errors = []
    with torch.no_grad():
        for x, y in loader:
            errors.append(torch.abs(model(x.to(device)).cpu() - y).numpy())
    return float(np.concatenate(errors).mean())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/phase-3"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--train-stride", type=int, default=20, help="10 Hz samples between training windows")
    parser.add_argument("--target", choices=("delta", "absolute"), default="delta")
    parser.add_argument("--all-drives", action="store_true", help="Final fit on every paired IO-VNBD drive; no independent test score is reported.")
    args = parser.parse_args()
    torch.manual_seed(26168); np.random.seed(26168)
    torch.set_num_threads(1)
    pairs = pair_by_name(args.dataset)
    train_names = tuple(sorted(pairs)) if args.all_drives else TRAIN_DRIVES
    x_train, y_train, mean, std = assemble(pairs, train_names, stride=args.train_stride, target_mode=args.target)
    if args.all_drives:
        x_val, y_val = x_train, y_train; x_test = y_test = None
    else:
        x_val, y_val, _, _ = assemble(pairs, (VALIDATION_DRIVE,), mean, std, stride=args.train_stride, target_mode=args.target)
        x_test, y_test, _, _ = assemble(pairs, (TEST_DRIVE,), mean, std, stride=1, target_mode=args.target)
    train = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train)), batch_size=256, shuffle=True)
    val = DataLoader(TensorDataset(torch.tensor(x_val), torch.tensor(y_val)), batch_size=512)
    test = None if x_test is None else DataLoader(TensorDataset(torch.tensor(x_test), torch.tensor(y_test)), batch_size=512)
    device = torch.device("mps" if args.device == "mps" and torch.backends.mps.is_available() else "cpu")
    model = SpeedTCN(len(FEATURES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    best_state, best_val = None, float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train:
            optimizer.zero_grad(); loss = loss_fn(model(x.to(device)), y.to(device)); loss.backward(); optimizer.step()
        validation_mae = mae(model, val, device)
        if validation_mae < best_val:
            best_val = validation_mae; best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"epoch {epoch:02d}: validation MAE {validation_mae:.3f} m/s")
    model.load_state_dict(best_state)
    test_mae = None if test is None else mae(model, test, device)
    args.out.mkdir(parents=True, exist_ok=True)
    artifact = {"state_dict": model.state_dict(), "features": FEATURES, "window": 20, "mean": mean.tolist(), "std": std.tolist()}
    torch.save(artifact, args.out / "speed_tcn.pt")
    report = {
        "model": "SpeedTCN (phone IMU only)", "output": f"two-second {args.target} speed estimate (m/s)", "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "sampling_hz": 10, "window_seconds": 2.0,
        "train_drives": list(train_names), "validation_drive": "training-set monitoring only" if args.all_drives else VALIDATION_DRIVE, "held_out_test_drive": None if args.all_drives else TEST_DRIVE,
        "best_validation_mae_mps": round(best_val, 4), "held_out_test_mae_mps": None if test_mae is None else round(test_mae, 4),
        "held_out_test_mae_kmh": None if test_mae is None else round(test_mae * 3.6, 4),
        "integrity": "Vehicle reference speed is a supervised training label only; no reference data is supplied to model inference. Final all-drive fit has no independent test score.",
    }
    (args.out / "speed_model_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
