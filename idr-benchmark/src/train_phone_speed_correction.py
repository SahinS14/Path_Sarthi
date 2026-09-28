"""Fine-tune a compact IMU speed-correction model on an iQOO bike session.

Training is chronological: the final 30% of the ride is never seen during
training, avoiding leakage from adjacent sensor samples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


FEATURES = ["acc_x", "acc_y", "acc_z", "gravity_x", "gravity_y", "gravity_z", "gyro_x", "gyro_y", "gyro_z"]
WINDOW = 20  # 2 seconds at the measured ~10 Hz rate


class PhoneSpeedTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(9, 24, 5, padding=2), nn.ReLU(),
            nn.Conv1d(24, 24, 5, padding=2, groups=24), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(24, 1),
        )
    def forward(self, x): return self.layers(x.transpose(1, 2)).squeeze(1)


def make_windows(frame, mean=None, std=None):
    values = frame[FEATURES].to_numpy(np.float32); speed = frame.gnss_speed_mps.to_numpy(np.float32)
    if mean is None: mean, std = values.mean(0), values.std(0).clip(1e-4)
    values = (values - mean) / std
    end = np.arange(WINDOW - 1, len(frame))
    x = np.stack([values[i - WINDOW + 1:i + 1] for i in end])
    # The app starts with a GNSS speed; learn the change over the next 2 s.
    y = speed[end] - speed[end - WINDOW + 1]
    return x, y, mean, std


def predict(model, x):
    model.eval()
    with torch.no_grad(): return model(torch.tensor(x, dtype=torch.float32)).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/phone-model"))
    args = parser.parse_args(); torch.manual_seed(26168); np.random.seed(26168); torch.set_num_threads(1)
    frame = pd.read_csv(args.input, na_values=["null"]).dropna(subset=FEATURES + ["gnss_speed_mps"]).reset_index(drop=True)
    split = int(.70 * len(frame))
    x_train, y_train, mean, std = make_windows(frame.iloc[:split])
    # Test windows are created from the holdout segment only—no overlap.
    x_test, y_test, _, _ = make_windows(frame.iloc[split: ], mean, std)
    loader = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train)), batch_size=128, shuffle=True)
    model = PhoneSpeedTCN(); optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4); loss_fn = nn.SmoothL1Loss()
    for _ in range(40):
        model.train()
        for x, y in loader:
            optimizer.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); optimizer.step()
    prediction = predict(model, x_test); learned_mae = float(np.abs(prediction - y_test).mean()); zero_mae = float(np.abs(y_test).mean())
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "features": FEATURES, "window": WINDOW, "mean":mean.tolist(), "std":std.tolist()}, args.out / "iqoo_speed_delta_tcn.pt")
    report = {"input": args.input.name, "model":"PhoneSpeedTCN", "parameters":sum(p.numel() for p in model.parameters()), "target":"two-second GNSS-speed correction (m/s)", "train_samples":len(x_train), "held_out_samples":len(x_test), "zero_change_baseline_mae_mps":round(zero_mae,4), "learned_model_mae_mps":round(learned_mae,4), "relative_improvement_percent":round(100*(zero_mae-learned_mae)/max(zero_mae,1e-6),2), "integrity":"Chronological holdout; raw GNSS speed is used only as an offline training/evaluation label."}
    (args.out / "iqoo_speed_model_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
