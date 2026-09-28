"""Plot held-out speed-increment predictions from the Phase-3 model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from io_vnbd import choose_pair, find_pairs, load_drive
from train_speed_model import FEATURES, SpeedTCN, windows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--drive", default="Vta10")
    parser.add_argument("--out", type=Path, default=Path("outputs/phase-3"))
    args = parser.parse_args()
    artifact = torch.load(args.model, map_location="cpu", weights_only=False)
    frame = load_drive(choose_pair(find_pairs(args.dataset), args.drive)).dropna().reset_index(drop=True)
    x, target, _, _ = windows(frame, np.array(artifact["mean"]), np.array(artifact["std"]), artifact["window"], 1, "delta")
    model = SpeedTCN(len(FEATURES)); model.load_state_dict(artifact["state_dict"]); model.eval()
    with torch.no_grad():
        prediction = model(torch.tensor(x, dtype=torch.float32)).numpy()
    time_s = frame.elapsed_s.to_numpy()[artifact["window"] - 1:]
    error = prediction - target
    args.out.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(time_s, target, label="vehicle reference Δspeed", alpha=.8)
    axes[0].plot(time_s, prediction, label="IMU model Δspeed", alpha=.8)
    axes[0].set(title=f"Held-out drive {args.drive}: 2-second velocity correction", ylabel="Δ speed (m/s)"); axes[0].legend()
    axes[1].plot(time_s, np.abs(error), color="#dc2626")
    axes[1].set(title=f"Absolute error — MAE {np.abs(error).mean():.3f} m/s", xlabel="time (s)", ylabel="error (m/s)")
    figure.tight_layout(); figure.savefig(args.out / f"{args.drive}_speed_model.png", dpi=160); plt.close(figure)
    print(f"Held-out delta-speed MAE: {np.abs(error).mean():.4f} m/s ({np.abs(error).mean() * 3.6:.4f} km/h)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
