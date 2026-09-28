"""Profile a raw NavRakshak Android recording for device-specific calibration."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/phone-profile"))
    args = parser.parse_args()
    frame = pd.read_csv(args.input, na_values=["null"])
    time_s = (frame.time_ns - frame.time_ns.iloc[0]) / 1e9
    dt = np.diff(time_s); positive_dt = dt[dt > 0]
    linear = frame[["acc_x", "acc_y", "acc_z"]].to_numpy() - frame[["gravity_x", "gravity_y", "gravity_z"]].to_numpy()
    linear_norm = np.linalg.norm(linear, axis=1)
    gyro_norm = np.linalg.norm(frame[["gyro_x", "gyro_y", "gyro_z"]].to_numpy(), axis=1)
    smooth = np.convolve(linear_norm, np.ones(9) / 9, mode="same")
    vibration = np.abs(linear_norm - smooth)
    gnss = frame.dropna(subset=["gnss_east_m", "gnss_north_m"])
    # Location logs repeat between GNSS callbacks; retain actual changes only.
    changed = gnss[(gnss.gnss_east_m.diff().fillna(1) != 0) | (gnss.gnss_north_m.diff().fillna(1) != 0)]
    gps_distance = float(np.hypot(np.diff(changed.gnss_east_m), np.diff(changed.gnss_north_m)).sum()) if len(changed) > 1 else 0.0
    args.out.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].plot(time_s, linear_norm, label="linear acceleration")
    axes[0, 0].plot(time_s, smooth, label="smoothed", linewidth=2)
    axes[0, 0].set(title="Bike/phone motion profile", xlabel="time (s)", ylabel="m/s²"); axes[0, 0].legend()
    axes[0, 1].plot(time_s, gyro_norm, color="#7c3aed")
    axes[0, 1].set(title="Angular-rate magnitude", xlabel="time (s)", ylabel="rad/s")
    axes[1, 0].plot(time_s, vibration, color="#d97706")
    axes[1, 0].axhline(np.percentile(vibration, 95), color="black", linestyle="--", label="95th percentile")
    axes[1, 0].set(title="Vibration / bump indicator", xlabel="time (s)", ylabel="m/s²"); axes[1, 0].legend()
    if len(changed): axes[1, 1].plot(changed.gnss_east_m, changed.gnss_north_m, marker=".", linewidth=1)
    axes[1, 1].set(title="Recorded GNSS path", xlabel="east (m)", ylabel="north (m)"); axes[1, 1].axis("equal")
    figure.tight_layout(); figure.savefig(args.out / "iqoo_z7_session_profile.png", dpi=160); plt.close(figure)
    report = [
        f"Source: {args.input.name}", f"Duration: {time_s.iloc[-1]:.1f} s", f"IMU samples: {len(frame):,}",
        f"Median IMU update rate: {1 / np.median(positive_dt):.2f} Hz", f"GNSS-bearing log rows: {len(gnss):,}",
        f"Distinct GNSS positions: {len(changed):,}", f"Approximate recorded GNSS distance: {gps_distance:.1f} m",
        f"Median reported GNSS accuracy: {gnss.gnss_accuracy_m.median():.2f} m", f"Linear-acceleration median / 95th: {np.median(linear_norm):.2f} / {np.percentile(linear_norm,95):.2f} m/s²",
        f"Vibration indicator 95th percentile: {np.percentile(vibration,95):.2f} m/s²",
        "Note: fusion state is WAITING_FOR_GNSS throughout this log due to the repaired initialization bug; raw IMU/GNSS measurements remain valid for calibration.",
    ]
    (args.out / "iqoo_z7_session_report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
