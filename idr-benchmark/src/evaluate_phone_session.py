"""Evaluate a real NavRakshak phone session using hidden GNSS as outage truth."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def longest_outage(mode: np.ndarray) -> tuple[int, int]:
    start = end = best_start = best_end = 0
    for index, value in enumerate(mode):
        if value == "DEAD_RECKONING":
            if start == end: start = index
            end = index + 1
            if end - start > best_end - best_start: best_start, best_end = start, end
        else: start = end = index + 1
    return best_start, best_end


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/phone-session"))
    args = parser.parse_args()
    data = pd.read_csv(args.input, na_values=["null"])
    time_s = (data.time_ns - data.time_ns.iloc[0]) / 1e9
    start, end = longest_outage(data["mode"].to_numpy())
    if end - start < 10: raise ValueError("No usable continuous DEAD_RECKONING interval found")
    interval = data.iloc[start:end].copy(); interval_time = time_s.iloc[start:end]
    valid_truth = interval.dropna(subset=["gnss_lat_deg", "gnss_lon_deg"]).copy()
    radius = 6_371_000.0
    lat0 = np.deg2rad(valid_truth.gnss_lat_deg.iloc[0]); lon0 = np.deg2rad(valid_truth.gnss_lon_deg.iloc[0])
    truth_north = radius * (np.deg2rad(valid_truth.gnss_lat_deg) - lat0)
    truth_east = radius * (np.deg2rad(valid_truth.gnss_lon_deg) - lon0) * np.cos(lat0)
    # Align local origins at outage start; only subsequent drift is measured.
    truth_east += valid_truth.east_m.iloc[0] - truth_east.iloc[0]
    truth_north += valid_truth.north_m.iloc[0] - truth_north.iloc[0]
    valid_truth["truth_east_m"] = truth_east; valid_truth["truth_north_m"] = truth_north
    error = np.hypot(valid_truth.east_m - truth_east, valid_truth.north_m - truth_north)
    changes = valid_truth[(valid_truth.truth_east_m.diff().fillna(1) != 0) | (valid_truth.truth_north_m.diff().fillna(1) != 0)]
    distance = float(np.hypot(np.diff(changes.truth_east_m), np.diff(changes.truth_north_m)).sum())
    args.out.mkdir(parents=True, exist_ok=True)
    table = valid_truth[["time_ns", "east_m", "north_m", "truth_east_m", "truth_north_m", "speed_mps", "gnss_speed_mps"]].copy()
    table["position_error_m"] = error; table.to_csv(args.out / "iqoo_outage_evaluation.csv", index=False)
    metrics = [
        f"Session: {args.input.name}", f"Longest simulated GNSS outage: {interval_time.iloc[-1] - interval_time.iloc[0]:.1f} s", f"Outage samples: {len(interval):,}",
        f"GNSS reference distance during outage: {distance:.1f} m", f"Final dead-reckoning error: {error.iloc[-1]:.1f} m", f"Final drift: {100 * error.iloc[-1] / max(distance, 1):.2f}%", f"Median GNSS truth accuracy during outage: {valid_truth.gnss_accuracy_m.median():.2f} m",
        "Method: the app withheld GNSS from fusion during DEMO OUTAGE while continuing to log the raw GNSS stream solely for post-run evaluation.",
    ]
    (args.out / "iqoo_outage_metrics.txt").write_text("\n".join(metrics) + "\n")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].plot(valid_truth.truth_east_m, valid_truth.truth_north_m, label="hidden GNSS reference", linewidth=2)
    axes[0].plot(valid_truth.east_m, valid_truth.north_m, label="on-device dead reckoning")
    axes[0].scatter(valid_truth.truth_east_m.iloc[0], valid_truth.truth_north_m.iloc[0], label="outage starts", c="#16a34a", zorder=3)
    axes[0].set(title="Real bike GNSS-outage replay", xlabel="east (m)", ylabel="north (m)"); axes[0].axis("equal"); axes[0].legend()
    axes[1].plot((valid_truth.time_ns - valid_truth.time_ns.iloc[0]) / 1e9, error, color="#dc2626")
    axes[1].set(title="Measured dead-reckoning drift", xlabel="outage time (s)", ylabel="position error (m)")
    figure.tight_layout(); figure.savefig(args.out / "iqoo_outage_evaluation.png", dpi=160); plt.close(figure)
    print("\n".join(metrics))


if __name__ == "__main__": main()
