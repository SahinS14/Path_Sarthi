"""Phase-2 GNSS outage simulation and calibrated phone-IMU baseline.

No vehicle reference samples from the outage are used for prediction. Reference
signals are used only before the outage to emulate normal GNSS-aided alignment,
and after it solely to calculate evaluation metrics.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from io_vnbd import choose_pair, find_pairs, load_drive
from profile_drive import local_xy


EARTH_RADIUS_M = 6_371_000.0


@dataclass
class Calibration:
    acceleration_axis: str
    acceleration_scale: float
    acceleration_bias: float
    yaw_axis: str
    yaw_scale: float
    yaw_bias: float


def derivative(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    return np.gradient(values, time_s)


def fit_axis(signals: dict[str, np.ndarray], target: np.ndarray) -> tuple[str, float, float]:
    """Select phone axis and fit target ~= scale * signal + bias."""
    best: tuple[str, float, float, float] | None = None
    design_target = np.column_stack((np.ones(len(target)), target))
    for name, signal in signals.items():
        scale, bias = np.linalg.lstsq(
            np.column_stack((signal, np.ones(len(signal)))), target, rcond=None
        )[0]
        estimate = scale * signal + bias
        mse = float(np.mean((estimate - target) ** 2))
        if best is None or mse < best[3]:
            best = (name, float(scale), float(bias), mse)
    assert best is not None
    return best[0], best[1], best[2]


def calibrate(frame, calibration_end: int) -> Calibration:
    time_s = frame.elapsed_s.to_numpy()[:calibration_end]
    reference_speed_mps = frame.ref_speed_kmh.to_numpy()[:calibration_end] / 3.6
    ref_acc = derivative(reference_speed_mps, time_s)
    linear = {
        axis: (frame[f"acc_{axis}_mps2"].to_numpy() - frame[f"gravity_{axis}_mps2"].to_numpy())[:calibration_end]
        for axis in ("x", "y", "z")
    }
    acc_axis, acc_scale, acc_bias = fit_axis(linear, ref_acc)

    reference_heading_rad = np.unwrap(np.deg2rad(frame.ref_heading_deg.to_numpy()[:calibration_end]))
    ref_yaw_rate = derivative(reference_heading_rad, time_s)
    gyro = {
        "yaw": frame.gyro_yaw_rps.to_numpy()[:calibration_end],
        "pitch": frame.gyro_pitch_rps.to_numpy()[:calibration_end],
        "roll": frame.gyro_roll_rps.to_numpy()[:calibration_end],
    }
    yaw_axis, yaw_scale, yaw_bias = fit_axis(gyro, ref_yaw_rate)
    return Calibration(acc_axis, acc_scale, acc_bias, yaw_axis, yaw_scale, yaw_bias)


def simulate(frame, start: int, end: int, calibration: Calibration) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate raw phone IMU from a GNSS-aided initial state."""
    time_s = frame.elapsed_s.to_numpy()
    ref_east, ref_north = local_xy(frame.ref_lat_deg.to_numpy(), frame.ref_lon_deg.to_numpy())
    dt = np.diff(time_s[start:end], prepend=time_s[start])
    dt[0] = np.median(dt[1:])
    axis = calibration.acceleration_axis
    linear_acc = frame[f"acc_{axis}_mps2"].to_numpy()[start:end] - frame[f"gravity_{axis}_mps2"].to_numpy()[start:end]
    forward_acc = calibration.acceleration_scale * linear_acc + calibration.acceleration_bias
    gyro = frame[f"gyro_{calibration.yaw_axis}_rps"].to_numpy()[start:end]
    yaw_rate = calibration.yaw_scale * gyro + calibration.yaw_bias

    east = np.empty(end - start); north = np.empty(end - start); speed = np.empty(end - start)
    east[0], north[0] = ref_east[start], ref_north[start]
    # In production this is Location.getSpeed() immediately before dropout.
    speed[0] = frame.ref_speed_kmh.iloc[start] / 3.6
    heading = np.deg2rad(frame.ref_heading_deg.iloc[start])
    for index in range(1, len(east)):
        speed[index] = max(0.0, speed[index - 1] + forward_acc[index] * dt[index])
        heading += yaw_rate[index] * dt[index]
        east[index] = east[index - 1] + speed[index] * np.sin(heading) * dt[index]
        north[index] = north[index - 1] + speed[index] * np.cos(heading) * dt[index]
    return east, north, speed


def path_distance(east: np.ndarray, north: np.ndarray) -> float:
    return float(np.sum(np.hypot(np.diff(east), np.diff(north))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--drive", default="Vta10")
    parser.add_argument("--out", type=Path, default=Path("outputs/phase-2"))
    parser.add_argument("--warmup-s", type=float, default=20.0)
    parser.add_argument("--outage-s", type=float, default=60.0)
    args = parser.parse_args()

    pair = choose_pair(find_pairs(args.dataset), args.drive)
    frame = load_drive(pair).dropna().reset_index(drop=True)
    elapsed = frame.elapsed_s.to_numpy() - frame.elapsed_s.iloc[0]
    start = int(np.searchsorted(elapsed, args.warmup_s))
    end = int(np.searchsorted(elapsed, args.warmup_s + args.outage_s))
    if end - start < 20:
        raise ValueError("Selected outage is too short for this drive")
    calibration = calibrate(frame, start)
    predicted_east, predicted_north, predicted_speed = simulate(frame, start, end, calibration)
    reference_east, reference_north = local_xy(frame.ref_lat_deg.to_numpy(), frame.ref_lon_deg.to_numpy())
    position_error = np.hypot(predicted_east - reference_east[start:end], predicted_north - reference_north[start:end])
    distance_m = path_distance(reference_east[start:end], reference_north[start:end])
    final_error_m = float(position_error[-1])

    args.out.mkdir(parents=True, exist_ok=True)
    metrics = [
        f"Drive: {pair.name}", f"GNSS outage: {elapsed[start]:.1f}s to {elapsed[end - 1]:.1f}s ({end - start} samples)",
        f"Reference distance: {distance_m:.1f} m", f"Final position error: {final_error_m:.1f} m",
        f"Drift: {100 * final_error_m / distance_m:.2f}%", f"Mean position error: {position_error.mean():.1f} m",
        f"Auto-calibrated forward acceleration: {calibration.acceleration_axis}-axis, scale={calibration.acceleration_scale:.4f}, bias={calibration.acceleration_bias:.4f}",
        f"Auto-calibrated yaw rate: {calibration.yaw_axis}-axis, scale={calibration.yaw_scale:.4f}, bias={calibration.yaw_bias:.4f}",
        "Integrity: vehicle reference was used only before the simulated outage for initial alignment, and after it for scoring.",
    ]
    (args.out / f"{pair.name}_baseline_metrics.txt").write_text("\n".join(metrics) + "\n")
    export = frame.iloc[start:end][["elapsed_s", "ref_lat_deg", "ref_lon_deg", "ref_speed_kmh"]].copy()
    export["baseline_east_m"] = predicted_east
    export["baseline_north_m"] = predicted_north
    export["position_error_m"] = position_error
    export.to_csv(args.out / f"{pair.name}_baseline.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].plot(reference_east[start:end], reference_north[start:end], label="vehicle reference", linewidth=2)
    axes[0].plot(predicted_east, predicted_north, label="raw phone-IMU dead reckoning", linewidth=2)
    axes[0].scatter(reference_east[start], reference_north[start], c="#16a34a", label="GNSS outage begins", zorder=4)
    axes[0].set(title="GNSS-denied trajectory", xlabel="east (m)", ylabel="north (m)"); axes[0].axis("equal"); axes[0].legend()
    axes[1].plot(elapsed[start:end] - elapsed[start], position_error, color="#dc2626", label="position error")
    axes[1].set(title="Drift during simulated GNSS outage", xlabel="outage time (s)", ylabel="error (m)"); axes[1].legend()
    figure.tight_layout(); figure.savefig(args.out / f"{pair.name}_baseline.png", dpi=160); plt.close(figure)
    print("\n".join(metrics))
    print(f"Wrote {args.out / f'{pair.name}_baseline.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
