"""Replay IO-VNBD phone data through the NavRakshak 10 Hz fusion engine."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine" / "src"))
from navrakshak_idr import IntelligentDeadReckoner
from io_vnbd import choose_pair, find_pairs, load_drive
from profile_drive import local_xy


def circular_heading(east: np.ndarray, north: np.ndarray) -> np.ndarray:
    return np.arctan2(np.gradient(east), np.gradient(north))


def fit_phone_alignment(time_s, phone_speed_mps, phone_heading_rad, linear_acc, gyro, calibration_mask):
    """GNSS-aided phone-to-vehicle alignment available before an outage."""
    indexes = np.where(calibration_mask)[0]
    speed_rate = np.gradient(phone_speed_mps[indexes], time_s[indexes])
    best_axis, best_scale, best_bias, best_mse = 0, 1.0, 0.0, float("inf")
    for axis in range(3):
        scale, bias = np.linalg.lstsq(np.c_[linear_acc[indexes, axis], np.ones(len(indexes))], speed_rate, rcond=None)[0]
        mse = np.mean((scale * linear_acc[indexes, axis] + bias - speed_rate) ** 2)
        if mse < best_mse:
            best_axis, best_scale, best_bias, best_mse = axis, float(scale), float(bias), float(mse)
    heading_rate = np.gradient(np.unwrap(phone_heading_rad[indexes]), time_s[indexes])
    yaw_scale, yaw_bias = np.linalg.lstsq(np.c_[gyro[indexes], np.ones(len(indexes))], heading_rate, rcond=None)[0]
    return best_axis, best_scale, best_bias, float(yaw_scale), float(yaw_bias)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--drive", default="Vta10")
    parser.add_argument("--outage-start-s", type=float, default=20.0)
    parser.add_argument("--outage-s", type=float, default=60.0)
    parser.add_argument("--out", type=Path, default=Path("outputs/phase-5"))
    args = parser.parse_args()
    frame = load_drive(choose_pair(find_pairs(args.dataset), args.drive)).dropna().reset_index(drop=True)
    ref_east, ref_north = local_xy(frame.ref_lat_deg.to_numpy(), frame.ref_lon_deg.to_numpy())
    phone_east, phone_north = local_xy(frame.phone_lat_deg.to_numpy(), frame.phone_lon_deg.to_numpy())
    # Translate phone coordinates to the reference local frame; in a live app
    # this common frame begins at the accepted first GNSS fix.
    phone_east += ref_east[0] - phone_east[0]; phone_north += ref_north[0] - phone_north[0]
    # Android's GNSS bearing is a vehicle-course observation, far more stable
    # than differentiating noisy latitude/longitude coordinates.
    phone_heading = np.deg2rad(frame.phone_bearing_deg.to_numpy())
    time_s = frame.elapsed_s.to_numpy(); relative_s = time_s - time_s[0]
    outage = (relative_s >= args.outage_start_s) & (relative_s < args.outage_start_s + args.outage_s)
    linear = np.column_stack([frame.acc_x_mps2 - frame.gravity_x_mps2, frame.acc_y_mps2 - frame.gravity_y_mps2, frame.acc_z_mps2 - frame.gravity_z_mps2])
    calibration_mask = relative_s < args.outage_start_s
    axis, acc_scale, acc_bias, yaw_scale, yaw_bias = fit_phone_alignment(time_s, frame.phone_speed_kmh.to_numpy() / 3.6, phone_heading, linear, frame.gyro_yaw_rps.to_numpy(), calibration_mask)
    forward_acc = acc_scale * linear[:, axis] + acc_bias
    yaw_rate = yaw_scale * frame.gyro_yaw_rps.to_numpy() + yaw_bias
    vibration = np.clip(np.abs(forward_acc - np.convolve(forward_acc, np.ones(5) / 5, mode="same")) / 4.0, 0.0, 1.0)
    engine = IntelligentDeadReckoner(); states = []
    for i in range(len(frame)):
        gnss = {} if outage[i] else dict(gnss_east_m=float(phone_east[i]), gnss_north_m=float(phone_north[i]), gnss_speed_mps=float(frame.phone_speed_kmh.iloc[i] / 3.6), gnss_heading_rad=float(phone_heading[i]), gnss_accuracy_m=float(frame.phone_accuracy_m.iloc[i]))
        state = engine.step(float(time_s[i]), float(forward_acc[i]), float(yaw_rate[i]), float(vibration[i]), **gnss)
        states.append((state.east_m, state.north_m, state.mode))
    states = np.array(states, dtype=object); fused_e = states[:, 0].astype(float); fused_n = states[:, 1].astype(float)
    # Phone and vehicle antennas have a dataset-specific spatial/time offset.
    # Score outage *drift* by aligning both trajectories at blackout start;
    # reference data is never passed to the filter.
    error = np.hypot(fused_e - ref_east, fused_n - ref_north)
    outage_start_index = np.where(outage)[0][0]
    outage_drift = np.hypot(
        (fused_e - fused_e[outage_start_index]) - (ref_east - ref_east[outage_start_index]),
        (fused_n - fused_n[outage_start_index]) - (ref_north - ref_north[outage_start_index]),
    )
    valid = ~outage
    gnss_conformance_error = np.hypot(fused_e - phone_east, fused_n - phone_north)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.out / "fusion_replay.csv", np.column_stack((relative_s, fused_e, fused_n, error, outage_drift, outage.astype(int))), delimiter=",", header="elapsed_s,fused_east_m,fused_north_m,absolute_error_m,outage_drift_m,gnss_denied", comments="")
    metrics = [f"Drive: {args.drive}", f"Update count: {len(frame)} ({len(frame) / (relative_s[-1] - relative_s[0]):.2f} Hz)", f"GNSS-aided fusion-to-phone median residual: {np.median(gnss_conformance_error[valid]):.2f} m", f"Vehicle-reference median offset (dataset antenna/time offset included): {np.median(error[valid]):.2f} m", f"Outage final drift versus vehicle reference: {outage_drift[np.where(outage)[0][-1]]:.2f} m", f"Phone alignment: acceleration axis {axis}, scale {acc_scale:.4f}; yaw scale {yaw_scale:.4f}", "Integrity: only S-dataset phone inputs were passed to the fusion engine; vehicle reference was used only for scoring."]
    (args.out / "fusion_metrics.txt").write_text("\n".join(metrics) + "\n")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].plot(ref_east, ref_north, label="vehicle reference", linewidth=2)
    axes[0].plot(fused_e, fused_n, label="NavRakshak GNSS+INS")
    axes[0].set(title="10 Hz GNSS+INS fusion replay", xlabel="east (m)", ylabel="north (m)"); axes[0].axis("equal"); axes[0].legend()
    axes[1].plot(relative_s, outage_drift, label="aligned outage drift"); axes[1].axvspan(args.outage_start_s, args.outage_start_s + args.outage_s, color="#fee2e2", label="GNSS denied")
    axes[1].set(title="Seamless mode transition", xlabel="time (s)", ylabel="drift (m)"); axes[1].legend()
    figure.tight_layout(); figure.savefig(args.out / "fusion_replay.png", dpi=160); plt.close(figure)
    print("\n".join(metrics)); return 0


if __name__ == "__main__":
    sys.exit(main())
