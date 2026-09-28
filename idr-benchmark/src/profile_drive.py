"""Create Phase-1 data-quality figures for an IO-VNBD synchronised drive."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless/reproducible output; no GUI dependency.
import matplotlib.pyplot as plt
import numpy as np

from io_vnbd import choose_pair, find_pairs, load_drive


def local_xy(lat_deg: np.ndarray, lon_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equirectangular local coordinates, sufficient for short-drive plots."""
    radius_m = 6_371_000.0
    lat0 = np.deg2rad(lat_deg[0])
    lon0 = np.deg2rad(lon_deg[0])
    north_m = radius_m * (np.deg2rad(lat_deg) - lat0)
    east_m = radius_m * (np.deg2rad(lon_deg) - lon0) * np.cos(lat0)
    return east_m, north_m


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--drive", help="Directory drive name, e.g. Vw01")
    parser.add_argument("--out", type=Path, default=Path("outputs/phase-1"))
    args = parser.parse_args()

    pair = choose_pair(find_pairs(args.dataset), args.drive)
    frame = load_drive(pair).dropna(subset=["ref_lat_deg", "ref_lon_deg", "ref_speed_kmh"])
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / f"{pair.name}_canonical.csv", index=False)

    east, north = local_xy(frame.ref_lat_deg.to_numpy(), frame.ref_lon_deg.to_numpy())
    phone_east, phone_north = local_xy(frame.phone_lat_deg.to_numpy(), frame.phone_lon_deg.to_numpy())
    imu_norm = np.sqrt(frame.acc_x_mps2**2 + frame.acc_y_mps2**2 + frame.acc_z_mps2**2)

    # Keep rendering predictable on resource-constrained machines used at demos.
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].plot(east, north, label="vehicle reference", linewidth=2)
    axes[0, 0].plot(phone_east, phone_north, label="phone GNSS", alpha=.65)
    axes[0, 0].set(title=f"{pair.name}: position sources", xlabel="east (m)", ylabel="north (m)")
    axes[0, 0].axis("equal"); axes[0, 0].legend()
    axes[0, 1].plot(frame.elapsed_s, frame.ref_speed_kmh, label="vehicle reference")
    axes[0, 1].plot(frame.elapsed_s, frame.phone_speed_kmh, label="phone GNSS", alpha=.7)
    axes[0, 1].set(title="Speed label quality", xlabel="time (s)", ylabel="km/h"); axes[0, 1].legend()
    axes[1, 0].plot(frame.elapsed_s, imu_norm, color="#d97706")
    axes[1, 0].axhline(9.80665, color="black", linestyle="--", linewidth=1, label="gravity")
    axes[1, 0].set(title="Phone acceleration magnitude", xlabel="time (s)", ylabel="m/s²"); axes[1, 0].legend()
    axes[1, 1].plot(frame.elapsed_s, frame.gyro_yaw_rps, label="yaw")
    axes[1, 1].plot(frame.elapsed_s, frame.gyro_pitch_rps, label="pitch")
    axes[1, 1].plot(frame.elapsed_s, frame.gyro_roll_rps, label="roll")
    axes[1, 1].set(title="Phone angular-rate inputs", xlabel="time (s)", ylabel="rad/s"); axes[1, 1].legend()
    figure.tight_layout()
    figure.savefig(args.out / f"{pair.name}_profile.png", dpi=150)
    plt.close(figure)

    summary = [
        f"Drive: {pair.name}", f"Samples: {len(frame):,}",
        f"Duration: {frame.elapsed_s.iloc[-1] - frame.elapsed_s.iloc[0]:.1f} s",
        f"Reference speed: mean {frame.ref_speed_kmh.mean():.2f} km/h, max {frame.ref_speed_kmh.max():.2f} km/h",
        f"Phone GPS accuracy: median {frame.phone_accuracy_m.median():.2f} m",
        "Input: phone accelerometer, gyroscope, magnetometer, gravity and orientation.",
        "Training/evaluation reference: paired vehicle trajectory and velocity.",
    ]
    (args.out / f"{pair.name}_summary.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))
    print(f"Wrote {args.out / f'{pair.name}_profile.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
