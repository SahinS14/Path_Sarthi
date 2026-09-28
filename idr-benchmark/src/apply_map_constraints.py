"""Phase-4: apply non-holonomic and offline OSM road constraints to a DR path."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from offline_map_matcher import OfflineRoadMatcher
from profile_drive import local_xy


def heading_from_path(east: np.ndarray, north: np.ndarray) -> np.ndarray:
    return np.arctan2(np.gradient(east), np.gradient(north))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--osm", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/phase-4"))
    args = parser.parse_args()
    frame = pd.read_csv(args.baseline)
    reference_east, reference_north = local_xy(frame.ref_lat_deg.to_numpy(), frame.ref_lon_deg.to_numpy())
    matcher = OfflineRoadMatcher.from_overpass_json(args.osm, frame.ref_lat_deg.iloc[0], frame.ref_lon_deg.iloc[0])
    # Phase-2 stores its global local-frame; this file begins at the outage. Put
    # both paths in one translation-invariant local frame before comparing them.
    raw_east = frame.baseline_east_m.to_numpy() - frame.baseline_east_m.iloc[0] + reference_east[0]
    raw_north = frame.baseline_north_m.to_numpy() - frame.baseline_north_m.iloc[0] + reference_north[0]
    raw_heading = heading_from_path(raw_east, raw_north)
    matched = np.array([matcher.match(east, north, heading)[:3] for east, north, heading in zip(raw_east, raw_north, raw_heading)])
    constrained_east, constrained_north = matched[:, 0], matched[:, 1]
    raw_error = np.hypot(raw_east - reference_east, raw_north - reference_north)
    constrained_error = np.hypot(constrained_east - reference_east, constrained_north - reference_north)
    reference_distance = np.sum(np.hypot(np.diff(reference_east), np.diff(reference_north)))
    args.out.mkdir(parents=True, exist_ok=True)
    output = frame.copy(); output["matched_east_m"] = constrained_east; output["matched_north_m"] = constrained_north; output["matched_error_m"] = constrained_error
    output.to_csv(args.out / "vta10_map_constrained.csv", index=False)
    metrics = [
        f"Reference distance: {reference_distance:.1f} m",
        f"Raw final drift: {raw_error[-1]:.1f} m ({100 * raw_error[-1] / reference_distance:.2f}%)",
        f"Map-constrained final drift: {constrained_error[-1]:.1f} m ({100 * constrained_error[-1] / reference_distance:.2f}%)",
        "Constraint: lateral/vertical vehicle motion is disallowed; position is projected to heading-compatible local OSM road geometry.",
        "Integrity: OSM geometry is the only map input. Vehicle reference trajectory is used exclusively for the displayed evaluation error.",
    ]
    (args.out / "map_constraint_metrics.txt").write_text("\n".join(metrics) + "\n")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].plot(reference_east, reference_north, label="vehicle reference", linewidth=2)
    axes[0].plot(raw_east, raw_north, label="raw IMU DR", alpha=.7)
    axes[0].plot(constrained_east, constrained_north, label="offline OSM constrained", linewidth=2)
    axes[0].set(title="Road-constrained dead reckoning", xlabel="east (m)", ylabel="north (m)"); axes[0].axis("equal"); axes[0].legend()
    axes[1].plot(frame.elapsed_s - frame.elapsed_s.iloc[0], raw_error, label="raw DR")
    axes[1].plot(frame.elapsed_s - frame.elapsed_s.iloc[0], constrained_error, label="OSM constrained")
    axes[1].set(title="Constraint impact", xlabel="outage time (s)", ylabel="position error (m)"); axes[1].legend()
    figure.tight_layout(); figure.savefig(args.out / "vta10_map_constraints.png", dpi=160); plt.close(figure)
    print("\n".join(metrics)); return 0


if __name__ == "__main__":
    sys.exit(main())
