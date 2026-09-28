"""Evaluate heading-aware offline OSM constraints on a real phone blackout."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_phone_session import longest_outage
from offline_map_matcher import OfflineRoadMatcher


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--osm", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/phone-map"))
    args = parser.parse_args()
    data = pd.read_csv(args.input, na_values=["null"])
    start, end = longest_outage(data["mode"].to_numpy())
    interval = data.iloc[start:end].copy()
    if len(interval) < 10:
        raise ValueError("No usable simulated GNSS outage")
    origin = data.dropna(subset=["gnss_lat_deg", "gnss_lon_deg"]).iloc[0]
    matcher = OfflineRoadMatcher.from_overpass_json(args.osm, origin.gnss_lat_deg, origin.gnss_lon_deg)
    headings = np.arctan2(np.gradient(interval.east_m), np.gradient(interval.north_m))
    matches = np.array([matcher.match(e, n, h)[:3] for e, n, h in zip(interval.east_m, interval.north_m, headings)])
    truth = interval.dropna(subset=["gnss_lat_deg", "gnss_lon_deg"]).copy()
    r = 6_371_000.0; lat0 = np.deg2rad(origin.gnss_lat_deg); lon0 = np.deg2rad(origin.gnss_lon_deg)
    truth["truth_n"] = r * (np.deg2rad(truth.gnss_lat_deg) - lat0)
    truth["truth_e"] = r * (np.deg2rad(truth.gnss_lon_deg) - lon0) * np.cos(lat0)
    truth["raw_error_m"] = np.hypot(truth.east_m - truth.truth_e, truth.north_m - truth.truth_n)
    match_table = pd.DataFrame({"time_ns": interval.time_ns, "matched_east_m": matches[:, 0], "matched_north_m": matches[:, 1], "matched_heading_rad": matches[:, 2]})
    score = truth.merge(match_table, on="time_ns", how="left")
    score["map_error_m"] = np.hypot(score.matched_east_m - score.truth_e, score.matched_north_m - score.truth_n)
    args.out.mkdir(parents=True, exist_ok=True)
    score.to_csv(args.out / "iqoo_map_outage_evaluation.csv", index=False)
    metrics = [
        f"Session: {args.input.name}", f"Offline OSM ways: {len(matcher.segments)} segments",
        f"Outage duration: {(interval.time_ns.iloc[-1] - interval.time_ns.iloc[0]) / 1e9:.1f} s",
        f"Raw final error: {score.raw_error_m.iloc[-1]:.1f} m",
        f"Map-constrained final error: {score.map_error_m.iloc[-1]:.1f} m",
        "Integrity: map matcher receives only dead-reckoned position and its heading; logged GNSS is evaluation-only.",
    ]
    (args.out / "iqoo_map_metrics.txt").write_text("\n".join(metrics) + "\n")
    print("\n".join(metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
