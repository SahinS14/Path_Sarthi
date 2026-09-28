"""Independent-drive blackout evaluation for the vehicle-state filter."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine" / "src"))
from vehicle_state_filter import VehicleStateFilter
from evaluate_fusion import fit_phone_alignment
from io_vnbd import choose_pair, find_pairs, load_drive
from profile_drive import local_xy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True); parser.add_argument("--drive", default="Vta10")
    parser.add_argument("--outage-start-s", type=float, default=20); parser.add_argument("--outage-s", type=float, default=5)
    args = parser.parse_args()
    frame = load_drive(choose_pair(find_pairs(args.dataset), args.drive)).dropna().reset_index(drop=True)
    t = frame.elapsed_s.to_numpy(); rel = t - t[0]; outage = (rel >= args.outage_start_s) & (rel < args.outage_start_s + args.outage_s)
    ref_e, ref_n = local_xy(frame.ref_lat_deg.to_numpy(), frame.ref_lon_deg.to_numpy()); phone_e, phone_n = local_xy(frame.phone_lat_deg.to_numpy(), frame.phone_lon_deg.to_numpy())
    phone_e += ref_e[0] - phone_e[0]; phone_n += ref_n[0] - phone_n[0]
    heading = np.deg2rad(frame.phone_bearing_deg.to_numpy()); linear = np.column_stack([frame.acc_x_mps2-frame.gravity_x_mps2, frame.acc_y_mps2-frame.gravity_y_mps2, frame.acc_z_mps2-frame.gravity_z_mps2])
    pre = rel < args.outage_start_s
    axis, scale, bias, yaw_scale, yaw_bias = fit_phone_alignment(t, frame.phone_speed_kmh.to_numpy()/3.6, heading, linear, frame.gyro_yaw_rps.to_numpy(), pre)
    forward = scale*linear[:,axis]+bias; jerk = np.abs(forward-np.convolve(forward,np.ones(5)/5,mode="same")); confidence = np.clip(1-jerk/4, .05, 1)
    stationary = ((np.abs(forward) < .12) & (np.abs(yaw_scale*frame.gyro_yaw_rps.to_numpy()+yaw_bias) < .04)).astype(float)
    filter_ = VehicleStateFilter(); out_e=[]; out_n=[]
    for i in range(len(frame)):
        kwargs={} if outage[i] else dict(gnss_east_m=float(phone_e[i]),gnss_north_m=float(phone_n[i]),gnss_speed_mps=float(frame.phone_speed_kmh.iloc[i]/3.6),gnss_heading_rad=float(heading[i]),gnss_accuracy_m=float(frame.phone_accuracy_m.iloc[i]))
        state=filter_.step(float(t[i]),float(forward[i]),float(yaw_scale*frame.gyro_yaw_rps.iloc[i]+yaw_bias),float(confidence[i]),float(stationary[i]),**kwargs); out_e.append(state.east_m);out_n.append(state.north_m)
    start=np.where(outage)[0][0]; end=np.where(outage)[0][-1]; drift=np.hypot((np.array(out_e)-out_e[start])-(ref_e-ref_e[start]),(np.array(out_n)-out_n[start])-(ref_n-ref_n[start]))
    print(f"Drive: {args.drive}\nOutage: {args.outage_s:.1f} s\nFinal aligned drift: {drift[end]:.2f} m\nFilter: vehicle-state NHC + ZUPT, no map correction\n")
    return 0

if __name__ == "__main__": raise SystemExit(main())
