"""Aggregate provisional IO-VNBD GNSS-blackout benchmark for NavRakshak.

Vehicle labels are used only after inference to score aligned outage drift.
Phone GNSS is allowed before each outage for normal GNSS-aided calibration and
initialisation, but never inside the simulated outage.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'engine/src'))
from vehicle_state_filter import VehicleStateFilter
from evaluate_fusion import fit_phone_alignment
from io_vnbd import find_pairs, load_drive
from profile_drive import local_xy


def scalar(value):
    return None if value is None else float(value)


def run_drive(pair, warmup_s, outage_s, outage_acceleration_gain=1., speed_anchor_time_s=None):
    frame = load_drive(pair).dropna().reset_index(drop=True)
    t = frame.elapsed_s.to_numpy(dtype=float)
    t -= t[0]
    start = int(np.searchsorted(t, warmup_s))
    end = int(np.searchsorted(t, warmup_s + outage_s))
    if start < 20 or end >= len(frame) or end - start < 10:
        raise ValueError('insufficient continuous duration')
    dt = np.diff(t[:end])
    if not np.all((dt > .05) & (dt < .15)):
        raise ValueError('non-10-Hz timestamp gap')
    ref_e, ref_n = local_xy(frame.ref_lat_deg.to_numpy(), frame.ref_lon_deg.to_numpy())
    phone_e, phone_n = local_xy(frame.phone_lat_deg.to_numpy(), frame.phone_lon_deg.to_numpy())
    # Both trajectories have arbitrary origins. Align only their starting frames.
    phone_e += ref_e[0] - phone_e[0]; phone_n += ref_n[0] - phone_n[0]
    heading = np.deg2rad(frame.phone_bearing_deg.to_numpy())
    linear = np.column_stack((frame.acc_x_mps2-frame.gravity_x_mps2,
                              frame.acc_y_mps2-frame.gravity_y_mps2,
                              frame.acc_z_mps2-frame.gravity_z_mps2))
    pre = np.arange(len(frame)) < start
    axis, scale, bias, yaw_scale, yaw_bias = fit_phone_alignment(
        t, frame.phone_speed_kmh.to_numpy()/3.6, heading, linear,
        frame.gyro_yaw_rps.to_numpy(), pre)
    forward = scale*linear[:, axis]+bias
    yaw = yaw_scale*frame.gyro_yaw_rps.to_numpy()+yaw_bias
    jerk = np.abs(forward-np.convolve(forward, np.ones(5)/5, mode='same'))
    confidence = np.clip(1-jerk/4, .05, 1)
    stationary = ((np.abs(forward) < .12) & (np.abs(yaw) < .04)).astype(float)
    f = VehicleStateFilter(outage_acceleration_gain=outage_acceleration_gain, speed_anchor_time_s=speed_anchor_time_s); east = []; north = []
    for i in range(end):
        gnss = {} if i >= start else dict(gnss_east_m=float(phone_e[i]), gnss_north_m=float(phone_n[i]),
                                           gnss_speed_mps=float(frame.phone_speed_kmh.iloc[i]/3.6),
                                           gnss_heading_rad=float(heading[i]),
                                           gnss_accuracy_m=float(frame.phone_accuracy_m.iloc[i]))
        state = f.step(float(t[i]), float(forward[i]), float(yaw[i]), float(confidence[i]),
                       float(stationary[i]), **gnss)
        east.append(state.east_m); north.append(state.north_m)
    east, north = np.asarray(east), np.asarray(north)
    ref_delta = np.column_stack((ref_e[start:end]-ref_e[start], ref_n[start:end]-ref_n[start]))
    state_delta = np.column_stack((east[start:end]-east[start], north[start:end]-north[start]))
    error = np.linalg.norm(state_delta-ref_delta, axis=1)
    distance = float(np.linalg.norm(np.diff(ref_delta, axis=0), axis=1).sum())
    speed0 = float(frame.phone_speed_kmh.iloc[start]/3.6)
    cv_delta = (t[start:end]-t[start])[:, None]*speed0*np.array([np.sin(heading[start]), np.cos(heading[start])])
    cv_error = np.linalg.norm(cv_delta-ref_delta, axis=1)
    return dict(drive=pair.name, outage_seconds=outage_s, reference_distance_m=distance,
                filter_endpoint_error_m=float(error[-1]), filter_rmse_m=float(np.sqrt(np.mean(error**2))),
                filter_drift_percent=scalar(100*error[-1]/distance if distance >= 1 else None),
                constant_velocity_endpoint_error_m=float(cv_error[-1]),
                constant_velocity_drift_percent=scalar(100*cv_error[-1]/distance if distance >= 1 else None),
                calibration=dict(acceleration_axis=int(axis), acceleration_scale=float(scale), yaw_scale=float(yaw_scale)))


def summarize(records):
    out = {}
    for duration in sorted({r['outage_seconds'] for r in records}):
        sample = [r for r in records if r['outage_seconds'] == duration and r['reference_distance_m'] >= 1]
        if not sample:
            out[str(duration)] = dict(cases=0, median_filter_drift_percent=None,
                                      median_constant_velocity_drift_percent=None,
                                      filter_under_10_percent=0, constant_velocity_under_10_percent=0)
            continue
        out[str(duration)] = dict(cases=len(sample),
                                  median_filter_drift_percent=float(np.median([r['filter_drift_percent'] for r in sample])),
                                  median_constant_velocity_drift_percent=float(np.median([r['constant_velocity_drift_percent'] for r in sample])),
                                  filter_under_10_percent=sum(r['filter_drift_percent'] < 10 for r in sample),
                                  constant_velocity_under_10_percent=sum(r['constant_velocity_drift_percent'] < 10 for r in sample))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--warmup-s', type=float, default=20)
    parser.add_argument('--outages-s', type=float, nargs='+', default=[10, 30, 60])
    parser.add_argument('--outage-acceleration-gain', type=float, default=1.)
    parser.add_argument('--speed-anchor-time-s', type=float, default=None)
    args = parser.parse_args()
    if args.out.exists(): parser.error('Use a new output filename')
    if args.warmup_s < 2 or any(x <= 0 for x in args.outages_s) or not 0 <= args.outage_acceleration_gain <= 1 or (args.speed_anchor_time_s is not None and args.speed_anchor_time_s <= 0): parser.error('Invalid outage configuration')
    records=[]; skipped=[]
    for pair in sorted(find_pairs(args.dataset), key=lambda x: x.name):
        for duration in args.outages_s:
            try: records.append(run_drive(pair, args.warmup_s, duration, args.outage_acceleration_gain, args.speed_anchor_time_s))
            except (ValueError, IndexError, np.linalg.LinAlgError) as error:
                skipped.append(dict(drive=pair.name, outage_seconds=duration, reason=str(error)))
    report=dict(scope='Provisional IO-VNBD replay. Vehicle data is scoring-only after the blackout begins.',
                configuration=dict(warmup_seconds=args.warmup_s, outages_seconds=args.outages_s, outage_acceleration_gain=args.outage_acceleration_gain, speed_anchor_time_s=args.speed_anchor_time_s),
                summary=summarize(records), records=records, skipped=skipped,
                limitations=['Phone/vehicle timing and sensor-frame contract remains provisional.',
                             'No offline map matching or learned adaptive covariance is included.',
                             'This report is a development gate, not PS compliance evidence.'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__': main()
