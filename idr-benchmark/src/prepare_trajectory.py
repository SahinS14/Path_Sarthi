"""Prepare IO-VNBD trajectory windows only from explicitly verified contracts.

Initialize a review template from the session audit, then verify conventions with
evidence before preparation. A template is not permission to guess sensor frames.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from input_contract import convert_imu, proper_rotation, speed_mps, align_labels
from io_vnbd import _read, _select, PHONE_COLUMNS, VEHICLE_COLUMNS
from session_manifest import sha256, recording_dates, assign_groups


def initial_attitude(gravity_body, bearing_rad):
    g = np.asarray(gravity_body, float)
    if not np.isfinite(g).all() or not np.isfinite(bearing_rad) or not 8 < np.linalg.norm(g) < 12:
        raise ValueError('Invalid gravity or GNSS bearing for initialization')
    up = g/np.linalg.norm(g)
    forward = np.array([1., 0, 0]); forward -= up*up[0]
    if np.linalg.norm(forward) < .1:
        raise ValueError('Vehicle forward direction is nearly vertical')
    forward /= np.linalg.norm(forward)
    level = np.vstack((forward, np.cross(up, forward), up))
    yaw = np.pi/2-bearing_rad
    heading = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    rotation = heading@level
    return np.array([np.arctan2(rotation[2, 1], rotation[2, 2]),
                     np.arctan2(-rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0])),
                     np.arctan2(rotation[1, 0], rotation[0, 0])])


def initialize(audit, out):
    source = json.loads(audit.read_text())
    records = []
    for row in source['records']:
        records.append(dict(drive=row['drive'], phone_path=row['phone_path'], vehicle_path=row['vehicle_path'],
                            phone_sha256=row['phone_sha256'], vehicle_sha256=row['vehicle_sha256'],
                            audited_session_group=row['session_group'],
                            split=row['proposed_split'], status='unverified', evidence='',
                            phone_speed_unit=None, acceleration_to_body=None, gyro_to_body=None,
                            gravity_to_body=None, acceleration_kind=None,
                            reference_offset_s=None))
    payload = dict(schema='idr-recording-contracts-v1', records=records,
                   instructions='Do not mark verified without frame/unit/timing evidence. Offsets align independently zeroed elapsed clocks: reference_time + offset = phone_time.')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump(payload, stream, indent=2); stream.write('\n')


def load_record(record):
    if record.get('status') not in ('verified', 'provisional') or not record.get('evidence', '').strip():
        raise ValueError('Unverified recording contract')
    for name in ('phone', 'vehicle'):
        if sha256(Path(record[name+'_path'])) != record[name+'_sha256']:
            raise ValueError('Raw source hash mismatch')
    phone = _select(_read(Path(record['phone_path'])), PHONE_COLUMNS)
    vehicle = _select(_read(Path(record['vehicle_path'])), VEHICLE_COLUMNS)
    tp = phone.phone_time_ms.to_numpy()/1000; tp -= tp[0]
    tv = vehicle.vehicle_time_s.to_numpy(); tv -= tv[0]
    # Source midnight wrap or reset needs explicit repair evidence, not auto-wrap.
    acc = phone[['acc_x_mps2', 'acc_y_mps2', 'acc_z_mps2']].to_numpy()
    # Published channel labels are [Yaw, Pitch, Roll], whereas the legacy
    # upstream-compatible convention is [Roll, Pitch, Yaw] before rotation.
    # The order is part of each explicit recording contract, never guessed.
    gyro_order = record.get('gyro_channel_order', ['roll', 'pitch', 'yaw'])
    channels = {'yaw': 'gyro_yaw_rps', 'pitch': 'gyro_pitch_rps', 'roll': 'gyro_roll_rps'}
    if sorted(gyro_order) != ['pitch', 'roll', 'yaw']:
        raise ValueError('gyro_channel_order must contain yaw, pitch and roll once')
    gyro = phone[[channels[name] for name in gyro_order]].to_numpy()
    gravity = phone[['gravity_x_mps2', 'gravity_y_mps2', 'gravity_z_mps2']].to_numpy()
    # For linear acceleration, gravity must first share the acceleration frame.
    acc_rotation = proper_rotation(record['acceleration_to_body'])
    grav_body = gravity @ proper_rotation(record['gravity_to_body']).T
    acc_body, gyro_body = convert_imu(acc, gyro, acceleration_to_body=acc_rotation,
                                      gyro_to_body=record['gyro_to_body'],
                                      acceleration_kind=record['acceleration_kind'],
                                      gravity=grav_body @ acc_rotation)
    speed = speed_mps(phone.phone_speed_kmh.to_numpy(), record['phone_speed_unit'])
    bearing = np.deg2rad(phone.phone_bearing_deg.to_numpy())
    lat = vehicle.ref_lat_deg.to_numpy(); lon = vehicle.ref_lon_deg.to_numpy()
    reference = np.column_stack((6371000*np.deg2rad(lon-lon[0])*np.cos(np.deg2rad(lat[0])),
                                 6371000*np.deg2rad(lat-lat[0]), np.zeros(len(lat))))
    if record['reference_offset_s'] is None:
        raise ValueError('Missing independently verified clock offset')
    labels, valid = align_labels(tp, tv, reference, reference_offset_s=record['reference_offset_s'], max_gap_s=.15)
    velocity = np.column_stack((speed*np.sin(bearing), speed*np.cos(bearing), np.zeros(len(speed))))
    return tp, np.column_stack((gyro_body, acc_body)), labels, valid, velocity, bearing, grav_body


def prepare(contracts, out, seconds, stride):
    if out.exists():
        raise ValueError('Prepared datasets are immutable; use a new directory')
    config = json.loads(contracts.read_text())
    if config.get('schema') != 'idr-recording-contracts-v1':
        raise ValueError('Wrong recording contract schema')
    eligible = [r for r in config['records'] if r.get('status') in ('verified', 'provisional')]
    if {r['split'] for r in eligible} != {'train', 'validation'}:
        raise ValueError('No verified train/validation pair of partitions; do not launch training')
    if len({r['drive'] for r in eligible}) != len(eligible):
        raise ValueError('Duplicate recording identifiers')
    for row in eligible:
        row['dates'] = recording_dates(Path(row['phone_path']))
    assign_groups(eligible)
    groups = {}
    for r in eligible:
        for group in (r['session_group'], r['audited_session_group']):
            if group in groups and groups[group] != r['split']:
                raise ValueError('Source dates or duplicate recordings cross split boundaries')
            groups[group] = r['split']
    # Preserve the more conservative grouping from the complete source audit,
    # including any excluded recording that connects dates across midnight.
    for r in eligible:
        r['session_group'] = r['audited_session_group']
    # Stage all windows first, so a failed preflight does not leave a usable manifest.
    pending = []; rejected = []
    for r in eligible:
        try:
            t, imu, label, valid, velocity, bearing, gravity = load_record(r)
        except (ValueError, OSError) as error:
            rejected.append(dict(drive=r['drive'], reason='recording preflight: '+str(error)))
            continue
        length = round(seconds*10)+1
        for start in range(0, len(t)-length+1, round(stride*10)):
            end = start+length; segment = slice(start, end)
            reason = None
            if not valid[segment].all(): reason = 'unavailable reference labels'
            elif not np.allclose(np.diff(t[segment]), .1, atol=.01, rtol=0): reason = 'IMU gap or rate mismatch'
            elif np.linalg.norm(velocity[start]) < 2: reason = 'GNSS heading initialization unreliable at low speed'
            if reason:
                rejected.append(dict(drive=r['drive'], start=start, reason=reason)); continue
            attitude = initial_attitude(gravity[start], bearing[start])
            pending.append((r, start, dict(t=t[segment]-t[start], imu=imu[segment],
                                           reference_position=label[segment]-label[start],
                                           initial_velocity=velocity[start], initial_rpy=attitude)))
    if {r['split'] for r, _, _ in pending} != {'train', 'validation'}:
        raise ValueError('Not enough usable windows in both partitions')
    out.mkdir(parents=True)
    records = []
    for index, (r, start, window) in enumerate(pending):
        filename = f'{index:06d}.npz'; np.savez_compressed(out/filename, **window)
        records.append(dict(file=filename, sha256=sha256(out/filename), drive=r['drive'],
                            session_group=r['session_group'], split=r['split'], start_sample=start))
    status = 'verified' if all(r['status'] == 'verified' for r in eligible) else 'provisional'
    manifest = dict(schema='idr-trajectory-v1', status=status, sample_hz=10,
                    contracts_sha256=sha256(contracts), records=records, rejected=rejected,
                    limits=['Contracts retain their verified/provisional status; not production accuracy.',
                            'Phone GNSS seeds window velocity/heading; no reference enters filter initialization.',
                            'Initial tilt assumes gravity estimate is reliable; mounting must remain fixed.',
                            'No pristine final test partition is claimed.'])
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Prepared {len(records)} windows, rejected {len(rejected)} windows')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contracts', type=Path)
    parser.add_argument('--init-from-audit', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--stride', type=float, default=10)
    args = parser.parse_args()
    if bool(args.contracts) == bool(args.init_from_audit):
        parser.error('Specify exactly one of --contracts or --init-from-audit')
    if not 1 <= args.seconds <= 120 or not .1 <= args.stride <= 120:
        parser.error('seconds must be 1..120 and stride .1..120')
    if args.init_from_audit: initialize(args.init_from_audit, args.out)
    else: prepare(args.contracts, args.out, args.seconds, args.stride)
