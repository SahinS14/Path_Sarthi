"""Create an auditable IO-VNBD exploratory contract from documented fields.

This is intentionally *provisional*: it makes the supplied synchronized dataset
trainable for development experiments, but it does not certify field frames or
production accuracy.  It does not alter source CSVs.
"""
import argparse
import json
from pathlib import Path


LEGACY_BODY = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]


def build(audit):
    source = json.loads(audit.read_text())
    records = []
    for row in source['records']:
        # We preserve the audit's day-level partition and only use paired rows
        # already accepted by the original synchronized collection audit.
        records.append(dict(
            drive=row['drive'], phone_path=row['phone_path'], vehicle_path=row['vehicle_path'],
            phone_sha256=row['phone_sha256'], vehicle_sha256=row['vehicle_sha256'],
            audited_session_group=row['session_group'], split=row['proposed_split'],
            status='provisional',
            evidence=('IO-VNBD synchronized pair; direct 10 Hz elapsed-clock agreement; '
                      'raw phone GPS-speed channel empirically behaves as m/s; '
                      'legacy proper body rotation retained as an exploratory convention.'),
            phone_speed_unit='m/s',
            acceleration_to_body=LEGACY_BODY, gravity_to_body=LEGACY_BODY,
            gyro_to_body=LEGACY_BODY, gyro_channel_order=['roll', 'pitch', 'yaw'],
            acceleration_kind='specific_force',
            reference_offset_s=0.0,
        ))
    return dict(schema='idr-recording-contracts-v1', contract_status='provisional', records=records,
                limitations=['This contract enables exploratory training only.',
                             'It reuses the synchronized-pair relative clock, not an independently measured absolute sensor latency.',
                             'Sensor frame convention must be revised after trajectory evidence or dataset-author clarification.',
                             'Do not claim PS compliance from this dataset-only training.'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args=parser.parse_args()
    if args.out.exists(): parser.error('Use a new output path')
    payload=build(args.audit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps(dict(records=len(payload['records']), status=payload['contract_status']), indent=2))
