"""Metadata-only, conservative recording-day groups; never claims fresh test data.

No sensor values are used to choose partitions. Existing raw files and prepared-v1
remain unchanged. Missing dates fail closed, rather than becoming unique sessions.
"""
import argparse
import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def recording_dates(path):
    dates = set()
    with path.open(encoding='latin-1', newline='') as stream:
        reader = csv.DictReader(stream)
        columns = [c for c in reader.fieldnames or [] if c.strip().startswith('DATE (')]
        if len(columns) != 1:
            raise ValueError('Missing or ambiguous DATE column')
        for row in reader:
            value = row[columns[0]].strip()
            # Validate the full timestamp, not merely an ISO-looking prefix.
            stamp = datetime.strptime(value, '%Y-%m-%d %H:%M:%S:%f')
            dates.add(stamp.date().isoformat())
    if not dates:
        raise ValueError('Empty recording')
    return sorted(dates)


def assign_groups(records):
    """Connected components of dates, subdivisions and duplicate source hashes."""
    parent = list(range(len(records)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    for i, record in enumerate(records):
        keys = [('date', d) for d in record['dates']]
        keys += [('family', re.sub(r'(?<=\d)[a-z]$', '', record['drive']))]
        keys += [('hash', record[k]) for k in ('phone_sha256', 'vehicle_sha256')]
        for key in keys:
            if key in owners:
                parent[root(i)] = root(owners[key])
            owners[key] = i
    components = {}
    for i, record in enumerate(records):
        components.setdefault(root(i), []).append(record)
    for members in components.values():
        identity = '|'.join(sorted(r['drive'] for r in members))
        group = hashlib.sha256(identity.encode()).hexdigest()[:16]
        for record in members:
            record['session_group'] = group
    return records


def build(dataset, historical):
    old = {r['drive']: r['split'] for r in historical['records']}
    records, rejected = [], []
    for directory in sorted(dataset.rglob('*')):
        if not directory.is_dir():
            continue
        phones, vehicles = list(directory.glob('S-*.csv')), list(directory.glob('V-*.csv'))
        if len(phones) != 1 or len(vehicles) != 1:
            continue
        try:
            records.append(dict(drive=directory.name, dates=recording_dates(phones[0]),
                                phone_sha256=sha256(phones[0]), vehicle_sha256=sha256(vehicles[0]),
                                historical_split=old.get(directory.name, 'not_in_prepared_v1'),
                                phone_path=str(phones[0]), vehicle_path=str(vehicles[0]),
                                interpretation_status='unverified', training_eligible=False))
        except (ValueError, KeyError) as error:
            rejected.append(dict(drive=directory.name, reason=str(error)))
    if not records:
        raise ValueError('No date-verified paired recordings')
    if len({r['drive'] for r in records}) != len(records):
        raise ValueError('Duplicate drive identifiers; select the categorized dataset root')
    assign_groups(records)
    groups = sorted({r['session_group'] for r in records})
    # Deterministic prospective development split, never relabelled pristine test.
    validation = set(sorted(groups, key=lambda g: hashlib.sha256(('26168:'+g).encode()).hexdigest())[:max(1, round(.2*len(groups)))] )
    for r in records:
        r['proposed_split'] = 'validation' if r['session_group'] in validation else 'train'
    leakage = []
    for group in groups:
        members = [r for r in records if r['session_group'] == group]
        splits = sorted({r['historical_split'] for r in members} - {'not_in_prepared_v1'})
        if len(splits) > 1:
            leakage.append(dict(group=group, historical_splits=splits, drives=[r['drive'] for r in members]))
    return dict(version=1, records=records, rejected=rejected,
                session_group_count=len(groups), historical_cross_split_groups=leakage,
                pristine_test_available=False,
                limitations=['Day grouping is conservative, not proof of independent drivers or routes.',
                             'All local recordings have prior development/audit exposure of varying extent.',
                             'Proposed partitions do not authorize training: sensor/time contracts remain unverified.',
                             'Original unsynchronized copies must inherit their source recording group.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--historical', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = build(args.dataset, json.loads(args.historical.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(recordings=len(report['records']), groups=report['session_group_count'],
                         rejected=len(report['rejected']), historical_cross_split_groups=len(report['historical_cross_split_groups'])), indent=2))
