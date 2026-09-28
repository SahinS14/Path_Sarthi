"""Inventory synchronized IO-VNBD without training or altering source files."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def inventory(root):
    records = []
    for path in sorted(root.rglob('*.csv')):
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(block)
        with path.open(encoding='latin-1', newline='') as source:
            rows = csv.reader(source)
            header = next(rows, [])
            count = sum(1 for row in rows if row)
        records.append(dict(path=str(path.relative_to(root)), bytes=path.stat().st_size,
                            rows=count, columns=len(header), sha256=digest.hexdigest()))
    groups = {}
    for record in records:
        groups.setdefault(record['sha256'], []).append(record['path'])
    return dict(root=str(root.resolve()), csv_count=len(records),
                unique_contents=len(groups), total_bytes=sum(r['bytes'] for r in records),
                duplicate_groups=[v for v in groups.values() if len(v) > 1], files=records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not args.dataset.is_dir():
        parser.error('Dataset directory does not exist')
    report = inventory(args.dataset)
    if not report['csv_count']:
        parser.error('No CSV files found')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('files', 'duplicate_groups')}, indent=2))
