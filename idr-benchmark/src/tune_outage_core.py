"""Select conservative outage-core parameters using train dates, then report validation dates.

This is a deterministic grid search over two physical safeguards, not an ML
training result. It never uses vehicle data inside filter inference.
"""
import argparse
import json
from pathlib import Path

from benchmark_blackouts import run_drive, summarize
from io_vnbd import find_pairs


def score(rows):
    values=[r['filter_drift_percent'] for r in rows if r['reference_distance_m'] >= 1]
    return float('inf') if not values else sum(values)/len(values)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--sessions',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--outage-s',type=float,default=30)
    args=parser.parse_args()
    if args.out.exists(): parser.error('Use a new output filename')
    session=json.loads(args.sessions.read_text())
    splits={r['drive']:r['proposed_split'] for r in session['records']}
    choices=[(gain, anchor) for gain in (0., .1, .25, .5, .75, 1.) for anchor in (None, 1., 3., 10., 30.)]
    pairs=[p for p in find_pairs(args.dataset) if p.name in splits]
    trials=[]
    for gain, anchor in choices:
        rows=[]
        for pair in pairs:
            if splits[pair.name] != 'train': continue
            try: rows.append(run_drive(pair,20,args.outage_s,gain,anchor))
            except ValueError: pass
        trials.append(dict(outage_acceleration_gain=gain,speed_anchor_time_s=anchor,train_mean_drift_percent=score(rows),cases=len(rows)))
    best=min(trials,key=lambda x:x['train_mean_drift_percent'])
    validation=[]; default=[]
    for pair in pairs:
        if splits[pair.name] != 'validation': continue
        try:
            validation.append(run_drive(pair,20,args.outage_s,best['outage_acceleration_gain'],best['speed_anchor_time_s']))
            default.append(run_drive(pair,20,args.outage_s,1.,None))
        except ValueError: pass
    report=dict(scope='Provisional session-group split; vehicle labels score only after each outage.',outage_seconds=args.outage_s,
                selected=best,validation_selected=summarize(validation),validation_default=summarize(default),trials=trials,
                limitations=['Only two scalar safeguards are tuned.', 'No final test or PS-compliance claim.'])
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(selected=best,validation_selected=report['validation_selected'],validation_default=report['validation_default']),indent=2))

if __name__=='__main__': main()
