"""Compare body versus earth acceleration hypotheses using training data only."""
import argparse
import json
from pathlib import Path
import numpy as np
from io_vnbd import find_pairs, _read, _select, PHONE_COLUMNS, VEHICLE_COLUMNS
from diagnose_conventions import smooth


def fit(x, y):
    good=np.isfinite(x).all(axis=1)&np.isfinite(y).all(axis=1)
    x=x[good]; y=y[good]
    if len(x)<100: return None
    # Fit the first half and score the second; avoid selecting on the same rows.
    n=len(x)//2
    W=np.linalg.lstsq(np.c_[x[:n],np.ones(n)],y[:n],rcond=None)[0]
    pred=np.c_[x[n:],np.ones(len(x)-n)]@W
    mse=np.mean((pred-y[n:])**2)
    baseline=np.mean((y[n:]-y[:n].mean(0))**2)
    return dict(heldout_r2=float(1-mse/max(baseline,1e-10)),matrix=W.tolist())


def diagnose(pair, lag):
    s=_select(_read(pair.phone_csv),PHONE_COLUMNS)
    v=_select(_read(pair.vehicle_csv),VEHICLE_COLUMNS)
    t=s.phone_time_ms.to_numpy()/1000
    if len(s)!=len(v) or not ((np.diff(t)>.05)&(np.diff(t)<.15)).all():
        raise ValueError('Rows or clocks unsuitable')
    speed=v.ref_speed_kmh.to_numpy()/3.6
    bearing=np.unwrap(np.deg2rad(v.ref_heading_deg.to_numpy()))
    acceleration=np.column_stack([smooth((s[f'acc_{k}_mps2']-s[f'gravity_{k}_mps2']).to_numpy(),21) for k in ('x','y')])
    forward=smooth(np.gradient(speed,t),21)
    lateral=smooth(-speed*np.gradient(bearing,t),21)
    body=np.column_stack([forward,lateral])
    world=np.column_stack([smooth(np.gradient(speed*np.sin(bearing),t),21),smooth(np.gradient(speed*np.cos(bearing),t),21)])
    # Lag is the previously diagnosed pitch/vehicle yaw lag, not re-fit on validation.
    margin=abs(lag)+30
    i=np.arange(margin,len(s)-margin,5)
    return dict(drive=pair.name,lag_samples=lag,body=fit(acceleration[i],body[i+lag]),earth=fit(acceleration[i],world[i+lag]))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True); p.add_argument('--conventions',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); records=json.loads(a.conventions.read_text())['drives']
    allowed={r['drive']:r for r in records if 'gyro_vs_vehicle_yaw' in r}
    results=[]
    for pair in find_pairs(a.dataset):
        if pair.name not in allowed: continue
        info=allowed[pair.name]['gyro_vs_vehicle_yaw']['pitch']
        if not info or info['correlation']<.7 or abs(info['lag_samples'])>=50: continue
        try: results.append(diagnose(pair,info['lag_samples']))
        except ValueError as e: results.append(dict(drive=pair.name,skipped=str(e)))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(dict(scope='training drives only; exploratory frame hypotheses',results=results),indent=2)+'\n')
    for r in results:
        print(r['drive'],{k:round(r[k]['heldout_r2'],3) if r.get(k) else None for k in ('body','earth')})
