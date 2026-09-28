"""Training-partition-only unit, time-lag and IMU channel diagnostics.

Fits are diagnostic hypotheses, never automatic production calibration.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from io_vnbd import find_pairs, _read, _select, PHONE_COLUMNS, VEHICLE_COLUMNS


def smooth(x, width=11):
    return np.convolve(x, np.ones(width)/width, mode='same')


def lag_fit(source, target, max_lag=50):
    fits=[]
    for lag in range(-max_lag,max_lag+1):
        i=np.arange(max_lag+11,len(source)-max_lag-11,5)
        x=source[i]; y=target[i+lag]
        valid=np.isfinite(x)&np.isfinite(y); x=x[valid]; y=y[valid]
        if len(x)<50 or np.std(x)<1e-5 or np.std(y)<1e-5: continue
        corr=float(np.corrcoef(x,y)[0,1])
        scale,bias=np.linalg.lstsq(np.column_stack([x,np.ones(len(x))]),y,rcond=None)[0]
        fits.append(dict(lag_samples=lag,correlation=corr,scale=float(scale),bias=float(bias)))
    return max(fits,key=lambda r:abs(r['correlation'])) if fits else None


def diagnose(pair):
    original=_read(pair.phone_csv); phone=_select(original,PHONE_COLUMNS)
    vehicle=_select(_read(pair.vehicle_csv),VEHICLE_COLUMNS)
    if len(phone)!=len(vehicle): raise ValueError('Unequal row counts')
    t=phone.phone_time_ms.to_numpy()/1000
    raw=phone.phone_speed_kmh.to_numpy(); ref=vehicle.ref_speed_kmh.to_numpy()/3.6
    intervals=np.diff(t)
    if not ((intervals>.05)&(intervals<.15)).all(): raise ValueError('Noncontinuous phone clock')
    if not np.isfinite(ref).all() or not np.isfinite(raw).all(): raise ValueError('Nonfinite speed')
    # Compare units using the same receiver's coordinate displacement as well as vehicle speed.
    lat=phone.phone_lat_deg.to_numpy(); lon=phone.phone_lon_deg.to_numpy()
    xy=np.column_stack([6371000*np.deg2rad(lon-lon[0])*np.cos(np.deg2rad(lat[0])),6371000*np.deg2rad(lat-lat[0])])
    step=20
    coordinate_speed=np.linalg.norm(xy[step:]-xy[:-step],axis=1)/(t[step:]-t[:-step])
    average_raw=(raw[step:]+raw[:-step])*.5
    valid=np.isfinite(coordinate_speed)&(coordinate_speed>2)&(average_raw>2)
    speed_units={unit:dict(vehicle_mae_mps=float(np.mean(np.abs(raw*factor-ref))),
                         phone_coordinate_mae_mps=float(np.mean(np.abs(average_raw[valid]*factor-coordinate_speed[valid]))) if valid.any() else None)
                 for unit,factor in [('m/s',1.),('km/h',1/3.6)]}
    acc_target=smooth(np.gradient(ref,t))
    yaw_target=smooth(np.deg2rad(vehicle.ref_yaw_rate_dps.to_numpy()))
    gyro={axis:lag_fit(smooth(phone[f'gyro_{axis}_rps'].to_numpy()),yaw_target) for axis in ('yaw','pitch','roll')}
    accel={axis:lag_fit(smooth((phone[f'acc_{axis}_mps2']-phone[f'gravity_{axis}_mps2']).to_numpy()),acc_target) for axis in ('x','y','z')}
    return dict(drive=pair.name,recording_date=str(original.filter(regex='DATE').iloc[0,0])[:10],
                speed_units=speed_units,speed_alignment=lag_fit(smooth(raw),smooth(ref)),
                gyro_vs_vehicle_yaw=gyro,accel_vs_vehicle_longitudinal=accel,
                gravity_mean=phone[['gravity_x_mps2','gravity_y_mps2','gravity_z_mps2']].mean().tolist())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); m=json.loads(a.manifest.read_text())
    names={r['drive'] for r in m['records'] if r['split']=='train'}
    reports=[]
    for pair in sorted(find_pairs(a.dataset),key=lambda p:p.name):
        if pair.name not in names: continue
        try: reports.append(diagnose(pair))
        except ValueError as e: reports.append(dict(drive=pair.name,skipped=str(e)))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(dict(scope='training partition only; diagnostic fits not deployed',drives=reports),indent=2)+'\n')
    for r in reports:
        if 'skipped' in r: continue
        best=lambda d:max(((k,v) for k,v in d.items() if v),key=lambda kv:abs(kv[1]['correlation']),default=(None,None))
        print(r['drive'],'speed',r['speed_alignment'],'gyro',best(r['gyro_vs_vehicle_yaw']),'acc',best(r['accel_vs_vehicle_longitudinal']),flush=True)
