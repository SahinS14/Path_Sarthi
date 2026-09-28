"""Validation-only diagnostic ESKF replay under explicit sensor hypotheses.

Not an acceptance benchmark: IO-VNBD sensor-frame conventions and absolute
reference alignment still require verification. No reference enters inference.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

from io_vnbd import find_pairs, load_drive, _read, _select, PHONE_COLUMNS
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'engine/src'))
from error_state_filter import ErrorStateFilter
from navigation_frame_filter import NavigationFrameFilter


def enu(lat,lon,lat0,lon0):
    return np.column_stack([6371000*np.deg2rad(lon-lon0)*np.cos(np.deg2rad(lat0)),
                            6371000*np.deg2rad(lat-lat0),np.zeros(len(lat))])


def initial_rotation(gravity, heading):
    # Establish tilt from measured gravity, then align vehicle forward to GNSS.
    z=gravity/np.linalg.norm(gravity)
    x=np.array([1.,0,0]); x-=z*(x@z); x/=np.linalg.norm(x)
    y=np.cross(z,x)
    level=np.vstack([x,y,z])
    yaw=np.pi/2-heading
    world=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    return world@level


def replay(pair, duration, mapping='legacy', initialization='warmup', calibrate_bias=False):
    f=load_drive(pair)
    raw=_select(_read(pair.phone_csv),PHONE_COLUMNS)
    t=f.elapsed_s.to_numpy(); t=t-t[0]
    end=int(np.searchsorted(t,20+duration)); start=int(np.searchsorted(t,20))
    if end>=len(t) or start<20: raise ValueError('insufficient duration')
    t=t[:end+1]; f=f.iloc[:end+1]; raw=raw.iloc[:end+1]
    if not ((np.diff(t)>.05)&(np.diff(t)<.15)).all(): raise ValueError('timestamp gap')
    acc=f[['acc_x_mps2','acc_y_mps2','acc_z_mps2']].to_numpy()
    grav=f[['gravity_x_mps2','gravity_y_mps2','gravity_z_mps2']].to_numpy()
    gyro=f[['gyro_roll_rps','gyro_pitch_rps','gyro_yaw_rps']].to_numpy()
    if mapping=='legacy':
        transform=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        acc=acc@transform.T; grav=grav@transform.T; gyro=gyro@transform.T
    elif mapping in ('documented-forward-yaw-only','navigation-frame'):
        # Figure 2 identifies +X as forward. Training diagnostics identify the
        # published Pitch channel with vehicle yaw. Other gyro axes remain
        # unresolved: explicitly test a reduced yaw-only hypothesis, not full INS.
        gyro=np.column_stack([np.zeros(len(t)),np.zeros(len(t)),f.gyro_pitch_rps.to_numpy()])
    else:
        raise ValueError('Unknown sensor mapping')
    heading=np.deg2rad(f.phone_bearing_deg.to_numpy())
    # Explicit hypothesis, not the legacy loader's label-derived unit heuristic.
    speed=raw.phone_speed_kmh.to_numpy()  # raw logged value interpreted as m/s
    velocity=np.column_stack([speed*np.sin(heading),speed*np.cos(heading),np.zeros(len(t))])
    lat0=f.phone_lat_deg.iloc[0]; lon0=f.phone_lon_deg.iloc[0]
    phone=enu(f.phone_lat_deg.to_numpy(),f.phone_lon_deg.to_numpy(),lat0,lon0)
    reference=enu(f.ref_lat_deg.to_numpy(),f.ref_lon_deg.to_numpy(),lat0,lon0)
    if not all(np.isfinite(a).all() for a in [acc,grav,gyro,heading,speed,phone,reference]):
        raise ValueError('nonfinite inputs')
    if np.linalg.norm(grav[0])<8 or np.linalg.norm(grav[0])>12: raise ValueError('invalid gravity norm')
    paths={}; timings={}
    modes=('navigation_no_nhc','navigation_fixed_nhc') if mapping=='navigation-frame' else ('eskf_no_nhc','eskf_fixed_nhc')
    for mode in modes:
        initial=start-1 if initialization=='outage-anchor' else 0
        is_navigation=mapping=='navigation-frame'
        k=(NavigationFrameFilter(phone[initial,:2],velocity[initial,:2],t[initial]) if is_navigation else
           ErrorStateFilter(phone[initial],velocity[initial],initial_rotation(grav[initial],heading[initial]),t[initial]))
        if calibrate_bias:
            if not is_navigation or initial<20: raise ValueError('Bias calibration requires navigation-frame outage-anchor initialization')
            # Integrate only pre-outage acceleration, compare against phone GNSS
            # velocity change over exactly the same interval. No vehicle labels.
            delta_t=np.diff(t[:initial+1])
            integrated=np.sum((acc[1:initial+1,:2]-grav[1:initial+1,:2])*delta_t[:,None],axis=0)
            k.x[4:6]=(integrated-(velocity[initial,:2]-velocity[0,:2]))/(t[initial]-t[0])
        position=lambda: np.r_[k.x[:2],0.] if is_navigation else k.p.copy()
        path=[phone[j].copy() for j in range(initial)]+[position()]; tick=time.perf_counter()
        for i in range(initial+1,len(t)):
            if is_navigation: k.predict(float(t[i]),(acc[i]-grav[i])[:2],gyro[i,2])
            else: k.predict(float(t[i]),acc[i],gyro[i])
            if mode.endswith('fixed_nhc'): k.update_nonholonomic()
            # At most nominal 1 Hz GNSS pre-outage; never any GNSS during outage.
            if i<start and i%10==0:
                sigma=float(f.phone_accuracy_m.iloc[i])
                if np.isfinite(sigma) and 0<sigma<=30:
                    if is_navigation: k.update_gnss(float(t[i]),phone[i,:2],velocity[i,:2],max(sigma,2),1.)
                    else: k.update_gnss(float(t[i]),phone[i],max(sigma,2),velocity[i],1.)
            path.append(position())
        timings[mode]=(time.perf_counter()-tick)*1000/(len(t)-initial-1)
        paths[mode]=np.array(path)
    anchor=start-1
    paths['constant_velocity']=phone[anchor]+(t-t[anchor])[:,None]*velocity[anchor]
    distance=float(np.linalg.norm(np.diff(reference[anchor:,:2],axis=0),axis=1).sum())
    metrics={}
    for mode,path in paths.items():
        displacement_error=(path[anchor:]-path[anchor])-(reference[anchor:]-reference[anchor])
        error=np.linalg.norm(displacement_error[:,:2],axis=1)
        metrics[mode]=dict(endpoint_relative_error_m=float(error[-1]),
                           max_relative_error_m=float(error.max()),
                           endpoint_absolute_error_m=float(np.linalg.norm(path[-1,:2]-reference[-1,:2])),
                           relative_drift_percent=float(100*error[-1]/distance) if distance>1 else None)
    return dict(drive=pair.name,outage_seconds=duration,reference_distance_m=distance,
                phone_reference_median_separation_m=float(np.median(np.linalg.norm(phone[:,:2]-reference[:,:2],axis=1))),
                raw_phone_speed_vs_reference_mae_mps=float(np.mean(np.abs(speed-f.ref_speed_kmh.to_numpy()/3.6))),
                desktop_ms_per_sample=timings,metrics=metrics),t,paths,reference,anchor


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mapping',choices=['legacy','documented-forward-yaw-only','navigation-frame'],default='legacy')
    p.add_argument('--initialization',choices=['warmup','outage-anchor'],default='warmup')
    p.add_argument('--calibrate-bias',action='store_true'); args=p.parse_args()
    if args.calibrate_bias and (args.mapping!='navigation-frame' or args.initialization!='outage-anchor'):
        p.error('--calibrate-bias requires navigation-frame and outage-anchor')
    manifest=json.loads(args.manifest.read_text())
    names={r['drive'] for r in manifest['records'] if r['split']=='validation'}
    args.out.mkdir(parents=True,exist_ok=True); results=[]
    for pair in sorted(find_pairs(args.dataset),key=lambda p:p.name):
        if pair.name not in names: continue
        for duration in (10,30,60):
            try:
                report,t,paths,ref,anchor=replay(pair,duration,args.mapping,args.initialization,args.calibrate_bias)
                np.savez_compressed(args.out/f'{pair.name}-{duration}.npz',t=t,reference=ref,anchor=anchor,**paths)
                results.append(report)
                key='navigation_fixed_nhc' if args.mapping=='navigation-frame' else 'eskf_fixed_nhc'
                print(pair.name,duration,report['metrics'][key]['relative_drift_percent'],flush=True)
            except ValueError as error: results.append(dict(drive=pair.name,outage_seconds=duration,skipped=str(error)))
    payload=dict(status='diagnostic; sensor mapping and speed units are hypotheses',
                 inference='phone-only initialization and pre-outage GNSS; IMU only during blackout; reference scoring only',
                 mapping=args.mapping,
                 initialization=args.initialization,
                 bias_calibration='pre-outage phone GNSS only' if args.calibrate_bias else 'none',
                 raw_phone_speed_unit='m/s hypothesis',results=results)
    (args.out/'report.json').write_text(json.dumps(payload,indent=2)+'\n')
