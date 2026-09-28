"""Run the actual upstream IEKF on deterministic synthetic vehicle motion.

This is a compatibility/latency check, not a KITTI or IO-VNBD accuracy result.
Upstream source and MIT notice are retained unchanged under work/reference.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np


def benchmark(source, hz, seconds):
    sys.path.insert(0,str(source/'src'))
    from utils_numpy_filter import NUMPYIEKF
    n=round(hz*seconds)+1; t=np.arange(n)/hz; results=[]
    for name,speed,rate in [('stationary',0.,0.),('straight',10.,0.),('turn',10.,.05)]:
        u=np.zeros((n,6)); u[:,2]=rate; u[:,4]=speed*rate; u[:,5]=9.80665
        velocity=np.zeros((n,3)); velocity[0,0]=speed
        truth=np.zeros((n,3))
        if rate:
            truth[:,0]=speed/rate*np.sin(rate*t); truth[:,1]=speed/rate*(1-np.cos(rate*t))
        else: truth[:,0]=speed*t
        f=NUMPYIEKF(); cov=np.tile([f.cov_lat,f.cov_up],(n,1))
        start=time.perf_counter()
        rotations,v,p,*_=f.run(t,u,cov,velocity,np.zeros((n,3)),n,np.zeros(3))
        elapsed=time.perf_counter()-start
        error=np.linalg.norm(p-truth,axis=1)
        if not np.isfinite(p).all(): raise ValueError('Upstream produced nonfinite state')
        results.append(dict(scenario=name,samples=n,seconds=seconds,hz=hz,
                            endpoint_error_m=float(error[-1]),max_error_m=float(error.max()),
                            cpu_ms_per_sample=1000*elapsed/(n-1),wall_seconds=elapsed,
                            rotation_orthogonality_max=float(np.max(np.abs(np.swapaxes(rotations,1,2)@rotations-np.eye(3))))))
    path=source/'src/utils_numpy_filter.py'
    return dict(scope='Synthetic compatibility/CPU benchmark; upstream fixed covariance, no learned weights',
                source='https://github.com/mbrossar/ai-imu-dr',
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),results=results)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('work/reference/ai-imu-dr'))
    p.add_argument('--hz',type=int,default=100); p.add_argument('--seconds',type=float,default=30)
    p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    if a.hz<1 or not np.isfinite(a.seconds) or a.seconds<=0: p.error('Invalid duration/rate')
    report=benchmark(a.source,a.hz,a.seconds)
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
