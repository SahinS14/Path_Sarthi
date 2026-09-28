"""Train a causal phone-IMU absolute-speed estimator on prepared IO-VNBD windows.

The input at time i contains samples i-19..i only. Vehicle-reference positions
create offline speed labels only; they never enter model inference. The manifest
already preserves proposed recording-day groups between train and validation.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from train_speed_model import SpeedTCN


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root, stride):
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('schema')!='idr-trajectory-v1' or manifest.get('status') not in ('provisional','verified'):
        raise ValueError('Expected prepared IO-VNBD trajectory data')
    data={split:[] for split in ('train','validation')}
    for record in manifest['records']:
        if digest(root/record['file']) != record['sha256']: raise ValueError('Prepared-window hash mismatch')
        with np.load(root/record['file'],allow_pickle=False) as z:
            imu=z['imu'].astype(np.float32); t=z['t'].astype(float); p=z['reference_position'].astype(float); initial=z['initial_velocity'].astype(float)
        if len(imu)<20 or imu.shape[1]!=6 or np.any(np.diff(t)<=0): raise ValueError('Invalid trajectory window')
        # The model consumes gyro xyz + specific-force xyz. Use a centered
        # 0.2-second reference displacement for robust offline speed labels.
        indexes=np.arange(19,len(imu)-1,stride)
        velocity=np.linalg.norm((p[indexes+1]-p[indexes-1])/(t[indexes+1,None]-t[indexes-1,None]),axis=1)
        valid=np.isfinite(imu).all(axis=1) & np.isfinite(p).all(axis=1)
        keep=np.array([valid[i-19:i+2].all() for i in indexes]) & np.isfinite(velocity)
        if not keep.any(): continue
        x=np.stack([imu[i-19:i+1] for i in indexes[keep]])
        data[record['split']].append((record['drive'],x,velocity[keep].astype(np.float32),float(np.linalg.norm(initial))))
    if not data['train'] or not data['validation']: raise ValueError('No usable train/validation samples')
    return manifest,data


def normalize(chunks):
    values=np.concatenate([x.reshape(-1,6) for _,x,_,_ in chunks])
    return values.mean(0).astype(np.float32),np.maximum(values.std(0),1e-3).astype(np.float32)


def array(chunks,mean,std):
    x=np.concatenate([part for _,part,_,_ in chunks]);y=np.concatenate([target for _,_,target,_ in chunks])
    return ((x-mean)/std).astype(np.float32),y.astype(np.float32)


def score(model,chunks,mean,std,device):
    rows=[];model.eval()
    with torch.no_grad():
        for drive,x,y,initial in chunks:
            prediction=model(torch.from_numpy((x-mean)/std).to(device)).cpu().numpy()
            rows.append(dict(drive=drive,samples=len(y),mae_mps=float(np.abs(prediction-y).mean()),
                             initial_speed_baseline_mae_mps=float(np.abs(initial-y).mean())))
    unique=sorted({r['drive'] for r in rows})
    return dict(macro_mae_mps=float(np.mean([np.mean([r['mae_mps'] for r in rows if r['drive']==d]) for d in unique])),
                macro_initial_speed_baseline_mae_mps=float(np.mean([np.mean([r['initial_speed_baseline_mae_mps'] for r in rows if r['drive']==d]) for d in unique])),drives=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=40);p.add_argument('--patience',type=int,default=8);p.add_argument('--stride',type=int,default=5)
    p.add_argument('--device',choices=('cpu','mps'),default='cpu');p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.epochs<1 or a.patience<1 or a.stride<1: p.error('epochs, patience and stride must be positive')
    if a.device=='mps' and not torch.backends.mps.is_available(): p.error('MPS unavailable')
    manifest,chunks=build(a.data,a.stride);mean,std=normalize(chunks['train']);x,y=array(chunks['train'],mean,std)
    identity=dict(data=digest(a.data/'manifest.json'),epochs_stride=a.stride,device=a.device,torch=torch.__version__)
    torch.manual_seed(26168);random.seed(26168);torch.set_num_threads(2);device=torch.device(a.device)
    model=SpeedTCN(6).to(device);optim=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    begin=0;best=float('inf');stale=0;history=[]
    if a.resume:
        saved=torch.load(a.out/'last.pt',map_location='cpu',weights_only=False)
        if saved['identity']!=identity: raise ValueError('Resume requires matching data, stride, device and PyTorch version')
        model.load_state_dict(saved['state_dict']);optim.load_state_dict(saved['optimizer']);begin=saved['epoch'];best=saved['best'];stale=saved['stale'];history=saved['history'];random.setstate(saved['python_rng']);torch.set_rng_state(saved['torch_rng'])
        if a.device=='mps':torch.mps.set_rng_state(saved['mps_rng'])
    elif a.out.exists(): raise ValueError('Use a new output folder or --resume')
    a.out.mkdir(parents=True,exist_ok=True)
    loader=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),batch_size=512,shuffle=True)
    for epoch in range(begin+1,a.epochs+1):
        model.train();losses=[]
        for bx,by in loader:
            optim.zero_grad();loss=torch.nn.functional.smooth_l1_loss(model(bx.to(device)),by.to(device));loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step();losses.append(loss.item())
        validation=score(model,chunks['validation'],mean,std,device);value=validation['macro_mae_mps'];improved=value<best;best=value if improved else best;stale=0 if improved else stale+1
        history.append(dict(epoch=epoch,train_huber=float(np.mean(losses)),validation=validation));payload=dict(state_dict=model.state_dict(),optimizer=optim.state_dict(),epoch=epoch,best=best,stale=stale,history=history,mean=mean.tolist(),std=std.tolist(),window=20,features=['gyro_x','gyro_y','gyro_z','specific_force_x','specific_force_y','specific_force_z'],target='absolute vehicle speed m/s',identity=identity,python_rng=random.getstate(),torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state() if a.device=='mps' else None)
        torch.save(payload,a.out/'last.pt')
        if improved:torch.save(payload,a.out/'best.pt');(a.out/'best-validation.json').write_text(json.dumps(validation,indent=2)+'\n')
        (a.out/'history.json').write_text(json.dumps(history,indent=2)+'\n');print(f"epoch {epoch}: validation MAE {value:.4f} m/s, initial-speed {validation['macro_initial_speed_baseline_mae_mps']:.4f}",flush=True)
        if stale>=a.patience:print('Validation early stopping');break

if __name__=='__main__':main()
