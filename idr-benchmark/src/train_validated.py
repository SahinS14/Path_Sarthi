"""Audited, session-grouped, resumable IO-VNBD speed-increment baseline.

This trains a candidate speed adapter, not a validated navigation product.
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from audit_pairs import audit
from io_vnbd import find_pairs, load_drive
from train_speed_model import FEATURES, SpeedTCN, windows


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(args):
    if args.out.exists():
        raise ValueError('Use a new output directory; prepared datasets are immutable')
    pairs = sorted(find_pairs(args.dataset), key=lambda p: p.name)
    accepted, rejected = [], []
    for pair in pairs:
        result = audit(pair)
        reasons = []
        if result['unequal_rows']:
            reasons.append('unequal paired row counts')
        if result['relative_clock_disagreement_max_s'] > .05:
            reasons.append('relative clocks disagree by >50 ms')
        if reasons:
            rejected.append(dict(drive=pair.name, reasons=reasons))
        else:
            accepted.append(pair)
    # Keep named subdivisions (e.g. Vw14a/b/c) in the same partition.
    group = lambda name: re.sub(r'(?<=\d)[a-z]$', '', name)
    groups = sorted({group(p.name) for p in accepted})
    random.Random(26168).shuffle(groups)
    n = max(1, round(len(groups) * .15))
    partitions = {g: ('test' if i < n else 'validation' if i < 2*n else 'train')
                  for i, g in enumerate(groups)}
    if len(groups) < 6:
        raise ValueError('Too few clean session groups')
    total, sums, squares = 0, np.zeros(len(FEATURES)), np.zeros(len(FEATURES))
    for pair in accepted:
        if partitions[group(pair.name)] != 'train':
            continue
        values = load_drive(pair)[FEATURES].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values).all(axis=1)]
        total += len(values); sums += values.sum(0); squares += (values**2).sum(0)
    mean = sums / total
    std = np.sqrt(np.maximum(squares/total - mean**2, 1e-8))
    args.out.mkdir(parents=True)
    records = []
    for i, pair in enumerate(accepted):
        split = partitions[group(pair.name)]
        x, y, _, _ = windows(load_drive(pair), mean, std, 20, 5, 'delta')
        if not len(y):
            rejected.append(dict(drive=pair.name, reasons=['no continuous finite windows']))
            continue
        filename = f'{i:03d}.npz'
        np.savez_compressed(args.out / filename, x=x.astype('float32'), y=y.astype('float32'))
        records.append(dict(drive=pair.name, group=group(pair.name), split=split,
                            file=filename, windows=len(y), sha256=digest(args.out/filename),
                            phone_sha256=digest(pair.phone_csv), vehicle_sha256=digest(pair.vehicle_csv)))
    manifest = dict(version=1, seed=26168, features=FEATURES, window=20,
                    target='speed change across trailing 1.9 seconds', stride=5,
                    mean=mean.tolist(), std=std.tolist(), records=records, rejected=rejected,
                    limitations=['Relative clock agreement does not prove absolute reference alignment.',
                                 'Session-name grouping is not verified driver independence.',
                                 'Existing IO-VNBD development exposure prevents claims of a pristine test set.'])
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({s: sum(r['windows'] for r in records if r['split']==s)
                      for s in ('train','validation','test')}, indent=2))
    print(f'Quarantined {len(rejected)} drives; details in manifest.json')


def read_manifest(root):
    manifest = json.loads((root/'manifest.json').read_text())
    for record in manifest['records']:
        if digest(root/record['file']) != record['sha256']:
            raise ValueError('Prepared data hash mismatch: '+record['file'])
    return manifest


def batches(root, manifest, split, shuffle=False):
    records = [r for r in manifest['records'] if r['split']==split]
    if shuffle:
        random.shuffle(records)
    for record in records:
        with np.load(root/record['file'], allow_pickle=False) as data:
            loader = DataLoader(TensorDataset(torch.from_numpy(data['x']), torch.from_numpy(data['y'])),
                                batch_size=256, shuffle=shuffle)
            yield record, loader


def score(model, root, manifest, split, device):
    model.eval(); reports = []
    with torch.no_grad():
        for record, loader in batches(root, manifest, split):
            count=0; absolute=squared=baseline=0.
            for x,y in loader:
                error=model(x.to(device)).cpu()-y
                count+=len(y); absolute+=error.abs().sum().item()
                squared+=(error**2).sum().item(); baseline+=y.abs().sum().item()
            reports.append(dict(drive=record['drive'], count=count, mae=absolute/count,
                                rmse=(squared/count)**.5, zero_change_mae=baseline/count))
    if not reports:
        raise ValueError('Empty evaluation split')
    return dict(macro_mae=float(np.mean([r['mae'] for r in reports])),
                macro_zero_change_mae=float(np.mean([r['zero_change_mae'] for r in reports])),
                drives=reports)


def save_checkpoint(path, payload):
    temporary=path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def train(args):
    manifest=read_manifest(args.data); fingerprint=digest(args.data/'manifest.json')
    if args.device=='mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS requested but unavailable; select --device cpu')
    device=torch.device(args.device)
    torch.set_num_threads(4); torch.manual_seed(26168); random.seed(26168)
    model=SpeedTCN(len(FEATURES)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    start=0; best=float('inf'); stale=0; history=[]
    if args.resume:
        saved=torch.load(args.out/'last.pt', map_location='cpu', weights_only=False)
        if saved['manifest_sha256']!=fingerprint or saved['device']!=args.device:
            raise ValueError('Resume requires the same dataset manifest and device')
        model.load_state_dict(saved['state_dict']); optimizer.load_state_dict(saved['optimizer'])
        start=saved['epoch']; best=saved['best']; stale=saved['stale']; history=saved['history']
        random.setstate(saved['python_rng']); torch.set_rng_state(saved['torch_rng'])
        if args.device=='mps': torch.mps.set_rng_state(saved['mps_rng'])
    elif args.out.exists():
        raise ValueError('Use --resume or a new run directory')
    args.out.mkdir(parents=True, exist_ok=True)
    for epoch in range(start+1, args.epochs+1):
        model.train(); total=0; loss_sum=0.
        for _,loader in batches(args.data,manifest,'train',True):
            for x,y in loader:
                optimizer.zero_grad()
                loss=torch.nn.functional.smooth_l1_loss(model(x.to(device)),y.to(device))
                if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
                loss_sum+=loss.item()*len(y); total+=len(y)
        validation=score(model,args.data,manifest,'validation',device)
        improved=validation['macro_mae']<best
        stale=0 if improved else stale+1
        if improved: best=validation['macro_mae']
        history.append(dict(epoch=epoch,train_huber=loss_sum/total,validation=validation))
        payload=dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
                     best=best,stale=stale,history=history,manifest_sha256=fingerprint,device=args.device,
                     features=FEATURES,window=20,mean=manifest['mean'],std=manifest['std'],
                     python_rng=random.getstate(),torch_rng=torch.get_rng_state(),
                     mps_rng=torch.mps.get_rng_state() if args.device=='mps' else None)
        save_checkpoint(args.out/'last.pt',payload)
        if improved: save_checkpoint(args.out/'best.pt',payload)
        (args.out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        print(f"epoch {epoch}: validation MAE {validation['macro_mae']:.4f}, zero-change {validation['macro_zero_change_mae']:.4f}",flush=True)
        if stale>=args.patience:
            print('Validation early stopping'); break


def evaluate(args):
    manifest=read_manifest(args.data)
    saved=torch.load(args.model,map_location='cpu',weights_only=False)
    if saved['manifest_sha256']!=digest(args.data/'manifest.json'):
        raise ValueError('Checkpoint/data mismatch')
    model=SpeedTCN(len(FEATURES)); model.load_state_dict(saved['state_dict'])
    report=score(model,args.data,manifest,'test','cpu')
    report['scope']='Speed-increment test only; not a position-drift acceptance test.'
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare'); p.add_argument('--dataset',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p=commands.add_parser('train'); p.add_argument('--data',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=100); p.add_argument('--patience',type=int,default=12)
    p.add_argument('--device',choices=['cpu','mps'],default='cpu'); p.add_argument('--resume',action='store_true')
    p=commands.add_parser('evaluate'); p.add_argument('--data',type=Path,required=True); p.add_argument('--model',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='train' and (args.epochs<1 or args.patience<1): parser.error('epochs and patience must be positive')
    dict(prepare=prepare,train=train,evaluate=evaluate)[args.command](args)
