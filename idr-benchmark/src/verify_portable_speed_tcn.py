"""Numerically verify the framework-free Android TCN equations against PyTorch."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from train_speed_model import FEATURES, SpeedTCN


def portable(window: np.ndarray, artifact: dict) -> float:
    state = artifact["state_dict"]; x = (window - np.asarray(artifact["mean"], np.float32)) / np.asarray(artifact["std"], np.float32)
    w1=state["network.0.weight"].numpy(); b1=state["network.0.bias"].numpy(); wd=state["network.2.weight"].numpy()[:,0]; bd=state["network.2.bias"].numpy(); linear=state["network.6.weight"].numpy()[0]; bias=float(state["network.6.bias"].numpy()[0])
    first=np.zeros((24,20),np.float32)
    for channel in range(24):
        for t in range(20):
            value=b1[channel]
            for feature in range(9):
                for k in range(5):
                    source=t+k-2
                    if 0<=source<20:value+=w1[channel,feature,k]*x[source,feature]
            first[channel,t]=max(0,value)
    output=bias
    for channel in range(24):
        values=[]
        for t in range(20):
            value=bd[channel]
            for k in range(5):
                source=t+k-2
                if 0<=source<20:value+=wd[channel,k]*first[channel,source]
            values.append(max(0,value))
        output+=linear[channel]*np.mean(values)
    return float(output)


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("--model",type=Path,required=True);args=parser.parse_args()
    artifact=torch.load(args.model,map_location="cpu",weights_only=False); model=SpeedTCN(len(FEATURES));model.load_state_dict(artifact["state_dict"]);model.eval(); rng=np.random.default_rng(26168)
    raw=rng.normal(np.asarray(artifact["mean"]),np.asarray(artifact["std"]),(32,20,9)).astype(np.float32)
    normal=(raw-np.asarray(artifact["mean"],np.float32))/np.asarray(artifact["std"],np.float32)
    with torch.no_grad(): torch_output=model(torch.tensor(normal)).numpy()
    port_output=np.asarray([portable(window,artifact) for window in raw]); error=float(np.max(np.abs(torch_output-port_output)))
    print(f"Windows checked: {len(raw)}\nMaximum PyTorch/portable difference: {error:.8f} m/s")
    if error > 1e-5: raise RuntimeError("Portable inference mismatch")
    return 0


if __name__ == "__main__": raise SystemExit(main())
