"""Check IO-VNBD-trained speed adapter on unseen real phone logs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_speed_model import FEATURES, SpeedTCN


PHONE_FEATURES = ["acc_x", "acc_y", "acc_z", "gravity_x", "gravity_y", "gravity_z", "gyro_z", "gyro_y", "gyro_x"]


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--model",type=Path,required=True); parser.add_argument("--phone-log",type=Path,required=True); parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(); artifact=torch.load(args.model,map_location="cpu",weights_only=False); model=SpeedTCN(len(FEATURES)); model.load_state_dict(artifact["state_dict"]); model.eval()
    data=pd.read_csv(args.phone_log,na_values=["null"]); values=data[PHONE_FEATURES].apply(pd.to_numeric,errors="coerce").to_numpy(np.float32); speed=pd.to_numeric(data["gnss_speed_mps"],errors="coerce").to_numpy(np.float32)
    valid=np.isfinite(values).all(1)&np.isfinite(speed); values=values[valid];speed=speed[valid]; mean=np.asarray(artifact["mean"],np.float32);std=np.asarray(artifact["std"],np.float32); values=(values-mean)/std
    indexes=np.arange(19,len(values)); x=np.stack([values[i-19:i+1] for i in indexes]); actual=speed[indexes]-speed[indexes-19]
    with torch.no_grad(): predicted=model(torch.tensor(x)).numpy()
    baseline=float(np.abs(actual).mean()); learned=float(np.abs(predicted-actual).mean()); args.out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame({"actual_delta_mps":actual,"predicted_delta_mps":predicted}).to_csv(args.out,index=False)
    print(f"Phone log: {args.phone_log.name}\nZero-change MAE: {baseline:.4f} m/s\nIO-VNBD model MAE: {learned:.4f} m/s\nImprovement: {(baseline-learned)/baseline*100:.2f}%")
    return 0


if __name__=="__main__":sys.exit(main())
