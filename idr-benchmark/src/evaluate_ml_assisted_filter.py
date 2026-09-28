"""Compare vehicle-state filtering with and without IO-VNBD ML confidence gating."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine" / "src"))
from vehicle_state_filter import VehicleStateFilter
from evaluate_fusion import fit_phone_alignment
from io_vnbd import choose_pair, find_pairs, load_drive
from profile_drive import local_xy
from train_speed_model import FEATURES, SpeedTCN


def run(frame, outage, forward, yaw, confidence, t, phone_e, phone_n, heading):
    engine=VehicleStateFilter(); east=[];north=[]; stationary=((np.abs(forward)<.12)&(np.abs(yaw)<.04)).astype(float)
    for i in range(len(frame)):
        gnss={} if outage[i] else dict(gnss_east_m=float(phone_e[i]),gnss_north_m=float(phone_n[i]),gnss_speed_mps=float(frame.phone_speed_kmh.iloc[i]/3.6),gnss_heading_rad=float(heading[i]),gnss_accuracy_m=float(frame.phone_accuracy_m.iloc[i]))
        state=engine.step(float(t[i]),float(forward[i]),float(yaw[i]),float(confidence[i]),float(stationary[i]),**gnss);east.append(state.east_m);north.append(state.north_m)
    return np.asarray(east),np.asarray(north)


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("--dataset",type=Path,required=True);parser.add_argument("--model",type=Path,required=True);parser.add_argument("--drive",default="Vta10");parser.add_argument("--outage-start-s",type=float,default=20);parser.add_argument("--outage-s",type=float,default=5);args=parser.parse_args()
    frame=load_drive(choose_pair(find_pairs(args.dataset),args.drive)).dropna().reset_index(drop=True);t=frame.elapsed_s.to_numpy();rel=t-t[0];outage=(rel>=args.outage_start_s)&(rel<args.outage_start_s+args.outage_s)
    ref_e,ref_n=local_xy(frame.ref_lat_deg.to_numpy(),frame.ref_lon_deg.to_numpy());phone_e,phone_n=local_xy(frame.phone_lat_deg.to_numpy(),frame.phone_lon_deg.to_numpy());phone_e+=ref_e[0]-phone_e[0];phone_n+=ref_n[0]-phone_n[0]
    heading=np.deg2rad(frame.phone_bearing_deg.to_numpy());linear=np.column_stack([frame.acc_x_mps2-frame.gravity_x_mps2,frame.acc_y_mps2-frame.gravity_y_mps2,frame.acc_z_mps2-frame.gravity_z_mps2]);pre=rel<args.outage_start_s
    axis,scale,bias,yaw_scale,yaw_bias=fit_phone_alignment(t,frame.phone_speed_kmh.to_numpy()/3.6,heading,linear,frame.gyro_yaw_rps.to_numpy(),pre);forward=scale*linear[:,axis]+bias;yaw=yaw_scale*frame.gyro_yaw_rps.to_numpy()+yaw_bias
    artifact=torch.load(args.model,map_location="cpu",weights_only=False);model=SpeedTCN(len(FEATURES));model.load_state_dict(artifact["state_dict"]);model.eval();values=frame[FEATURES].to_numpy(np.float32);values=(values-np.asarray(artifact["mean"],np.float32))/np.asarray(artifact["std"],np.float32);pred=np.zeros(len(frame));
    windows=np.stack([values[i-19:i+1] for i in range(19,len(values))]);
    with torch.no_grad():pred[19:]=model(torch.tensor(windows)).numpy()
    integrated=np.convolve(forward,np.ones(20)/10,mode="same") # approximate 2-second inertial velocity change
    baseline_conf=np.clip(1-np.abs(forward-np.convolve(forward,np.ones(5)/5,mode="same"))/4,.05,1)
    ml_conf=baseline_conf*np.exp(-np.abs(pred-integrated)/1.5);ml_conf[:19]=baseline_conf[:19]
    be,bn=run(frame,outage,forward,yaw,baseline_conf,t,phone_e,phone_n,heading);me,mn=run(frame,outage,forward,yaw,ml_conf,t,phone_e,phone_n,heading);start=np.where(outage)[0][0];end=np.where(outage)[0][-1]
    base=float(np.hypot((be[end]-be[start])-(ref_e[end]-ref_e[start]), (bn[end]-bn[start])-(ref_n[end]-ref_n[start])))
    ml=float(np.hypot((me[end]-me[start])-(ref_e[end]-ref_e[start]), (mn[end]-mn[start])-(ref_n[end]-ref_n[start])))
    print(f"Drive: {args.drive}\nOutage: {args.outage_s:.1f} s\nFilter-only drift: {base:.2f} m\nML-confidence drift: {ml:.2f} m\nImprovement: {(base-ml)/max(base,1e-6)*100:.2f}%")
    return 0


if __name__=="__main__":raise SystemExit(main())
