"""Memory-safe final fit over every IO-VNBD paired drive."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from io_vnbd import find_pairs, load_drive
from train_speed_model import FEATURES, SpeedTCN, windows


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", type=Path, required=True); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--epochs", type=int, default=8); parser.add_argument("--device", choices=("cpu","mps"), default="cpu"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--exclude-drive", default=None, help="Drive reserved for independent evaluation.")
    args = parser.parse_args(); torch.manual_seed(26168); torch.set_num_threads(1)
    pairs = sorted(find_pairs(args.dataset), key=lambda p: p.name)
    if args.exclude_drive: pairs = [pair for pair in pairs if pair.name.lower() != args.exclude_drive.lower()]
    if not pairs: raise ValueError("No training drives remain after exclusion")
    # Two-pass streaming global normalization: no dataset-wide feature tensor.
    total = 0; total_sum = np.zeros(len(FEATURES), dtype=np.float64); total_sq = total_sum.copy()
    for pair in pairs:
        values = load_drive(pair)[FEATURES].to_numpy(dtype=np.float32); valid = np.isfinite(values).all(axis=1); values = values[valid]; total += len(values); total_sum += values.sum(axis=0); total_sq += (values * values).sum(axis=0)
    mean = (total_sum / total).astype(np.float32); std = np.sqrt(np.maximum(total_sq / total - mean.astype(np.float64) * mean.astype(np.float64), 1e-8)).astype(np.float32)
    device = torch.device("mps" if args.device == "mps" and torch.backends.mps.is_available() else "cpu"); model = SpeedTCN(len(FEATURES)).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4); loss_fn = nn.SmoothL1Loss(); args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "speed_tcn_all_io_vnbd.pt"
    completed_epochs = 0
    if args.resume and checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False); model.load_state_dict(saved["state_dict"]); optimizer.load_state_dict(saved["optimizer_state"]); completed_epochs = int(saved.get("completed_epochs", 0))
    for epoch in range(1, args.epochs + 1):
        model.train(); losses=[]; windows_seen=0
        for pair in pairs:
            x, y, _, _ = windows(load_drive(pair), mean, std, 20, 20, "delta")
            loader = DataLoader(TensorDataset(torch.tensor(x), torch.tensor(y)), batch_size=256, shuffle=True)
            for batch_x, batch_y in loader:
                optimizer.zero_grad(); loss = loss_fn(model(batch_x.to(device)), batch_y.to(device)); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu())); windows_seen += len(batch_y)
        completed_epochs += 1; torch.save({"state_dict":model.state_dict(),"optimizer_state":optimizer.state_dict(),"features":FEATURES,"window":20,"mean":mean.tolist(),"std":std.tolist(),"completed_epochs":completed_epochs}, checkpoint); print(f"epoch {completed_epochs:02d}: windows {windows_seen:,}, mean Huber loss {np.mean(losses):.4f}", flush=True)
    report={"model":"SpeedTCN IMU speed-change adapter","training":"streaming fit on paired IO-VNBD drives","excluded_drive":args.exclude_drive,"drive_count":len(pairs),"feature_samples":int(total),"epochs":completed_epochs,"parameters":sum(p.numel() for p in model.parameters()),"independent_test_score":None,"integrity":"The excluded drive, if specified, is reserved for independent evaluation."}
    (args.out / "speed_tcn_all_io_vnbd_report.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
