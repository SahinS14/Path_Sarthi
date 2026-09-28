"""Export the tiny PyTorch speed adapter as portable JSON weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--model", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); artifact = torch.load(args.model, map_location="cpu", weights_only=False); state = artifact["state_dict"]
    # This exporter is intentionally limited to the legacy delta-speed TCN.
    # Mobile code must reject a model with different output semantics rather
    # than accidentally interpreting a delta as absolute vehicle speed.
    payload = {"format":"navrakshak-speed-tcn-v1","output_semantics":"speed_delta_mps_over_2s",
               "window":artifact["window"],"features":artifact["features"],"mean":artifact["mean"],"std":artifact["std"],"weights":{key:value.detach().cpu().tolist() for key,value in state.items()}}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(payload,separators=(",",":")))
    print(f"Exported {sum(v.numel() for v in state.values())} parameters to {args.out}"); return 0


if __name__ == "__main__": raise SystemExit(main())
