"""Create a compact offline OSM road pack around a recorded phone route."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phone-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--margin-deg", type=float, default=0.004)
    args = parser.parse_args()
    frame = pd.read_csv(args.phone_log)
    lat = pd.to_numeric(frame["gnss_lat_deg"], errors="coerce")
    lon = pd.to_numeric(frame["gnss_lon_deg"], errors="coerce")
    valid = lat.notna() & lon.notna()
    if valid.sum() < 2:
        raise ValueError("Phone log has fewer than two GNSS points")
    south, north = lat[valid].min() - args.margin_deg, lat[valid].max() + args.margin_deg
    west, east = lon[valid].min() - args.margin_deg, lon[valid].max() + args.margin_deg
    query = f"[out:json][timeout:60];way[highway]({south},{west},{north},{east});out geom;"
    url = "https://overpass-api.de/api/interpreter?data=" + quote(query)
    request = Request(url, headers={"User-Agent": "NavRakshak-IDR-hackathon-prototype/1.0"})
    with urlopen(request, timeout=90) as response:
        payload = json.loads(response.read().decode("utf-8"))
    ways = [e for e in payload.get("elements", []) if len(e.get("geometry", [])) >= 2]
    if not ways:
        raise RuntimeError("Overpass returned no usable road geometry")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"elements": ways}, separators=(",", ":")))
    print(f"Saved {len(ways)} road ways to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
