# NavRakshak IDR — Benchmark Pipeline

Phase 1 establishes an auditable link between the smartphone sensors and vehicle
reference measurements in IO-VNBD. The original files under `work/datasets/` are
read-only input; generated canonical tables and figures go to `outputs/`.

Run from the repository root:

```bash
python3 idr-benchmark/src/profile_drive.py \
  --dataset "work/datasets/io-vnbd/raw/Synchronised V abd S datasets/Categorised IOVNB Dataset" \
  --drive Vw01
```

The next phase uses the exported canonical table to simulate GNSS blackouts and
evaluate a physics baseline without leaking vehicle speed into inference.

Phase 2:

```bash
MPLCONFIGDIR="$PWD/work/matplotlib" XDG_CACHE_HOME="$PWD/work/cache" \
python3 idr-benchmark/src/dead_reckoning_baseline.py \
  --dataset "work/datasets/io-vnbd/raw/Synchronised V abd S datasets/Categorised IOVNB Dataset" \
  --drive Vta10
```

Phase 3 trains a small temporal CNN with a drive-level split:

```bash
python3 idr-benchmark/src/train_speed_model.py \
  --dataset "work/datasets/io-vnbd/raw/Synchronised V abd S datasets/Categorised IOVNB Dataset"
```
