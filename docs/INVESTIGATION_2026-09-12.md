# Dataset adaptation and reference-filter checkpoint

The current work remains a research baseline. No new experimental filter has been
put into the Android app, and the <10% outage-drift requirement is not met.

## What was verified

- The actual upstream `NUMPYIEKF` runs locally, with its source unchanged and MIT
  notice preserved. At 100 Hz on three 30-second synthetic scenarios it costs
  approximately 0.21–0.23 ms per sample on this Mac. Stationary and constant-speed
  errors are numerical zero; the ideal constant-turn endpoint error is 0.0415 m.
  This is not a KITTI reproduction or a phone accuracy result; learned weights
  were not used. The source hash is recorded with the benchmark results.
- The upstream training configuration uses 400 epochs. Its objective trains an
  uncertainty adapter through the filter, not our original scalar speed-change
  regression. Epoch counts are not a like-for-like measure of training effort.
- The original upstream unsynchronized S-Vta2 CSV was downloaded and its SHA-256
  verified against the Git LFS pointer:
  `97ef388d6e5cc3ce17ae3d56d69c77052e5d51a6234176eb95a5e4de6cec7fe5`.
  Its first sensor rows and mean acceleration/gravity exhibit the same conventions
  as our synchronized copy. Re-downloading the full collection will not, by itself,
  resolve the convention problem demonstrated by this recording.

## Frame investigations

Training-only diagnostics compare recorded acceleration against vehicle-frame and
earth-frame accelerations, fitting on the first half of each drive and evaluating
on the second. Examples of earth/body R²: Vw05 0.688/-0.908, Vw06 0.614/-0.636,
Vta24 0.843/0.279. Other drives do not show this pattern. This supports testing an
earth-frame hypothesis but does not establish a single mapping for the dataset.

Added an 8-state navigation-frame EKF for that hypothesis. It tracks position,
velocity, acceleration bias, heading and gyro bias jointly. Its input contract
explicitly requires ENU acceleration with gravity removed. It cannot be fed raw
device acceleration without conversion. Its tests cover constant-speed travel,
acceleration without a second rotation, GNSS-assisted bias convergence, duplicate
measurements, and a heading error that must not force the vehicle to stop.

Validation diagnostic variants still fail. Across 21 outages covering at least
50 m, the uncalibrated navigation-frame model without NHC has 54.29% median
relative endpoint drift. Pre-outage phone-GNSS bias calibration worsens it to
79.66%. Constant velocity remains 23.18% for the same cases. The variants with
NHC are also worse than constant velocity. Keep failed experiments for diagnosis;
do not describe one passing short outage as compliance.

## Files and reproduction

- `idr-benchmark/src/diagnose_accel_frame.py`
- `engine/src/navigation_frame_filter.py`
- `tests/test_navigation_frame_filter.py`
- `idr-benchmark/src/benchmark_reference.py`
- `outputs/data-audit/acceleration-frames-train.json`
- `outputs/navigation-frame-diagnostic-v2/report.json`
- `outputs/navigation-frame-calibrated-v1/report.json`
- `outputs/reference-benchmark/synthetic-100hz.json`

```sh
PYTHONPATH=work/python-deps MPLCONFIGDIR=work/matplotlib python3 idr-benchmark/src/benchmark_reference.py --out outputs/reference-benchmark/synthetic-100hz.json
PYTHONPATH=work/python-deps python3 -m unittest discover -s tests
```

Seventeen automated tests passed after these changes. They validate limited
software behavior, not the unresolved data-to-filter mapping.

## Remaining decisions before another long training run

1. Establish a per-recording sensor-frame and timing contract. Do not infer a
   universal convention from header names or one good correlation.
2. Make split independence stronger: the current split groups named subdivisions,
   but multiple separately named drives can belong to the same day/session.
3. Validate reference label synchronization independently of trajectory model error.
4. Reproduce the upstream learned method on its correctly specified input format,
   then adapt it with an explicit phone adapter and trajectory-based validation.

No 400-epoch phone run is ready at this checkpoint. The existing training command
continues to train only the original speed-change component.

Sources: https://github.com/mbrossar/ai-imu-dr,
https://arxiv.org/html/1904.06064v1,
https://github.com/onyekpeu/IO-VNBD.
