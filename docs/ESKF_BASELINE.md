# 3D filter baseline: diagnostic results

Implemented a new, original 15-state multiplicative error-state EKF in
`engine/src/error_state_filter.py`. This is not a port or reproduction of the
21-state invariant EKF from AI-IMU-DR. It provides position, velocity, orientation,
accelerometer/gyro biases, full covariance propagation, Joseph-form updates,
innovation gating, duplicate-GNSS rejection and non-holonomic pseudo-measurements.
The covariance interface permits a future learned positive-variance adapter.
Extrinsic mounting estimation, delayed-GNSS replay and learned covariance are not
implemented. The Android app has not been switched to this experimental core.

Twelve tests pass across the filter and prior training-window code, including
gravity cancellation under rotated axes, acceleration, constant-speed travel,
coordinated turning, positive covariance, the NHC attitude Jacobian, invalid
timestamps, duplicate GNSS and outlier rejection. These tests establish limited
mathematical behavior, not field or dataset accuracy.

## Replay protocol

`idr-benchmark/src/replay_eskf.py` uses only the validation drives from prepared-v1.
There are 26 completed simulated outages and one skipped short recording.
Each outage follows 20 seconds of phone-only warmup and lasts 10, 30 or 60 seconds.
Vehicle reference enters scoring only. No GNSS is passed during the blackout.
Raw phone speed is explicitly interpreted as m/s, rather than using the old loader's
vehicle-label-based unit guess. Body axes are provisionally mapped to
(-sensor Y, sensor X, sensor Z); gyroscope channels are provisionally interpreted
as (roll, pitch, yaw) before the same transform. These remain hypotheses.

The report includes raw absolute endpoint error and outage-relative error; the
latter removes the starting position offset but does not align rotation, scale,
or timestamps using outage truth. Stationary/very short reference distances do
not yield meaningful percentage metrics.

## Results, not acceptance claims

Across completed outages with reference travel >=50 m, median relative endpoint
drift was 713.81% without NHC, 110.71% with fixed NHC, and 23.18% for constant
velocity. The assumed adapter fails. No learned covariance training or deployment
is justified by these results. Validation phone/reference position separations
also vary substantially; they cannot simply be described as antenna offsets.

Median desktop processing time with NHC was about 0.913 ms per sample on this run.
This includes numerical substeps but is not an Android measurement or a certified
200 Hz end-to-end benchmark. Detailed results and trajectories are under
`outputs/eskf-diagnostic-v1/`.

Reproduce:

```sh
python3 idr-benchmark/src/replay_eskf.py --dataset 'work/datasets/io-vnbd/raw/Synchronised V abd S datasets/Categorised IOVNB Dataset' --manifest work/datasets/io-vnbd/prepared-v1/manifest.json --out outputs/eskf-diagnostic-v1
PYTHONPATH=work/python-deps python3 -m unittest discover -s tests
```

Next prerequisite: verify sensor axes/signs, speed units and phone/reference
temporal alignment from dataset documentation and training-only diagnostics.
The observed orientation/gravity disagreement is an unresolved warning, not proof
of a particular transformation. Do not optimize mapping or model parameters on
reserved test trajectories. Follow with fixed-covariance replay, then train the
uncertainty adapter using trajectory loss only once the baseline is physically sound.

References:
- https://github.com/mbrossar/ai-imu-dr
- https://arxiv.org/html/1904.06064v1
- https://github.com/onyekpeu/IO-VNBD
- https://pmc.ncbi.nlm.nih.gov/articles/PMC7907232/
