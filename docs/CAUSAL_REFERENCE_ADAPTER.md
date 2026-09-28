# Causal covariance head and reference adapter

## Implemented

- `engine/src/causal_covariance.py`: a new trainable, past-only convolutional
  covariance head for lateral/vertical velocity constraints. It starts exactly at
  the fixed covariance baseline. Its output multipliers are bounded to
  0.001–1000, which ensures positivity but is not a tuned phone noise model.
- Chunked inference with caller-owned history. The history must be cleared on
  session changes or timestamp gaps. Tests compare chunked and batch outputs.
- `idr-benchmark/src/reference_adapter.py`: a strict array boundary to the
  unchanged upstream NumPy IEKF. It accepts initial state, body-frame IMU and
  optional covariance predictions, but no reference trajectory. It rejects
  nonfinite inputs, nonmonotonic timestamps and excessive IMU gaps.

The adapter adds the initial position externally because the upstream filter
starts its position at zero. It does not add live GNSS fusion or provide a
streaming filter implementation. The existing Android engine is unchanged.

## Real-time issue verified

The local upstream `MesNet` applies replication padding after convolution.
In a deterministic test using randomly initialized weights (not the released
trained weights), changing input samples 50 onward changed output samples before
50. The maximum variance change was 14.362888. This demonstrates future-sample
dependence in this architecture when used at matching timestamps; it is not an
accuracy comparison or a claim about the published evaluation.

The new causal head had exactly zero past-output change in the same test. Its
test head has nonzero random output weights so the test cannot pass simply because
baseline initialization makes all outputs constant.

The causal head uses only left padding and a 17-sample receptive field. At 10 Hz
that spans 1.6 seconds; at 100 Hz it spans 0.16 seconds. Sample rate and train-only
normalization must be included in the eventual trained artifact. The architecture
is new: copying pretrained upstream weights does not validate it.

## Integration check

On ideal 10 Hz straight motion for 30 seconds, the actual upstream filter was
called through the new adapter with both fixed covariance and baseline-initialized
neural covariance. Both maximum position errors were approximately 1.14e-13 m.
This is an ideal synthetic software check, not phone accuracy or trained model
performance. No long training was performed.

Run from the project root, choosing a new output filename on repeat:

```sh
PYTHONPATH=work/python-deps MPLCONFIGDIR=work/matplotlib python3 idr-benchmark/src/check_reference_causality.py --out outputs/reference-benchmark/causality-v1.json
PYTHONPATH=work/python-deps python3 -m unittest discover -s tests
```

## Still needed before dataset training

The per-recording timing and sensor-frame interpretations remain unresolved.
The new array adapter does not yet ingest IO-VNBD CSVs. Next, connect verified
recording contracts, test the differentiable filter training path, and measure
trajectory loss against fixed-filter and constant-speed baselines. Do not describe
the new untrained head as an accuracy improvement.

## Differentiable training check

`train_filter_smoke.py` now exercises trajectory-loss backpropagation through a
compatibility subclass of the upstream Torch filter. The downloaded files remain
unchanged. `torch_filter_compat.py` replaces the removed `torch.gesv` call with
`torch.linalg.solve` and replaces tensor constructions that detach gradients in
skew/rotation operations with differentiable operations. Small-angle series avoid
undefined derivatives at zero rotation. This path is currently CPU float64 only;
it has not been qualified for MPS or Android.

The controlled eight-second test includes small injected IMU biases. Results:

- Torch/NumPy maximum position difference: 2.59e-8 m.
- Full-filter output-bias gradient: analytic -0.0203671127 versus finite-difference
  -0.0203671116.
- Eight optimizer steps: synthetic position-coordinate MSE 1.87643 to 1.81718 m².
- 37 automated tests passed, including rotation gradient checks at zero and
  nonzero angles. The separate integration smoke script also passed.

This deliberately overfits one synthetic case to verify machinery. No dataset
training, generalization claim or production model export has occurred. Position
MSE here is not the PS's percentage-drift metric. Per-recording dataset contracts
remain the next unresolved prerequisite.

```sh
PYTHONPATH=work/python-deps MPLCONFIGDIR=work/matplotlib python3 idr-benchmark/src/train_filter_smoke.py --out outputs/reference-benchmark/trajectory-training-smoke-v2.json
```

Choose a new output path on repeat. This smoke command is not a long training job.

Reference source inspected locally:
https://github.com/mbrossar/ai-imu-dr (MIT license retained with unchanged source).
