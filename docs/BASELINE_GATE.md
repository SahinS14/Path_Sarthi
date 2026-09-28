# GNSS blackout baseline gate

The first aggregate replay evaluates the current portable 2D vehicle filter on
the supplied synchronized IO-VNBD data. Each case receives phone GNSS only before
the simulated blackout. Vehicle labels are used after inference to score aligned
trajectory drift. It includes no map-matching correction and no learned adaptive
covariance model.

## Results

| Blackout | Cases | Current filter median drift | Constant-velocity median drift | Current filter under 10% |
| --- | ---: | ---: | ---: | ---: |
| 10 seconds | 59 | 76.04% | 23.38% | 2 |
| 30 seconds | 57 | 85.60% | 50.52% | 0 |
| 60 seconds | 50 | 92.88% | 60.08% | 1 |

The present baseline does not satisfy the ISRO less-than-10%-drift target and
performs worse than constant velocity under this provisional interpretation of
the dataset. It must not be called production-ready or used as the final Android
navigation engine.

## Decision

Keep the baseline for regression testing. The next engine iteration must beat
constant velocity at every evaluated blackout duration before it is integrated
into the Android app. Map matching will be evaluated separately and must not use
vehicle reference data during a blackout.

Run the benchmark:

```sh
PYTHONPATH=work/python-deps MPLCONFIGDIR=work/matplotlib python3 idr-benchmark/src/benchmark_blackouts.py \
  --dataset 'work/datasets/io-vnbd/raw/Synchronised V abd S datasets/Categorised IOVNB Dataset' \
  --out outputs/benchmarks/blackout-baseline-v2.json
```

This remains a development replay, not a field-validation or lane-level test.
