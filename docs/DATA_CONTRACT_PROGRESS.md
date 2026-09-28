# IO-VNBD preparation checkpoint — 13 September 2026

The dataset remains the development source. No new road collection is required
for this stage. Original CSVs, prepared-v1, checkpoints and historical reports have
not been changed. The CSV audit follows the scientific-data guidance to preserve
raw observations and distinguish assumptions from verified measurements.

## Recording independence

The new metadata-only audit reads every phone DATE field and fingerprints both
source files. Connected components join shared dates, identical source hashes and
named subdivisions. A recording spanning midnight connects both days.

All 72 categorized pairs have parseable dates. They form eight conservative groups.
Two groups cross the old split boundaries:

- 6 November 2019: 42 recordings, with historical training, validation and test
  membership (and some excluded from prepared-v1).
- 8 January 2020: 20 recordings, with historical training and validation membership
  (and some excluded from prepared-v1).

This establishes shared recording days, not identical trips or measured inflation
of the scores. It means the old evaluation is not recording-day independent.
Its scores must remain historical diagnostics.

The generated proposed train/validation assignment is deterministic and has no
date overlap. It is not approved for training yet: only eight groups exist, some
have unusable pairs, and balance/eligibility must be assessed after interpretation.
There is no claim of a pristine local test set. Repartitioning previously used
recordings cannot erase their development exposure. Unsynchronized originals must
inherit their recording's group, not become additional independent examples.

## New conversion and alignment primitives

`input_contract.py` provides explicit speed conversion, separate proper rotations
for acceleration and gyro, and a declared specific-force/linear-acceleration
convention. It does not infer units from vehicle reference labels. Linear
acceleration needs same-frame gravity to restore specific force.

Offline reference-label alignment requires a declared clock offset, strictly
increasing clocks and a positive maximum gap. It rejects extrapolation and leaves
missing/unusable labels as NaN with a validity mask. This interpolation is not
appropriate for live GNSS availability: runtime fusion must consume only received
measurements. Time-varying frame conversions need a separate adapter.

These are tested building blocks, not verified IO-VNBD per-recording transforms.
They have not replaced the historical loader or been integrated into the GitHub
training path. Every new manifest record remains `training_eligible: false` until
its sensor and timing interpretation is verified. That field describes the new
manifest; it does not disable the old training command.

## Reproduction

Run from the project directory (use a new output name if one already exists):

```sh
python3 idr-benchmark/src/session_manifest.py --dataset 'work/datasets/io-vnbd/raw/Synchronised V abd S datasets/Categorised IOVNB Dataset' --historical work/datasets/io-vnbd/prepared-v1/manifest.json --out outputs/data-audit/session-groups-v1.json
PYTHONPATH=work/python-deps python3 -m unittest discover -s tests
```

## Next implementation gate

Establish per-recording frame and absolute timing evidence, then connect the strict
adapter to the reference filter. Test supervised alignment independently of model
trajectory error before running long training. Keep the eight date groups intact
when selecting validation or cross-validation folds.

## Product acceptance remains broader than training

| PS capability | Evidence still required |
| --- | --- |
| Smartphone-only speed and vibration handling | Held-out trajectory evaluation, shocks and stop/start cases |
| Mount alignment and recalibration | Pitch/roll/yaw recovery and mount-change tests |
| Offline map matching and kinematic constraints | Offline road graph, ambiguous junctions, wrong-road recovery |
| AI GNSS+INS fusion and outage transition | Learned correction comparison, GNSS freshness, reacquisition tests |
| Less than 10% blackout drift | Per-outage distance/error plots and failure rates; current replay fails |
| 10 Hz phone output | Integrated Android timing, battery and thermal measurements |
| External IMU edge engine around 200 Hz | Defined sensor interface and end-to-end FOG-data benchmark |
| Smooth complete navigation UI | Integrated app replay, error states and real-device verification |

Lane-level positioning in the descriptive requirement must also be evaluated;
meeting the relative drift threshold alone would not prove lane-level accuracy.
Dataset and device-replay results precede road tests, but cannot certify production
performance across real vehicles and phones on their own.
