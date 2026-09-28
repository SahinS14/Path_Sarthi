# NavRakshak production engineering record

Status: existing Android application and research filters; production acceptance is not met.
Reviewed 2026-09-10 against the supplied ISRO statement.

## Data and training decision

Use the synchronized IO-VNBD collection first. The local dataset contains categorized
and uncategorized copies; never treat those copies as separate independent drives.
The upstream repository also contains unsynchronized data, absent locally. Downloading
that collection is optional until synchronization and label quality can be validated.
Do not train on every drive and then report accuracy on the same drives. Reserve
complete recording sessions, preferably drivers and devices, for validation and test.
The existing final model trained on all 72 paired drives cannot be independently
evaluated on those drives. Existing development results also make a newly selected
IO-VNBD test set retrospective; genuinely new phone recordings are needed.

Before long training, validate timestamp monotonicity, S/V correspondence, units,
sensor coordinate conventions, gaps, invalid samples and reference quality. The
current loader truncates unequal pairs and guesses phone-speed units using vehicle
labels. These decisions need explicit dataset metadata and reproducible tests.
Normalization must use only the training partition. No future samples may enter
online features. Save split hashes, configuration, optimizer and RNG state with
checkpoints, and compare validation trajectory error against inertial and constant
velocity baselines. More epochs alone will not repair alignment or label errors.

Run the non-training inventory from the project root in a VS Code terminal:

```sh
python3 idr-benchmark/src/audit_dataset.py --dataset work/datasets/io-vnbd/raw --out outputs/data-audit/inventory.json
```

## AI-IMU-DR assessment

Sources: https://github.com/mbrossar/ai-imu-dr and https://arxiv.org/abs/1904.06064

The method combines a 3D invariant EKF, bias and mounting-transform estimation,
non-holonomic measurements, and learned measurement covariance. Its reported
1.10% KITTI translational error is not a phone benchmark or a guarantee for our
outage metric. The released weights use many KITTI sequences for training; the
README identifies sequence 02 as the test sequence for those weights.

Local source is already available under work/reference/ai-imu-dr. The code uses
a 21-dimensional error covariance and a small convolutional covariance adapter;
it is a useful algorithmic reference, not an Android production package. The
sequence convolution/padding implementation needs a causality audit and streaming
equivalence tests before online reuse. Desktop/phone latency and memory have not
been reproduced. Preserve its MIT notice for any reused code.

Adopt the architecture as a candidate: a tested 3D error-state/invariant EKF with
GNSS updates, learned positive bounded constraint covariance, and explicit sensor
extrinsics. Retrain on phone data. Compare against fixed-covariance filtering.
Do not blindly apply car lateral/vertical constraints to leaning two-wheelers.

## Implementation and acceptance order

1. Data integrity and immutable evaluation splits; chronological replay with GNSS
   withheld from every estimator input during outages. Score reference independently.
2. Sensor ingestion with monotonic timestamps, synchronization, full 3D mount
   calibration, gap detection, and high-rate raw logging. The current app samples
   latest sensor values from a 100 ms UI timer and uses a planar mount approximation.
3. Validated covariance filter: propagation, bias estimation, gated GNSS updates,
   fresh measurements consumed once, dropout/reacquisition state machine, uncertainty
   output and degraded tracking. Current app reuses stale GNSS indefinitely.
4. Causal learned adapter with resumable CPU/CUDA training and validation-based model
   selection. Benchmark both trajectory accuracy and on-device runtime before export.
5. Offline road graph and probabilistic map matching, ambiguity handling at junctions,
   parallel roads and stacked roads. Avoid unconditional route snapping.
6. Android integration: sensor processing outside UI, foreground recording lifecycle,
   offline maps/routing, model versioning, session recovery and export diagnostics.
7. External IMU adapter and measured 200 Hz processing; Android 10 Hz position output
   measured separately from GNSS fix rate. Validate battery and thermal behavior.
8. Independent field acceptance across phones, mounts, cars, trucks and two-wheelers:
   real and simulated 50 m / 1 km outages, turns, braking, potholes, stops, mount shifts
   and return of poor-quality GNSS. Report endpoint and maximum error, drift fraction,
   percentiles, failures and uncertainty coverage, not only an average.

The statement asks for lane-level behavior as well as <10% drift; passing the latter
alone does not establish lane accuracy. A product must disclose degraded confidence
when its sensors cannot support the requested accuracy. Ground-truth instrumentation
is allowed for evaluation but must never enter deployed inference during an outage.

## Current evidence and next dependency

Saved phone results include 136.77% drift on the cleaned session and a phone model
that underperforms its zero-change baseline. Earlier reports reference phone sessions
153755 and 162838, which are absent from the six local raw recordings. These reports
cannot currently be reproduced from this workspace. A complete training run should
wait for data quality and reference alignment checks, not be launched merely because
all synchronized drives are present.
