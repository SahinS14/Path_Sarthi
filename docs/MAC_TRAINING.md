# Dataset-first training on the M2 Mac

This run trains the compact IMU speed-increment baseline. It does not yet train
the proposed invariant-EKF covariance adapter or certify navigation accuracy.
The prepared data is already present. Work from the project root in VS Code's terminal.
These commands use the verified local Python installation plus the bundled
typing_extensions correction under work/python-deps. Do not delete that directory.

## Train

```sh
PYTHONPATH=work/python-deps python3 idr-benchmark/src/train_validated.py train --data work/datasets/io-vnbd/prepared-v1 --out outputs/validated-speed-v1 --epochs 100 --patience 12 --device cpu
```

100 is the maximum total epoch count; validation early stopping can finish sooner.
CPU is supported on this Mac. MPS is also supported with `--device mps`; for this
small network GPU transfer overhead can outweigh its benefit. Use a different
output directory for each new experiment. Checkpoints are saved after each epoch.

Resume the same CPU run after interruption:

```sh
PYTHONPATH=work/python-deps python3 idr-benchmark/src/train_validated.py train --data work/datasets/io-vnbd/prepared-v1 --out outputs/validated-speed-v1 --epochs 100 --patience 12 --device cpu --resume
```

Resume requires the same prepared data and device. An interrupted partial epoch
restarts from the last completed epoch. Epoch count is a total, not additional epochs.
The checkpoint restores weights, optimizer, random generators and validation history.

## Evaluate the selected model

After training/model choices are finished, evaluate once on the reserved test split:

```sh
PYTHONPATH=work/python-deps python3 idr-benchmark/src/train_validated.py evaluate --data work/datasets/io-vnbd/prepared-v1 --model outputs/validated-speed-v1/best.pt --out outputs/validated-speed-v1/test.json
```

Share `history.json` and `test.json` when done. The report compares each drive against
predicting zero speed change, and gives equal weight to each drive in the macro mean.
If the model loses to that baseline, do not deploy it. Even beating the baseline
requires a subsequent position-outage replay before claiming improved navigation.

## Data policy

The prepared manifest records raw and prepared file hashes, training-only feature
normalization, fixed session-group splits and all exclusions. 18 questionable drives
are quarantined because row correspondence is ambiguous or elapsed clocks disagree
by more than 50 ms. Raw files remain intact. No arbitrary truncation/alignment repair
was used to make these drives pass. Repair and validate them before preparing v2.
54 accepted drives produce 60,319 train, 4,018 validation and 5,681 test windows.
Twenty samples at nominal 10 Hz span 1.9 seconds between target endpoints.

This is a retrospective IO-VNBD development split: earlier project work has already
used the dataset. It is not proof of generalization to unseen phones. Absolute label
alignment and eventual independent field validation remain acceptance requirements.
