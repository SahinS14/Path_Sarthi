<p align="center">
  <img src="logo/logo.png" width="144" alt="Path Sārathi logo">
</p>

<h1 align="center">Path Sārathi</h1>

<p align="center">
  Smartphone-first intelligent dead reckoning for navigation through GNSS-denied environments.
</p>

<p align="center">
  <strong>Android 8.0+</strong> · <strong>On-device inference</strong> · <strong>GNSS + IMU</strong> · <strong>Research / pre-production build</strong>
</p>

Path Sārathi is an Android-first Intelligent Dead Reckoning (IDR) platform
developed for the SIH problem statement, <em>“AI-ML based Intelligent Dead
Reckoning system for seamless navigation.”</em> It uses a normal Android
phone's GNSS, accelerometer, gyroscope, gravity, and rotation-vector sensors
to preserve a clearly labelled estimated position when fresh GNSS measurements
are unavailable.

> **Important status:** this repository contains a functional engineering
> demonstrator and reproducible research pipeline. It is not a certified
> safety-critical navigation product and does not yet claim the problem
> statement's final drift target. Real, multi-phone field qualification is a
> required next step.

## What judges can see today

- Live Android navigation with destination search, route guidance, a moving
  vehicle marker, start/destination markers, voice prompts, and a saved-trip
  library.
- Fresh GNSS is fused with phone IMU measurements at a 10 Hz UI/filter cadence.
- A **Test outage** control deliberately withholds fresh GNSS from the fusion
  filter. It is a safe software simulation; it does **not** jam, block, or
  disable the phone's GPS radio.
- Trace provenance is always visible: **blue** is observed GNSS/route guidance;
  **gold** is the IDR estimate during a GNSS outage. The app never presents an
  estimated segment as a GNSS observation.
- Per-trip CSV logging and visual replay for later analysis.
- Automatic mount learning from trusted GNSS course observations, avoiding a
  fixed portrait/landscape phone-axis assumption.
- A small local SpeedTCN model is packaged inside the APK; no cloud model call
  is needed during an outage.

## System architecture

```text
 Android phone
 ┌─────────────────────────────────────────────────────────────────┐
 │ GNSS / GPS ──► freshness + accuracy gate ─┐                       │
 │                                             │                       │
 │ Accelerometer + gravity ─► mount alignment ├─► vehicle-forward     │
 │ Gyroscope + rotation vector ───────────────┘    acceleration/yaw  │
 │                                                   │                 │
 │ 9-feature, 20-sample IMU window ─► SpeedTCN ─────┤                 │
 └───────────────────────────────────────────────────┼─────────────────┘
                                                     ▼
                                      VehicleStateFilter (2D GNSS/INS)
                                      • GNSS updates once per fresh fix
                                      • acceleration / gyro bias adaptation
                                      • uncertainty tracking
                                      • non-holonomic vehicle constraint
                                                     │
                           ┌─────────────────────────┼─────────────────────────┐
                           ▼                         ▼                         ▼
                    live vehicle marker       source-coloured paths      local CSV + replay
                    & navigation UI           GNSS blue / IDR gold
```

### Runtime behaviour

1. The user starts outdoors and the app waits for a fresh, accurate GNSS fix.
2. While GNSS is fresh, the state filter propagates from IMU and accepts each
   new GNSS observation once.
3. When GNSS becomes stale, unavailable, or is withheld by **Test outage**, the
   state transitions to **DEAD_RECKONING** and propagates only with device
   motion estimates.
4. On GNSS return, the next accepted observation corrects the existing state;
   the UI remains continuous rather than resetting the trip.

**VehicleStateFilter** is a custom lightweight 2D GNSS/INS filter, not a claim
of a certified EKF/UKF implementation. Its non-holonomic constraint prevents
an independent lateral-velocity state from accumulating during vehicle motion.

## AI/ML model

| Item | Current implementation |
|---|---|
| Model | Compact depthwise temporal CNN (**SpeedTCN**) |
| Inputs | 9 IMU-derived features: acceleration, gravity, and gyro axes |
| Window | 20 samples at 10 Hz |
| Parameters | 1,273 |
| Local model asset | `app/src/main/assets/speed_tcn_all_io_vnbd.json` |
| Output | Estimated 2-second speed change, in m/s |
| Runtime use | A bounded consistency/confidence signal during short outages; it is not shown as a falsely precise absolute position predictor |

The training and verification code lives in `idr-benchmark/`. The Android
implementation checks its feature order, window length, and output semantics
before accepting the packaged model.

## Problem-statement coverage

| Required capability | Current status | Evidence in this repository |
|---|---|---|
| Smartphone GNSS + IMU capture | Implemented | `MainActivity.java` |
| GNSS freshness / outage handling | Implemented | 2-second freshness gate and explicit `DEAD_RECKONING` mode |
| Phone-to-vehicle alignment | Implemented, requires moving GNSS observations | `VehicleMountCalibration.java` |
| On-device ML speed signal | Implemented as a bounded assist signal | `SpeedTcnInference.java` |
| GNSS + INS state fusion | Implemented as a lightweight 2D filter | `VehicleStateFilter.java` |
| Visual separation of GNSS and IDR | Implemented | blue GNSS / gold IDR map segments |
| Trip logging and replay | Implemented | `LogsActivity.java`, `ReplayActivity.java` |
| Offline map matching for a national road network | Research/demo only | bundled graph is a small demo extract; no live position snapping is performed |
| 200 Hz external-IMU edge adapter | Planned | Python engine provides research foundations; hardware adapter is not yet shipped |
| Verified <10% blackout drift across phones/vehicles | Pending field validation | do not claim this benchmark yet |

The repository deliberately does not project the live estimate onto an
unverified map road. A map pack from the wrong region can make a vehicle appear
to jump or become stuck, which would be misleading.

## Repository layout

```text
app/                 Android application and on-device model assets
engine/              Python navigation/filter reference implementations
idr-benchmark/       IO-VNBD preparation, training, and evaluation tools
tests/               Unit and contract tests
docs/                Engineering notes and judge demo guide
logo/                Path Sārathi brand asset
```

## Data and reproducibility

The benchmark pipeline uses the public
[IO-VNBD dataset](https://github.com/onyekpeu/IO-VNBD). Original recordings,
local SDKs, generated models, logs, and evaluation outputs are intentionally
excluded from version control. Download the dataset yourself and follow its
licence/terms before use.

The training scripts maintain drive/session-oriented splits and include data
contract checks so that vehicle reference data is used as a training label,
never silently fed into model inference.

## Acknowledgements

- [IO-VNBD](https://github.com/onyekpeu/IO-VNBD) for the public inertial and
  odometry benchmark dataset.
- [MapLibre](https://maplibre.org/) and
  [OpenStreetMap contributors](https://www.openstreetmap.org/copyright) for
  mapping foundations.
- OpenFreeMap, OSRM, and Nominatim are used by the current demonstration map,
  routing, and search flow. A production deployment will use an approved,
  appropriately licensed provider or self-hosted equivalent.

## Build the Android app

### Requirements

- Android Studio with a compatible Android SDK
- JDK 17+
- A physical Android 8.0 (API 26) or newer phone with GNSS, accelerometer,
  gyroscope, gravity, and rotation-vector sensors
- Internet for the current live map tiles, destination search, and route
  service

```bash
git clone https://github.com/SahinS14/Path_Sarthi.git
cd Path_Sarthi
./gradlew assembleDebug
```

The debug APK is generated at:

```text
app/build/outputs/apk/debug/app-debug.apk
```

Install it on a connected test phone:

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

For a judge-facing distribution build, create a **signed release APK** in
Android Studio using **Build → Generate Signed Bundle / APK**. Never commit a
keystore, passwords, APK, or AAB to this repository.

## Run the Python checks

The benchmark tools use Python 3 with `numpy`, `pandas`, `matplotlib`, and
PyTorch installed in an isolated local environment.

```bash
PYTHONPATH="engine/src:idr-benchmark/src" \
python3 -m unittest discover -s tests -p 'test_*.py'
```

An example SpeedTCN training command, after downloading IO-VNBD, is:

```bash
PYTHONPATH=idr-benchmark/src python3 idr-benchmark/src/train_speed_model.py \
  --dataset "/absolute/path/to/IO-VNBD/raw/Synchronised V abd S datasets/Categorised IOVNB Dataset"
```


## Safety, privacy, and responsible use

- Run outage demonstrations while stationary or with a passenger operating the
  phone. Do not interact with the app while driving.
- Grant **Precise Location** to establish a start fix. Sensor/trip CSV files
  are written to the app's local storage and are shared only when the user
  explicitly exports them.
- The live map/route provider is suitable for demonstration only. A production
  rollout requires a licensed or self-hosted map/routing service, offline map
  packaging, device qualification, monitoring, and field validation.

## Contribution focus

High-value next work is: multi-device field data collection, a verified offline
OSM map package with conservative map matching, end-to-end blackout evaluation,
external-IMU adapter validation, and signed release/CI hardening.
