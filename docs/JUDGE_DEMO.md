# Path Sārathi — Judge Demo Guide

This guide is deliberately short and factual. It helps the team demonstrate
what the Android application does now without overstating field performance.

## Before the judges arrive

1. Install the signed demo APK on a physical Android 8.0+ phone.
2. Confirm the phone has GNSS, accelerometer, gyroscope, gravity, and
   rotation-vector sensors in the app's **Sensors** screen.
3. Grant **Precise Location** and confirm Location/GPS is enabled.
4. Keep mobile data or Wi-Fi available for the current map, search, and route
   services.
5. Start outdoors or near an open sky long enough to obtain a fresh GNSS fix.
6. Keep a second phone or the local APK/QR code ready in case the primary
   device has a network issue.

## Two-minute live flow

1. Open **Path Sārathi** and select **Live Navigation**.
2. Search for, or tap, a destination. Explain that the blue line is route
   guidance and observed GNSS.
3. Press **Start trip** and wait for the **GNSS LIVE** state.
4. Move gently as a passenger, or demonstrate while stationary. The app learns
   the phone mount orientation from trusted GNSS course while the vehicle is
   moving.
5. Press **Test outage** and confirm the warning dialog.
6. Explain the display:
   - **Blue:** observed GNSS measurements / planned route.
   - **Gold:** Path Sārathi's estimated inertial trajectory.
   - The vehicle marker continues from the filter state; an estimated segment
     is never recoloured as GNSS.
7. Press **Restore GPS**. The next fresh GNSS measurement is fused back into
   the existing state instead of starting a new trip.
8. Press **Save & stop**, then open **Trips** to show local CSV logging and
   replay.

## Accurate answers to likely questions

| Question | Recommended answer |
|---|---|
| Is this a real GPS jammer test? | No. **Test outage** safely withholds GNSS observations from our navigation filter. It does not jam or disable any radio. |
| What sensors are used? | GNSS/GPS, accelerometer, gyroscope, gravity, and rotation vector. No OBD-II link is required. |
| What is the AI model? | A compact 1,273-parameter SpeedTCN using a 20-sample, 9-feature IMU window. It runs locally and provides a bounded motion-consistency signal. |
| What is the fusion algorithm? | A lightweight custom 2D GNSS/INS state filter with GNSS freshness gating, bias adaptation, uncertainty tracking, and a non-holonomic vehicle-motion constraint. |
| Does the app work with every phone mount orientation? | It learns a vehicle-forward vector from trusted GNSS course while moving. Cross-device/mount validation is an active qualification task. |
| Is map matching complete? | The current Android build bundles a small offline road-graph demonstration and does not force a live position to an unverified road. A full offline OSM/HMM-style production map-matching package remains planned. |
| Have you achieved the final PS drift requirement? | Not yet as a validated field claim. The app and reproducible IO-VNBD pipeline are ready for that qualification; we will report only measured, held-out and real-device results. |
| Does it need internet? | GNSS and the on-device filter do not need cloud inference. The current live map, destination search, and routing use internet; production offline map packaging is planned. |

## What not to claim

- Do not call the software simulation a physical tunnel, jammer, or RF blackout.
- Do not claim lane-level accuracy, <10% drift, nationwide offline coverage, or
  production certification before independently measured field evidence exists.
- Do not describe the current map graph as a nationwide offline map or the
  runtime filter as a certified EKF/UKF.

## Simple deployment statement

> Path Sārathi is smartphone-first: an existing Android phone plus a secure
> holder and charger is sufficient for the base system. It has no mandatory
> OBD connection, external GNSS receiver, or vehicle wiring. The external
> high-rate-IMU edge variant is a separate future fleet configuration.

## Safety

The driver must not operate the phone during motion. Perform the demo while
stationary or with a passenger controlling the device.
