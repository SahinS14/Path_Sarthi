"""Portable 2D error-state vehicle filter for NavRakshak IDR.

The filter is intentionally framework-free: the same equations can be ported
to Android Java/Kotlin or run against an external IMU stream.  It models the
vehicle in its forward direction, which enforces the non-holonomic constraint
that a road vehicle does not independently move sideways.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, exp, sin
from typing import Optional


def wrap(angle: float) -> float:
    return atan2(sin(angle), cos(angle))


@dataclass
class VehicleState:
    east_m: float = 0.0
    north_m: float = 0.0
    speed_mps: float = 0.0
    heading_rad: float = 0.0
    acc_bias_mps2: float = 0.0
    gyro_bias_rps: float = 0.0
    position_sigma_m: float = 25.0
    speed_sigma_mps: float = 4.0
    mode: str = "WAITING_FOR_GNSS"


class VehicleStateFilter:
    """10–200 Hz GNSS/IMU filter with NHC, ZUPT, and confidence inputs.

    `imu_confidence` is supplied by the learned adapter (0=noisy, 1=clean).
    `route_heading_rad`, when provided by a high-confidence map match, is a
    gentle lateral/heading constraint rather than an unconditional map snap.
    """
    def __init__(self, outage_acceleration_gain: float = 1.0, speed_anchor_time_s: Optional[float] = None) -> None:
        if not 0 <= outage_acceleration_gain <= 1 or (speed_anchor_time_s is not None and speed_anchor_time_s <= 0):
            raise ValueError('Invalid outage tuning parameters')
        self.state = VehicleState()
        self._last_time_s: Optional[float] = None
        self._stationary_s = 0.0
        self._outage_acceleration_gain = outage_acceleration_gain
        self._speed_anchor_time_s = speed_anchor_time_s
        self._last_gnss_speed_mps: Optional[float] = None

    def step(self, time_s: float, forward_acc_mps2: float, yaw_rate_rps: float,
             imu_confidence: float, stationary_probability: float,
             gnss_east_m: Optional[float] = None, gnss_north_m: Optional[float] = None,
             gnss_speed_mps: Optional[float] = None, gnss_heading_rad: Optional[float] = None,
             gnss_accuracy_m: Optional[float] = None, route_heading_rad: Optional[float] = None,
             route_confidence: float = 0.0) -> VehicleState:
        if self._last_time_s is None:
            self._last_time_s = time_s
            if gnss_east_m is not None and gnss_north_m is not None:
                self._initialise(gnss_east_m, gnss_north_m, gnss_speed_mps, gnss_heading_rad)
            return self.state
        dt = max(0.001, min(time_s - self._last_time_s, 0.25)); self._last_time_s = time_s
        if self.state.mode == "WAITING_FOR_GNSS":
            if gnss_east_m is not None and gnss_north_m is not None:
                self._initialise(gnss_east_m, gnss_north_m, gnss_speed_mps, gnss_heading_rad)
            return self.state
        has_gnss = gnss_east_m is not None and gnss_north_m is not None and gnss_accuracy_m is not None and gnss_accuracy_m <= 30.0
        confidence = min(1.0, max(0.02, imu_confidence))
        if stationary_probability > .8:
            self._stationary_s += dt
        else:
            self._stationary_s = 0.0
        # Propagation in the vehicle-forward direction is the 2D NHC: there is
        # no free lateral velocity state to accumulate sideways drift.
        outage_gain = 1.0 if has_gnss else self._outage_acceleration_gain
        acceleration = (forward_acc_mps2 - self.state.acc_bias_mps2) * confidence * outage_gain
        self.state.speed_mps = max(0.0, self.state.speed_mps + acceleration * dt)
        self.state.heading_rad = wrap(self.state.heading_rad + (yaw_rate_rps - self.state.gyro_bias_rps) * dt)
        self.state.east_m += self.state.speed_mps * sin(self.state.heading_rad) * dt
        self.state.north_m += self.state.speed_mps * cos(self.state.heading_rad) * dt
        self.state.position_sigma_m = min(500.0, self.state.position_sigma_m + (0.25 + 3.0 * (1-confidence)) * dt)
        self.state.speed_sigma_mps = min(30.0, self.state.speed_sigma_mps + (.04 + .8 * (1-confidence)) * dt)
        # Zero-velocity update prevents idle vibration from becoming distance.
        if self._stationary_s > 0.5:
            self.state.acc_bias_mps2 += .08 * forward_acc_mps2
            self.state.speed_mps *= .25
            self.state.speed_sigma_mps *= .6
        if has_gnss:
            self._gnss_update(gnss_east_m, gnss_north_m, gnss_speed_mps, gnss_heading_rad, gnss_accuracy_m, dt)
            self.state.mode = "GNSS_AIDED"
        else:
            # A low-cost MEMS accelerometer cannot sustain an unconstrained speed
            # integral for a long blackout. Decay only toward the last measured
            # GNSS speed, never toward zero, and only when configured.
            if self._speed_anchor_time_s is not None and self._last_gnss_speed_mps is not None:
                amount = min(1.0, dt / self._speed_anchor_time_s)
                self.state.speed_mps += amount * (self._last_gnss_speed_mps-self.state.speed_mps)
            self.state.mode = "DEAD_RECKONING"
        # Map matching may correct heading only when it is unambiguous. Position
        # projection belongs to the map matcher, not this inertial core.
        if route_heading_rad is not None and route_confidence >= .85 and self.state.speed_mps > 1.0:
            gain = .10 * min(1.0, route_confidence)
            self.state.heading_rad = wrap(self.state.heading_rad + gain * wrap(route_heading_rad - self.state.heading_rad))
        return self.state

    def _initialise(self, east: float, north: float, speed: Optional[float], heading: Optional[float]) -> None:
        self.state.east_m, self.state.north_m = east, north
        self.state.speed_mps = max(0.0, speed or 0.0); self.state.heading_rad = heading or 0.0
        self._last_gnss_speed_mps = self.state.speed_mps
        self.state.mode = "GNSS_AIDED"

    def _gnss_update(self, east: float, north: float, speed: Optional[float], heading: Optional[float], accuracy: float, dt: float) -> None:
        measurement_sigma = max(2.0, accuracy)
        gain = self.state.position_sigma_m ** 2 / (self.state.position_sigma_m ** 2 + measurement_sigma ** 2)
        self.state.east_m += gain * (east - self.state.east_m); self.state.north_m += gain * (north - self.state.north_m)
        self.state.position_sigma_m *= (1.0 - gain) ** .5
        if speed is not None:
            speed_gain = self.state.speed_sigma_mps ** 2 / (self.state.speed_sigma_mps ** 2 + .7 ** 2)
            residual = speed - self.state.speed_mps
            self.state.speed_mps += speed_gain * residual
            self._last_gnss_speed_mps = max(0.0, speed)
            self.state.acc_bias_mps2 -= .02 * residual / max(.05, dt)
            self.state.speed_sigma_mps *= (1.0 - speed_gain) ** .5
        if heading is not None and self.state.speed_mps > 2.0:
            error = wrap(heading - self.state.heading_rad)
            self.state.heading_rad = wrap(self.state.heading_rad + .55 * error)
            self.state.gyro_bias_rps -= .001 * error / max(.05, dt)
