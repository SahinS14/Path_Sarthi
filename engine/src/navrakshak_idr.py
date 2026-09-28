"""Portable NavRakshak Intelligent Dead Reckoning fusion engine.

Input units: east/north metres, m/s, radians, seconds. The class deliberately
has no Android/Python framework dependency so it can be adapted for smartphone
or external IMUs (including a 200 Hz FOG source).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from typing import Optional


@dataclass
class NavigationState:
    east_m: float = 0.0
    north_m: float = 0.0
    speed_mps: float = 0.0
    heading_rad: float = 0.0
    position_variance_m2: float = 100.0
    speed_variance_m2ps2: float = 16.0
    mode: str = "UNINITIALIZED"


class IntelligentDeadReckoner:
    """Adaptive GNSS+INS fusion with NHC and seamless GNSS-loss handling.

    `step` is called for every IMU sample. Supply GNSS fields when a quality
    gated location fix exists; omit them during a tunnel/outage. This is a
    compact error-state/complementary filter designed for deterministic edge
    execution and easy porting to Kotlin/C++.
    """
    def __init__(self) -> None:
        self.state = NavigationState()
        self._last_time_s: Optional[float] = None
        self._acc_bias_mps2 = 0.0
        self._gyro_bias_rps = 0.0

    def step(
        self,
        time_s: float,
        forward_acc_mps2: float,
        yaw_rate_rps: float,
        vibration_score: float = 0.0,
        gnss_east_m: Optional[float] = None,
        gnss_north_m: Optional[float] = None,
        gnss_speed_mps: Optional[float] = None,
        gnss_heading_rad: Optional[float] = None,
        gnss_accuracy_m: Optional[float] = None,
    ) -> NavigationState:
        if self._last_time_s is None:
            self._last_time_s = time_s
            if gnss_east_m is not None and gnss_north_m is not None:
                self.state.east_m, self.state.north_m = gnss_east_m, gnss_north_m
                self.state.speed_mps = max(0.0, gnss_speed_mps or 0.0)
                self.state.heading_rad = gnss_heading_rad or 0.0
                self.state.mode = "GNSS_AIDED"
            return self.state
        dt = max(0.001, min(time_s - self._last_time_s, 0.25)); self._last_time_s = time_s
        if self.state.mode == "UNINITIALIZED":
            if gnss_east_m is not None and gnss_north_m is not None:
                self.state.east_m, self.state.north_m = gnss_east_m, gnss_north_m
                self.state.speed_mps = max(0.0, gnss_speed_mps or 0.0)
                self.state.heading_rad = gnss_heading_rad or 0.0
                self.state.mode = "GNSS_AIDED"
            return self.state
        # Vibration-aware process noise prevents potholes from being interpreted
        # as sustained vehicle acceleration.
        quality = max(0.05, 1.0 - min(1.0, vibration_score))
        effective_acc = (forward_acc_mps2 - self._acc_bias_mps2) * quality
        self.state.speed_mps = max(0.0, self.state.speed_mps + effective_acc * dt)
        self.state.heading_rad += (yaw_rate_rps - self._gyro_bias_rps) * dt
        self.state.east_m += self.state.speed_mps * sin(self.state.heading_rad) * dt
        self.state.north_m += self.state.speed_mps * cos(self.state.heading_rad) * dt
        self.state.position_variance_m2 += (0.6 + 5.0 * vibration_score) * dt
        self.state.speed_variance_m2ps2 += (0.15 + vibration_score) * dt

        has_gnss = gnss_east_m is not None and gnss_north_m is not None and (gnss_accuracy_m or 999.0) <= 30.0
        if not has_gnss:
            self.state.mode = "DEAD_RECKONING"
            return self.state
        # Scalar adaptive gains are equivalent to a diagonal error-state Kalman
        # update. Reliability comes directly from Android's reported accuracy.
        measurement_variance = max(4.0, float(gnss_accuracy_m) ** 2)
        # GNSS is the authoritative source when available; retain enough IMU
        # influence for smooth 10 Hz output but prevent accumulated MEMS error.
        position_gain = max(0.75, self.state.position_variance_m2 / (self.state.position_variance_m2 + measurement_variance))
        residual_e = gnss_east_m - self.state.east_m; residual_n = gnss_north_m - self.state.north_m
        self.state.east_m += position_gain * residual_e; self.state.north_m += position_gain * residual_n
        self.state.position_variance_m2 *= (1.0 - position_gain)
        if gnss_speed_mps is not None:
            speed_gain = max(0.80, self.state.speed_variance_m2ps2 / (self.state.speed_variance_m2ps2 + 1.0))
            speed_residual = gnss_speed_mps - self.state.speed_mps
            self.state.speed_mps += speed_gain * speed_residual
            self._acc_bias_mps2 -= 0.01 * speed_residual / max(dt, 0.05)
            self.state.speed_variance_m2ps2 *= (1.0 - speed_gain)
        if gnss_heading_rad is not None and self.state.speed_mps > 2.0:
            heading_error = (gnss_heading_rad - self.state.heading_rad + 3.141592653589793) % (2 * 3.141592653589793) - 3.141592653589793
            self.state.heading_rad += 0.60 * heading_error
            self._gyro_bias_rps -= 0.002 * heading_error / max(dt, 0.05)
        self.state.mode = "GNSS_AIDED"
        return self.state
