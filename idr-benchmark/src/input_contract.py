"""Explicit input conversions and offline label alignment for future adapters.

These helpers never estimate units or clock offsets from reference labels. Label
interpolation is offline-only and must not be used to supply future GNSS at runtime.
"""
import numpy as np


def proper_rotation(value):
    rotation = np.asarray(value, dtype=float)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6)):
        raise ValueError('Expected a finite proper 3D rotation, not an axis reflection')
    return rotation


def convert_imu(acceleration, gyro, *, acceleration_to_body, gyro_to_body,
                acceleration_kind, gravity=None):
    """Return body-frame specific force and angular rate (SI units).

    Separate rotations are deliberate: published channels must not be assumed
    to share a frame. Navigation-frame signals with time-varying attitude need a
    different adapter; this fixed-mount conversion does not silently accept them.
    """
    a, w = np.asarray(acceleration, float), np.asarray(gyro, float)
    if a.ndim != 2 or a.shape[1] != 3 or w.shape != a.shape:
        raise ValueError('Expected matching N x 3 IMU arrays')
    if not np.isfinite(a).all() or not np.isfinite(w).all():
        raise ValueError('Nonfinite IMU input')
    if acceleration_kind == 'linear_acceleration':
        g = np.asarray(gravity, float)
        if g.shape != a.shape or not np.isfinite(g).all():
            raise ValueError('Linear acceleration requires same-frame measured gravity')
        a = a + g
    elif acceleration_kind != 'specific_force':
        raise ValueError('Acceleration convention is unverified')
    return a @ proper_rotation(acceleration_to_body).T, w @ proper_rotation(gyro_to_body).T


def speed_mps(values, unit):
    if unit not in ('m/s', 'km/h'):
        raise ValueError('Explicit verified speed unit required')
    result = np.asarray(values, float)
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError('Invalid speed magnitude')
    return result.copy() if unit == 'm/s' else result / 3.6


def align_labels(phone_time, reference_time, values, *, reference_offset_s, max_gap_s):
    """Interpolate labels onto phone time with no extrapolation or gap bridging.

    Offset definition: reference_time + reference_offset_s is on phone clock.
    Returns labels plus validity mask; rejected samples remain NaN, never zero.
    Angles must be unwrapped before calling. Clock units are seconds.
    """
    target, source = np.asarray(phone_time, float), np.asarray(reference_time, float)
    data = np.asarray(values, float)
    if (target.ndim != 1 or source.ndim != 1 or len(source) < 2
            or data.ndim != 2 or len(data) != len(source)):
        raise ValueError('Expected 1D clocks and N x D labels')
    if (not np.isfinite(reference_offset_s) or not np.isfinite(max_gap_s)
            or max_gap_s <= 0 or not np.isfinite(target).all()
            or not np.isfinite(source).all()
            or np.any(np.diff(target) <= 0) or np.any(np.diff(source) <= 0)):
        raise ValueError('Clocks must be finite and strictly increasing; gap must be positive')
    source = source + reference_offset_s
    right = np.searchsorted(source, target, side='left')
    safe = np.clip(right, 0, len(source)-1)
    exact = (right < len(source)) & (source[safe] == target)
    left = np.clip(right-1, 0, len(source)-1)
    inside = (right > 0) & (right < len(source))
    gap = source[safe] - source[left]
    valid = exact & np.isfinite(data[safe]).all(axis=1)
    interpolate = (~exact & inside & (gap <= max_gap_s)
                   & np.isfinite(data[left]).all(axis=1) & np.isfinite(data[safe]).all(axis=1))
    result = np.full((len(target), data.shape[1]), np.nan)
    result[valid] = data[safe[valid]]
    weight = (target[interpolate]-source[left[interpolate]])/gap[interpolate]
    result[interpolate] = (data[left[interpolate]] * (1-weight[:, None])
                           + data[safe[interpolate]] * weight[:, None])
    return result, valid | interpolate
