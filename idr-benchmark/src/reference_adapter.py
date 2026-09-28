"""Array boundary to unchanged upstream fixed/adaptive-covariance NumPy IEKF.

Only initial pose/velocity and IMU enter inference. Reference trajectories are
deliberately absent from this API. Inputs must already satisfy a verified body
frame contract. This is offline replay, not a GNSS fusion or Android engine.
"""
import sys
from pathlib import Path
import numpy as np


def run_reference(t, imu, initial_velocity, initial_rpy, *, initial_position=(0, 0, 0),
                  covariance=None, max_gap_s=0.15, source=Path('work/reference/ai-imu-dr')):
    t, imu = np.asarray(t, float), np.asarray(imu, float)
    v0, rpy, p0 = [np.asarray(x, float) for x in (initial_velocity, initial_rpy, initial_position)]
    if t.ndim != 1 or len(t) < 2 or imu.shape != (len(t), 6):
        raise ValueError('Expected N timestamps and N x 6 IMU samples, N >= 2')
    if any(x.shape != (3,) for x in (v0, rpy, p0)):
        raise ValueError('Initial states must be three-vectors')
    if not all(np.isfinite(x).all() for x in (t, imu, v0, rpy, p0)):
        raise ValueError('Nonfinite input')
    if not np.isfinite(max_gap_s) or max_gap_s <= 0 or np.any(np.diff(t) <= 0) or np.any(np.diff(t) > max_gap_s):
        raise ValueError('Nonmonotonic clock or unsupported IMU gap')
    sys.path.insert(0, str(source.resolve() / 'src'))
    from utils_numpy_filter import NUMPYIEKF
    f = NUMPYIEKF()
    covariance = (np.tile([f.cov_lat, f.cov_up], (len(t), 1)) if covariance is None
                  else np.asarray(covariance, float))
    if covariance.shape != (len(t), 2) or not np.isfinite(covariance).all() or np.any(covariance <= 0):
        raise ValueError('Expected positive finite N x 2 measurement variances')
    # Upstream reads v_mes[0] only, and ignores p_mes. Never pass reference arrays.
    velocity_seed = np.zeros((len(t), 3)); velocity_seed[0] = v0
    rotations, velocity, position, *_ = f.run(t-t[0], imu, covariance, velocity_seed,
                                             np.zeros((len(t), 3)), len(t), rpy)
    if not all(np.isfinite(x).all() for x in (rotations, velocity, position)):
        raise ValueError('Nonfinite reference-filter output')
    return dict(rotation=rotations, velocity=velocity, position=position+p0)
