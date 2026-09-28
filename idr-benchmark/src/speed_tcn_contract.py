"""Single SpeedTCN feature contract shared by training metadata and Android.

IO-VNBD publishes accelerometer/gravity in m/s² and gyro channels in rad/s at
nominal 10 Hz. Its yaw/pitch/roll-to-Android device-axis relationship is not
documented locally; the Android mapping below is explicitly provisional.
"""
import numpy as np

FEATURES = ('acc_x_mps2','acc_y_mps2','acc_z_mps2','gravity_x_mps2','gravity_y_mps2','gravity_z_mps2','gyro_yaw_rps','gyro_pitch_rps','gyro_roll_rps')
SAMPLE_HZ = 10
WINDOW_SAMPLES = 20
OUTPUT_SEMANTICS = 'speed_delta_mps_over_2s'
LEGACY_ANDROID_GYRO_ORDER = (2, 1, 0)  # dataset yaw,pitch,roll ← Android z,y,x; UNVERIFIED

def android_features(acceleration, gravity, gyro):
    a, g, w = (np.asarray(value, dtype=np.float32) for value in (acceleration, gravity, gyro))
    if a.shape != (3,) or g.shape != (3,) or w.shape != (3,) or not np.isfinite(np.r_[a,g,w]).all():
        raise ValueError('Expected finite Android acceleration, gravity and gyro XYZ')
    return np.r_[a,g,w[list(LEGACY_ANDROID_GYRO_ORDER)]].astype(np.float32)

def io_vnbd_features(row):
    values=np.asarray([row[name] for name in FEATURES],dtype=np.float32)
    if not np.isfinite(values).all(): raise ValueError('Expected finite IO-VNBD model features')
    return values
