"""IO-VNBD phone/vehicle paired-drive loader.

The original CSV files remain untouched.  This module exposes a small, stable
schema for every subsequent experiment:
  phone IMU/GPS -> input features
  vehicle speed/position/heading -> reference labels
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


PHONE_COLUMNS = {
    "phone_lat_deg": "GPS LATITUDE (degrees)",
    "phone_lon_deg": " GPS LONGITUDE (degrees)",
    "phone_alt_m": " GPS ALTITUDE (m)",
    "phone_speed_kmh": " GPS SPEED (Kmh)",
    "phone_accuracy_m": " GPS ACCURACY (m)",
    "phone_bearing_deg": " GPS ORIENTATION (°)",
    "phone_time_ms": " TIME SINCE START (ms)",
    "acc_x_mps2": " ACCELEROMETER X (m/s²) ",
    "acc_y_mps2": " ACCELEROMETER Y (m/s²)",
    "acc_z_mps2": " ACCELEROMETER Z (m/s²)",
    "gravity_x_mps2": " GRAVITY X (m/s²)",
    "gravity_y_mps2": " GRAVITY Y (m/s²)",
    "gravity_z_mps2": " GRAVITY Z (m/s²)",
    "gyro_yaw_rps": " GYROSCOPE Yaw (rad/s)",
    "gyro_pitch_rps": " GYROSCOPE Pitch (rad/s)",
    "gyro_roll_rps": " GYROSCOPE Roll (rad/s)",
    "mag_x_ut": " MAGNETIC FIELD X (μT)",
    "mag_y_ut": " MAGNETIC FIELD Y (μT)",
    "mag_z_ut": " MAGNETIC FIELD Z (μT)",
    "orientation_yaw_deg": " ORIENTATION (Yaw) (°)",
    "orientation_pitch_deg": " ORIENTATION (Pitch) (°)",
    "orientation_roll_deg": " ORIENTATION (Roll ) (°)",
}

VEHICLE_COLUMNS = {
    "vehicle_time_s": " Time Since Start of Day (seconds)",
    "ref_lat_deg": " Latitude (degrees)",
    "ref_lon_deg": " Longitude (degrees)",
    "ref_speed_kmh": " Velocity (km/hr)",
    "ref_heading_deg": " Heading (degrees)",
    "ref_yaw_rate_dps": " Yaw Rate (deg/sec)",
    "ref_indicated_speed_kmh": " Indicated Vehicle Speed (km/hr)",
    "ref_long_acc_g": " Indicated Longitudinal Acceleration (g)",
    "ref_lat_acc_g": " Indicated Lateral Acceleration (g)",
}


@dataclass(frozen=True)
class DrivePair:
    name: str
    phone_csv: Path
    vehicle_csv: Path


def _read(path: Path) -> pd.DataFrame:
    # IO-VNBD headers contain degree/superscript characters stored as Latin-1.
    frame = pd.read_csv(path, encoding="latin-1")
    # Dataset headers have inconsistent leading/trailing whitespace. Normalise it
    # once while retaining the published spelling and units.
    frame.columns = [str(column).strip().replace("Â", "").replace("Î¼", "μ") for column in frame.columns]
    return frame


def _select(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    columns = {key: source.strip() for key, source in mapping.items()}
    absent = [source for source in columns.values() if source not in frame.columns]
    if absent:
        raise ValueError(f"Unexpected IO-VNBD header. Missing: {absent}")
    output = frame[[source for source in columns.values()]].copy()
    output.columns = list(columns)
    return output.apply(pd.to_numeric, errors="coerce")


def find_pairs(dataset_root: Path) -> list[DrivePair]:
    """Find directories that contain a synchronised S-/V- CSV pair."""
    pairs: list[DrivePair] = []
    for directory in dataset_root.rglob("*"):
        if not directory.is_dir():
            continue
        phones = sorted(directory.glob("S-*.csv"))
        vehicles = sorted(directory.glob("V-*.csv"))
        if len(phones) == 1 and len(vehicles) == 1:
            pairs.append(DrivePair(directory.name, phones[0], vehicles[0]))
    return pairs


def load_drive(pair: DrivePair) -> pd.DataFrame:
    """Load a paired drive row-for-row; data is synchronised at 10 Hz."""
    phone = _select(_read(pair.phone_csv), PHONE_COLUMNS)
    vehicle = _select(_read(pair.vehicle_csv), VEHICLE_COLUMNS)
    length = min(len(phone), len(vehicle))
    combined = pd.concat(
        [phone.iloc[:length].reset_index(drop=True), vehicle.iloc[:length].reset_index(drop=True)],
        axis=1,
    )
    combined.insert(0, "sample", range(length))
    combined.insert(1, "elapsed_s", combined["phone_time_ms"] / 1000.0)
    combined["drive"] = pair.name
    # IO-VNBD's published phone header says Kmh, but its paired recordings store
    # this channel in m/s (it is consistently ~3.6x below vehicle km/h). Keep
    # the canonical field semantically correct for downstream experiments.
    phone_speed = combined["phone_speed_kmh"].to_numpy()
    vehicle_speed = combined["ref_speed_kmh"].to_numpy()
    valid = (phone_speed > 1.0) & (vehicle_speed > 1.0)
    if valid.sum() > 50:
        ratio = float(pd.Series(vehicle_speed[valid] / phone_speed[valid]).median())
        if 3.2 <= ratio <= 4.0:
            combined["phone_speed_kmh"] *= 3.6
    return combined


def choose_pair(pairs: Iterable[DrivePair], name: str | None) -> DrivePair:
    pairs = list(pairs)
    if not pairs:
        raise FileNotFoundError("No matched S-/V- drive pairs found")
    if name is None:
        return pairs[0]
    matches = [pair for pair in pairs if pair.name.lower() == name.lower()]
    if not matches:
        available = ", ".join(pair.name for pair in pairs[:20])
        raise ValueError(f"Drive '{name}' not found. Examples: {available}")
    return matches[0]
