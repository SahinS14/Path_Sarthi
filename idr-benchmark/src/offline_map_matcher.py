"""Offline OSM road constraint used by NavRakshak's edge engine.

This deliberately uses a local OSM JSON file at inference time. It has no
network dependency and never reads trajectory ground truth.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


EARTH_RADIUS_M = 6_371_000.0


def angle_difference(a: float, b: np.ndarray) -> np.ndarray:
    return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)


@dataclass(frozen=True)
class Segment:
    a: np.ndarray
    b: np.ndarray
    heading: float
    highway: str


class OfflineRoadMatcher:
    """Heading-aware nearest-road matcher with continuity penalty."""
    def __init__(self, segments: list[Segment], distance_sigma_m: float = 25.0):
        if not segments:
            raise ValueError("The offline road extract contains no usable road segments")
        self.segments = segments
        self.distance_sigma_m = distance_sigma_m
        self._a = np.array([segment.a for segment in segments])
        self._b = np.array([segment.b for segment in segments])
        self._heading = np.array([segment.heading for segment in segments])
        self._last_index: int | None = None

    @classmethod
    def from_overpass_json(cls, path: Path, origin_lat_deg: float, origin_lon_deg: float) -> "OfflineRoadMatcher":
        payload = json.loads(path.read_text())
        lat0, lon0 = np.deg2rad([origin_lat_deg, origin_lon_deg])

        def xy(point):
            north = EARTH_RADIUS_M * (np.deg2rad(point["lat"]) - lat0)
            east = EARTH_RADIUS_M * (np.deg2rad(point["lon"]) - lon0) * np.cos(lat0)
            return np.array([east, north])

        segments: list[Segment] = []
        for way in payload.get("elements", []):
            geometry = way.get("geometry", [])
            for start, end in zip(geometry, geometry[1:]):
                a, b = xy(start), xy(end)
                vector = b - a
                if np.linalg.norm(vector) < 0.3:
                    continue
                # Heading is clockwise from north, consistent with navigation.
                heading = float(np.arctan2(vector[0], vector[1]))
                segments.append(Segment(a, b, heading, way.get("tags", {}).get("highway", "unknown")))
        return cls(segments)

    def match(self, east_m: float, north_m: float, heading_rad: float) -> tuple[float, float, float, str]:
        point = np.array([east_m, north_m])
        vector = self._b - self._a
        length_sq = np.sum(vector * vector, axis=1)
        fraction = np.clip(np.sum((point - self._a) * vector, axis=1) / length_sq, 0.0, 1.0)
        projected = self._a + fraction[:, None] * vector
        distance = np.linalg.norm(projected - point, axis=1)
        # A vehicle may travel in either direction on a road segment.
        heading_error = np.minimum(angle_difference(heading_rad, self._heading), angle_difference(heading_rad, self._heading + np.pi))
        score = (distance / self.distance_sigma_m) ** 2 + 0.35 * (heading_error / 0.7) ** 2
        if self._last_index is not None:
            # Avoid arbitrary jumps between equally close parallel/side roads.
            last_midpoint = (self._a[self._last_index] + self._b[self._last_index]) / 2
            score += 0.002 * np.linalg.norm((self._a + self._b) / 2 - last_midpoint, axis=1)
        index = int(np.argmin(score)); self._last_index = index
        matched_heading = self._heading[index]
        if angle_difference(heading_rad, np.array([matched_heading + np.pi]))[0] < angle_difference(heading_rad, np.array([matched_heading]))[0]:
            matched_heading += np.pi
        return float(projected[index, 0]), float(projected[index, 1]), float(matched_heading), self.segments[index].highway
