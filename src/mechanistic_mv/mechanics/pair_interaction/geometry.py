"""Minimal, explicitly parameterized geometry for future two-gel contact data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TwoSphereGeometry:
    """Two undeformed sphere radii in metres; no geometric defaults are assumed."""

    first_radius_m: float
    second_radius_m: float
    calibration_status: str

    def __post_init__(self) -> None:
        radii = np.asarray([self.first_radius_m, self.second_radius_m], dtype=np.float64)
        if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
            raise ValueError("sphere radii must be finite and positive in metres")
        if not isinstance(self.calibration_status, str) or not self.calibration_status.strip():
            raise ValueError("calibration_status must be a non-empty string")

    def surface_gap_m(self, center_distance_m: np.ndarray | float) -> np.ndarray:
        distance = np.asarray(center_distance_m, dtype=np.float64)
        if not np.all(np.isfinite(distance)) or np.any(distance < 0.0):
            raise ValueError("center distances must be finite and non-negative")
        return distance - self.first_radius_m - self.second_radius_m
