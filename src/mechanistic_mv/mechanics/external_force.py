"""Passive external-potential backends with explicit energy/force units."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ExternalPotential(Protocol):
    name: str

    def potential_joule(self, positions_m: np.ndarray) -> np.ndarray: ...

    def force_newton(self, positions_m: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class ZeroExternalPotential:
    """No passive external field."""

    name: str = "zero_external_potential"

    def potential_joule(self, positions_m: np.ndarray) -> np.ndarray:
        points = _points(positions_m)
        return np.zeros(points.shape[:-1], dtype=np.float64)

    def force_newton(self, positions_m: np.ndarray) -> np.ndarray:
        return np.zeros_like(_points(positions_m))


@dataclass(frozen=True, slots=True)
class HarmonicTestPotential:
    """TEST_ONLY_NOT_FINAL_PHYSICS harmonic trap for analytic checks."""

    centre_m: tuple[float, float]
    stiffness_newton_per_m: float
    name: str = "test_only_harmonic_external_potential"
    physical_status: str = "TEST_ONLY_NOT_FINAL_PHYSICS"

    def __post_init__(self) -> None:
        centre = np.asarray(self.centre_m, dtype=np.float64)
        if centre.shape != (2,) or not np.all(np.isfinite(centre)):
            raise ValueError("centre_m must be a finite two-vector")
        if (
            not np.isfinite(self.stiffness_newton_per_m)
            or self.stiffness_newton_per_m <= 0.0
        ):
            raise ValueError("stiffness must be finite and positive")

    def potential_joule(self, positions_m: np.ndarray) -> np.ndarray:
        displacement = _points(positions_m) - np.asarray(self.centre_m)
        return 0.5 * self.stiffness_newton_per_m * np.sum(
            displacement * displacement, axis=-1
        )

    def force_newton(self, positions_m: np.ndarray) -> np.ndarray:
        displacement = _points(positions_m) - np.asarray(self.centre_m)
        return -self.stiffness_newton_per_m * displacement


def _points(values: np.ndarray) -> np.ndarray:
    points = np.asarray(values, dtype=np.float64)
    if points.shape[-1:] != (2,) or not np.all(np.isfinite(points)):
        raise ValueError("positions must be finite and end in two components")
    return points
