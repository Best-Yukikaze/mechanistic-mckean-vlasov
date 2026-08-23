"""Replaceable pair-potential backends and mean-field particle forces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class PairPotential(Protocol):
    """Physical contract: energy in J and force in N."""

    name: str
    physical_status: str

    def potential_joule(self, displacement_m: np.ndarray) -> np.ndarray: ...

    def force_newton(self, displacement_m: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class ZeroPairPotential:
    """Physical null-interaction limit used for Fokker--Planck checks."""

    name: str = "zero_pair_potential"
    physical_status: str = "physical null interaction; W=0 special case"

    def potential_joule(self, displacement_m: np.ndarray) -> np.ndarray:
        displacement = _displacements(displacement_m)
        return np.zeros(displacement.shape[:-1], dtype=np.float64)

    def force_newton(self, displacement_m: np.ndarray) -> np.ndarray:
        return np.zeros_like(_displacements(displacement_m))


@dataclass(frozen=True, slots=True)
class TestOnlyGaussianRepulsion:
    """TEST_ONLY_NOT_FINAL_PHYSICS repulsion for numerical validation.

    ``W(r)=A exp(-|r|^2/(2 ell^2))`` and ``F=-grad W``. It is not a
    calibrated colloidal, magnetic, or microgel constitutive law.
    """

    energy_scale_joule: float
    length_scale_m: float
    name: str = "test_only_gaussian_repulsion"
    physical_status: str = "TEST_ONLY_NOT_FINAL_PHYSICS"

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.energy_scale_joule, self.length_scale_m], dtype=np.float64
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("Gaussian test-potential scales must be positive")

    def potential_joule(self, displacement_m: np.ndarray) -> np.ndarray:
        displacement = _displacements(displacement_m)
        squared_radius = np.sum(displacement * displacement, axis=-1)
        return self.energy_scale_joule * np.exp(
            -0.5 * squared_radius / self.length_scale_m**2
        )

    def force_newton(self, displacement_m: np.ndarray) -> np.ndarray:
        displacement = _displacements(displacement_m)
        potential = self.potential_joule(displacement)
        return (
            potential[..., None]
            * displacement
            / self.length_scale_m**2
        )


def _displacements(values: np.ndarray) -> np.ndarray:
    displacement = np.asarray(values, dtype=np.float64)
    if displacement.shape[-1:] != (2,):
        raise ValueError("displacements must end with two Cartesian components")
    if not np.all(np.isfinite(displacement)):
        raise ValueError("displacements must be finite")
    return displacement


def mean_field_pair_force_newton(
    positions_m: np.ndarray,
    potential: PairPotential,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    """Return ``(1/N) sum_{j != i} F_pair(X_i-X_j)`` in newtons."""

    positions = _displacements(positions_m)
    if positions.ndim != 2 or positions.shape[0] < 2:
        raise ValueError("positions_m must have shape (N, 2), N >= 2")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if isinstance(potential, ZeroPairPotential):
        return np.zeros_like(positions)
    count = positions.shape[0]
    result = np.empty_like(positions)
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        displacement = positions[start:stop, None, :] - positions[None, :, :]
        forces = potential.force_newton(displacement)
        local_rows = np.arange(stop - start)
        forces[local_rows, start + local_rows, :] = 0.0
        result[start:stop] = np.sum(forces, axis=1) / count
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("pair force produced non-finite values")
    return result
