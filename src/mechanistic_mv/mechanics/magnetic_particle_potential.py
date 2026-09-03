"""SI-audited magnetic single-particle energies for the active validation path.

This module deliberately contains no fitted field magnitude, particle size, or
susceptibility.  A production instance must be built from the external source
provenance gate in :mod:`magnetic_validation`.  Direct constructors remain
useful only for explicitly labelled unit fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np


MU0_N_PER_A2 = 4.0e-7 * math.pi
"""Vacuum permeability in ``N/A^2`` (equivalently ``T m/A``)."""


class MagneticFieldMagnitude2D(Protocol):
    """A sourced scalar ``|B|`` field and its in-plane gradient.

    ``B`` is in T and ``grad_B`` is in T/m.  The force expressions below are
    written for the magnetic-flux-density magnitude rather than a signed field
    component, so callers must supply a non-negative magnitude consistently.
    """

    name: str

    def flux_density_tesla(self, positions_m: np.ndarray) -> np.ndarray: ...

    def gradient_flux_density_tesla_per_m(self, positions_m: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class LinearMagneticParticle:
    """Source-parameterized linear magnetic particle law.

    ``chi_v`` is the dimensionless volume susceptibility and ``particle_volume``
    is in ``m^3``.  The reversible linear-law energy is

    ``V_mag = -chi_v * V_m * B^2 / (2*mu0)`` [J],

    and its in-plane force is

    ``F = chi_v * V_m * B * grad(B) / mu0`` [N].
    """

    chi_v_dimensionless: float
    particle_volume_m3: float
    source_locator: str
    provenance_class: str
    physical_status: str = "SOURCE_PARAMETERIZED_MAGNETIC_LAW"

    def __post_init__(self) -> None:
        _positive_finite(self.chi_v_dimensionless, "chi_v_dimensionless")
        _positive_finite(self.particle_volume_m3, "particle_volume_m3")
        _nonempty(self.source_locator, "source_locator")
        _nonempty(self.provenance_class, "provenance_class")

    def energy_joule_from_flux_density_tesla(self, flux_density_tesla: np.ndarray) -> np.ndarray:
        """Return ``V_mag`` [J] for finite, non-negative ``B`` [T]."""

        field = _field_magnitude(flux_density_tesla)
        return -self.chi_v_dimensionless * self.particle_volume_m3 * field**2 / (2.0 * MU0_N_PER_A2)

    def force_newton_from_flux_density_gradient(
        self,
        flux_density_tesla: np.ndarray,
        gradient_flux_density_tesla_per_m: np.ndarray,
    ) -> np.ndarray:
        """Return ``-grad(V_mag)`` [N] from ``B`` [T] and ``grad(B)`` [T/m]."""

        field = _field_magnitude(flux_density_tesla)
        gradient = _gradient(gradient_flux_density_tesla_per_m, field.shape)
        return (
            self.chi_v_dimensionless * self.particle_volume_m3
            * field[..., np.newaxis] * gradient / MU0_N_PER_A2
        )

    def dipole_moment_A_m2(self, flux_density_tesla: np.ndarray) -> np.ndarray:
        """Return the linear induced moment ``m=chi_v*V_m*B/mu0`` [A m²]."""

        field = _field_magnitude(flux_density_tesla)
        return self.chi_v_dimensionless * self.particle_volume_m3 * field / MU0_N_PER_A2


@dataclass(frozen=True, slots=True)
class TabulatedMagnetizationLaw:
    """Source-backed reversible nonlinear magnetization route.

    Field samples are in T, magnetization samples in A/m, and linear segment
    interpolation is used exactly as declared source-table interpolation.  The
    magnetic energy is ``-V_m integral_0^B M(b) db`` [J], while the force is
    ``V_m M(B) grad(B)`` [N].  No unsourced saturation model is introduced.
    """

    flux_density_samples_tesla: tuple[float, ...]
    magnetization_samples_A_per_m: tuple[float, ...]
    source_locator: str
    provenance_class: str
    physical_status: str = "SOURCE_PARAMETERIZED_NONLINEAR_MAGNETIZATION"

    def __post_init__(self) -> None:
        fields = np.asarray(self.flux_density_samples_tesla, dtype=np.float64)
        magnetization = np.asarray(self.magnetization_samples_A_per_m, dtype=np.float64)
        if fields.ndim != 1 or fields.size < 2 or magnetization.shape != fields.shape:
            raise ValueError("nonlinear magnetization table requires matching one-dimensional samples")
        if not np.all(np.isfinite(fields)) or not np.all(np.isfinite(magnetization)):
            raise ValueError("nonlinear magnetization table must be finite")
        if fields[0] != 0.0 or np.any(np.diff(fields) <= 0.0):
            raise ValueError("nonlinear magnetization field samples must start at 0 T and strictly increase")
        if np.any(magnetization < 0.0):
            raise ValueError("nonlinear magnetization samples must be non-negative")
        _nonempty(self.source_locator, "source_locator")
        _nonempty(self.provenance_class, "provenance_class")

    def magnetization_A_per_m(self, flux_density_tesla: np.ndarray) -> np.ndarray:
        field = _field_magnitude(flux_density_tesla)
        fields = np.asarray(self.flux_density_samples_tesla, dtype=np.float64)
        values = np.asarray(self.magnetization_samples_A_per_m, dtype=np.float64)
        if np.any(field > fields[-1]):
            raise ValueError("nonlinear magnetization query exceeds source-supported field range")
        return np.interp(field, fields, values)

    def integral_magnetization_A_T_per_m(self, flux_density_tesla: np.ndarray) -> np.ndarray:
        """Return ``integral_0^B M(b) db`` using the source-table line segments."""

        field = _field_magnitude(flux_density_tesla)
        knots = np.asarray(self.flux_density_samples_tesla, dtype=np.float64)
        values = np.asarray(self.magnetization_samples_A_per_m, dtype=np.float64)
        if np.any(field > knots[-1]):
            raise ValueError("nonlinear magnetization query exceeds source-supported field range")
        cumulative = np.concatenate(([0.0], np.cumsum(0.5 * (values[:-1] + values[1:]) * np.diff(knots))))
        flat = field.reshape(-1)
        indices = np.searchsorted(knots, flat, side="right") - 1
        indices = np.clip(indices, 0, knots.size - 2)
        b0 = knots[indices]
        b1 = knots[indices + 1]
        m0 = values[indices]
        m1 = values[indices + 1]
        delta = flat - b0
        slope = (m1 - m0) / (b1 - b0)
        integral = cumulative[indices] + m0 * delta + 0.5 * slope * delta**2
        return integral.reshape(field.shape)


@dataclass(frozen=True, slots=True)
class MagneticParticlePotential:
    """External-potential backend compatible with the continuum solver protocol."""

    particle_law: LinearMagneticParticle | TabulatedMagnetizationLaw
    particle_volume_m3: float
    magnetic_field: MagneticFieldMagnitude2D
    name: str = "source_parameterized_magnetic_particle_potential"

    def __post_init__(self) -> None:
        _positive_finite(self.particle_volume_m3, "particle_volume_m3")
        for method in ("flux_density_tesla", "gradient_flux_density_tesla_per_m"):
            if not callable(getattr(self.magnetic_field, method, None)):
                raise TypeError(f"magnetic_field must provide {method}")

    @property
    def physical_status(self) -> str:
        return getattr(self.particle_law, "physical_status")

    def potential_joule(self, positions_m: np.ndarray) -> np.ndarray:
        points = _points(positions_m)
        field = self.magnetic_field.flux_density_tesla(points)
        if isinstance(self.particle_law, LinearMagneticParticle):
            if not math.isclose(self.particle_law.particle_volume_m3, self.particle_volume_m3, rel_tol=0.0, abs_tol=0.0):
                raise ValueError("linear particle law volume disagrees with potential volume")
            return self.particle_law.energy_joule_from_flux_density_tesla(field)
        return -self.particle_volume_m3 * self.particle_law.integral_magnetization_A_T_per_m(field)

    def force_newton(self, positions_m: np.ndarray) -> np.ndarray:
        points = _points(positions_m)
        field = self.magnetic_field.flux_density_tesla(points)
        gradient = self.magnetic_field.gradient_flux_density_tesla_per_m(points)
        if isinstance(self.particle_law, LinearMagneticParticle):
            return self.particle_law.force_newton_from_flux_density_gradient(field, gradient)
        return self.particle_volume_m3 * self.particle_law.magnetization_A_per_m(field)[..., np.newaxis] * _gradient(gradient, field.shape)


def _points(values: np.ndarray) -> np.ndarray:
    points = np.asarray(values, dtype=np.float64)
    if points.shape[-1:] != (2,) or not np.all(np.isfinite(points)):
        raise ValueError("positions must be finite and end in two spatial components")
    return points


def _field_magnitude(values: np.ndarray) -> np.ndarray:
    field = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(field)) or np.any(field < 0.0):
        raise ValueError("magnetic flux-density magnitude must be finite and non-negative in T")
    return field


def _gradient(values: np.ndarray, field_shape: tuple[int, ...]) -> np.ndarray:
    gradient = np.asarray(values, dtype=np.float64)
    if gradient.shape != field_shape + (2,) or not np.all(np.isfinite(gradient)):
        raise ValueError("magnetic flux-density gradient must be finite with shape B.shape + (2,)")
    return gradient


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty source string")
    return value
