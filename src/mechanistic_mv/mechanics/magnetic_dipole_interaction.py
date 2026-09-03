"""Source-parameterized, anisotropic magnetic dipole pair mechanics.

This is a three-dimensional pair law.  It intentionally does *not* pretend
that a raw ``1/r^3`` law is a valid 2-D continuum convolution.  The active
2-D solver may consume it only after :mod:`magnetic_validation` has admitted a
source-backed depth reduction and a zero-separation/contact closure.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .magnetic_particle_potential import MU0_N_PER_A2


@dataclass(frozen=True, slots=True)
class SourceBackedContactGeometry:
    """Hard physical contact scale, in metres; never a fitted smooth length."""

    contact_diameter_m: float
    source_locator: str
    provenance_class: str

    def __post_init__(self) -> None:
        _positive_finite(self.contact_diameter_m, "contact_diameter_m")
        _nonempty(self.source_locator, "source_locator")
        _nonempty(self.provenance_class, "provenance_class")


@dataclass(frozen=True, slots=True)
class FieldOrientedDipoleInteraction:
    """3-D dipolar pair energy with a source-backed hard contact exclusion.

    For a common induced moment magnitude ``m`` [A m²] oriented along the unit
    vector ``b``,

    ``W(r)=mu0*m²/(4*pi*r³) * (1 - 3*(b dot r_hat)²)`` [J].

    The model rejects ``r < contact_diameter`` rather than introducing an
    arbitrary softening length.  This is particle-pair mechanics, not yet a
    continuum-ready 2-D kernel.
    """

    dipole_moment_A_m2: float
    field_orientation_unit_vector_3d: tuple[float, float, float]
    contact: SourceBackedContactGeometry
    source_locator: str
    provenance_class: str
    name: str = "source_parameterized_field_oriented_dipole_pair"
    physical_status: str = "SOURCE_PARAMETERIZED_3D_DIPOLAR_PAIR_NOT_2D_CONTINUUM_READY"
    minimum_supported_distance_m: float = 0.0
    continuum_ready: bool = False

    def __post_init__(self) -> None:
        _positive_finite(self.dipole_moment_A_m2, "dipole_moment_A_m2")
        orientation = np.asarray(self.field_orientation_unit_vector_3d, dtype=np.float64)
        if orientation.shape != (3,) or not np.all(np.isfinite(orientation)):
            raise ValueError("field_orientation_unit_vector_3d must be finite with shape (3,)")
        if not math.isclose(float(np.linalg.norm(orientation)), 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError("field_orientation_unit_vector_3d must have unit norm")
        _nonempty(self.source_locator, "source_locator")
        _nonempty(self.provenance_class, "provenance_class")

    @property
    def contact_diameter_m(self) -> float:
        return self.contact.contact_diameter_m

    def energy_joule(self, separation_m: np.ndarray) -> np.ndarray:
        """Return anisotropic pair energy [J] for source-allowed 3-D separations."""

        vector, radius = self._separation_and_radius(separation_m)
        direction = vector / radius[..., np.newaxis]
        alignment = np.einsum("...i,i->...", direction, self._orientation())
        prefactor = MU0_N_PER_A2 * self.dipole_moment_A_m2**2 / (4.0 * math.pi * radius**3)
        return prefactor * (1.0 - 3.0 * alignment**2)

    def force_newton(self, separation_m: np.ndarray) -> np.ndarray:
        """Return ``-grad_r W`` [N] for source-allowed 3-D separations."""

        vector, radius = self._separation_and_radius(separation_m)
        orientation = self._orientation()
        projection = np.einsum("...i,i->...", vector, orientation)
        prefactor = MU0_N_PER_A2 * self.dipole_moment_A_m2**2 / (4.0 * math.pi)
        return prefactor * (
            3.0 * vector / radius[..., np.newaxis]**5
            + 6.0 * projection[..., np.newaxis] * orientation / radius[..., np.newaxis]**5
            - 15.0 * projection[..., np.newaxis]**2 * vector / radius[..., np.newaxis]**7
        )

    def coupling_lambda_at_contact(self, thermal_energy_joule: float) -> float:
        """Return the dimensionless dipolar energy scale ``mu0*m²/(4*pi*d³*kBT)``."""

        thermal = _positive_finite(thermal_energy_joule, "thermal_energy_joule")
        return (
            MU0_N_PER_A2 * self.dipole_moment_A_m2**2
            / (4.0 * math.pi * self.contact_diameter_m**3 * thermal)
        )

    def _separation_and_radius(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape[-1:] != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("dipole separations must be finite and end in three components")
        radius = np.linalg.norm(vector, axis=-1)
        if np.any(radius < self.contact_diameter_m):
            raise ValueError(
                "dipole pair law is undefined inside source-backed hard contact; "
                "no fitted softening or 2-D zero-separation closure is available"
            )
        return vector, radius

    def _orientation(self) -> np.ndarray:
        return np.asarray(self.field_orientation_unit_vector_3d, dtype=np.float64)


def sphere_volume_m3(diameter_m: float) -> float:
    """Return the geometric sphere volume [m³] from a source-backed diameter."""

    diameter = _positive_finite(diameter_m, "diameter_m")
    return math.pi * diameter**3 / 6.0


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty source string")
    return value
