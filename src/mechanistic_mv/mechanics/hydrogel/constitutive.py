"""Kinematics and a small facade for Hong/Guo Model II."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parameters import HydrogelParameters


@dataclass(frozen=True, slots=True)
class DeformationInvariants:
    """Three-dimensional invariants used by the supplied gel formula."""

    first_invariant: np.ndarray
    jacobian: np.ndarray


def validated_deformation_gradient(
    deformation_gradient: np.ndarray,
    parameters: HydrogelParameters,
) -> np.ndarray:
    """Return finite ``(..., 3, 3)`` F after enforcing the log domain.

    Model II contains ``log(1-phi0/J)``.  Its strict domain is therefore
    ``J=det(F)>phi0``, which is stronger than orientation preservation alone.
    """

    deformation = np.asarray(deformation_gradient, dtype=np.float64)
    if deformation.shape[-2:] != (3, 3):
        raise ValueError("F must have shape (..., 3, 3)")
    if not np.all(np.isfinite(deformation)):
        raise ValueError("F must be finite")
    jacobian = np.linalg.det(deformation)
    if np.any(jacobian <= 0.0):
        raise ValueError("det(F) must be positive")
    if np.any(jacobian <= parameters.initial_polymer_volume_fraction):
        raise ValueError("Model II requires det(F) > phi0")
    return deformation


def deformation_invariants(
    deformation_gradient: np.ndarray,
    parameters: HydrogelParameters,
) -> DeformationInvariants:
    """Return ``I1=F:F`` and ``J=det(F)`` for scalar or batched F."""

    deformation = validated_deformation_gradient(deformation_gradient, parameters)
    return DeformationInvariants(
        first_invariant=np.sum(deformation * deformation, axis=(-2, -1)),
        jacobian=np.linalg.det(deformation),
    )


@dataclass(frozen=True, slots=True)
class HongGuoModelII:
    """Convenience facade that keeps gel variables below the MV density layer."""

    parameters: HydrogelParameters

    def gibbs_free_energy_density_pa(self, deformation_gradient: np.ndarray) -> np.ndarray:
        from .free_energy import gibbs_free_energy_density_pa

        return gibbs_free_energy_density_pa(deformation_gradient, self.parameters)

    def reduced_helmholtz_free_energy_density_pa(
        self, deformation_gradient: np.ndarray
    ) -> np.ndarray:
        from .free_energy import reduced_helmholtz_free_energy_density_pa

        return reduced_helmholtz_free_energy_density_pa(
            deformation_gradient, self.parameters
        )

    def first_piola_stress_pa(self, deformation_gradient: np.ndarray) -> np.ndarray:
        from .stress import first_piola_stress_pa

        return first_piola_stress_pa(deformation_gradient, self.parameters)

    def gibbs_conjugate_solvent_content(
        self, deformation_gradient: np.ndarray
    ) -> np.ndarray:
        from .chemical_potential import (
            gibbs_conjugate_solvent_content_dimensionless,
        )

        return gibbs_conjugate_solvent_content_dimensionless(
            deformation_gradient, self.parameters
        )
