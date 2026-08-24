"""Analytic first Piola stress for the source Model II Gibbs density."""

from __future__ import annotations

import numpy as np

from .constitutive import validated_deformation_gradient
from .parameters import HydrogelParameters


def dimensionless_first_piola_stress(
    deformation_gradient: np.ndarray,
    parameters: HydrogelParameters,
) -> np.ndarray:
    """Return ``P/(kT/nu)=partial(Delta G)/partial F`` from PDF page 7."""

    deformation = validated_deformation_gradient(deformation_gradient, parameters)
    jacobian = np.linalg.det(deformation)
    inverse_transpose = np.swapaxes(np.linalg.inv(deformation), -1, -2)
    n_nu = parameters.network_density_times_solvent_volume
    phi0 = parameters.initial_polymer_volume_fraction
    scalar = (
        (1.0 - n_nu) * phi0 / jacobian
        + np.log1p(-phi0 / jacobian)
        + parameters.flory_huggins_chi * phi0**2 / jacobian**2
        - parameters.delta_chemical_potential_over_kbt
        - parameters.initial_chemical_potential_over_kbt
    )
    return (
        n_nu * phi0 ** (1.0 / 3.0) * deformation
        + scalar[..., None, None]
        * jacobian[..., None, None]
        * inverse_transpose
    )


def first_piola_stress_pa(
    deformation_gradient: np.ndarray,
    parameters: HydrogelParameters,
) -> np.ndarray:
    """Return initial-swollen-reference first Piola--Kirchhoff stress in Pa."""

    return parameters.thermal_energy_density_pa * dimensionless_first_piola_stress(
        deformation_gradient, parameters
    )
