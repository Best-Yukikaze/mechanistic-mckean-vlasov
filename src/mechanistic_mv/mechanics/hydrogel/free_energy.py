"""Gibbs and constrained Legendre APIs for gel Model II.

The expanded right-hand side on PDF page 6 and its page-7 stress form a
self-consistent source chain.  The printed page-6 left-hand reference-volume
normalization is ambiguous by a factor of ``phi0``; this module therefore does
not claim that both printed sides define an additional independent API.
"""

from __future__ import annotations

import numpy as np

from .constitutive import deformation_invariants
from .parameters import HydrogelParameters


def _dimensionless_nonchemical_energy(
    deformation_gradient: np.ndarray,
    parameters: HydrogelParameters,
) -> tuple[np.ndarray, np.ndarray]:
    invariants = deformation_invariants(deformation_gradient, parameters)
    first = invariants.first_invariant
    jacobian = invariants.jacobian
    n_nu = parameters.network_density_times_solvent_volume
    phi0 = parameters.initial_polymer_volume_fraction
    chi = parameters.flory_huggins_chi
    value = (
        0.5
        * n_nu
        * (phi0 ** (1.0 / 3.0) * (first - 3.0) - 2.0 * phi0 * np.log(jacobian))
        + (jacobian - phi0) * np.log1p(-phi0 / jacobian)
        - (1.0 - phi0) * np.log1p(-phi0)
        + chi * phi0**2 * (1.0 - 1.0 / jacobian)
    )
    return value, jacobian


def dimensionless_gibbs_free_energy_density(
    deformation_gradient: np.ndarray,
    parameters: HydrogelParameters,
) -> np.ndarray:
    """Return the page-6 dimensionless Gibbs increment ``Delta G``."""

    nonchemical, jacobian = _dimensionless_nonchemical_energy(
        deformation_gradient, parameters
    )
    phi0 = parameters.initial_polymer_volume_fraction
    return (
        nonchemical
        - parameters.initial_chemical_potential_over_kbt * (jacobian - 1.0)
        - parameters.delta_chemical_potential_over_kbt * (jacobian - phi0)
    )


def gibbs_free_energy_density_pa(
    deformation_gradient: np.ndarray,
    parameters: HydrogelParameters,
) -> np.ndarray:
    """Return Gibbs density per initial-swollen reference volume in Pa."""

    return (
        parameters.thermal_energy_density_pa
        * dimensionless_gibbs_free_energy_density(deformation_gradient, parameters)
    )


def reduced_helmholtz_free_energy_density_pa(
    deformation_gradient: np.ndarray,
    parameters: HydrogelParameters,
) -> np.ndarray:
    """Initial-reference Legendre transform along the constrained manifold.

    The conjugate initial-reference solvent content is ``q=J-phi0``.  Thus
    ``Psi_reduced = G + (kT/nu) * (mu/kT) * q``.  This is not an independent
    ``Psi(F, C)`` because the source has already constrained ``C`` through
    ``det(F)``.
    """

    invariants = deformation_invariants(deformation_gradient, parameters)
    conjugate = invariants.jacobian - parameters.initial_polymer_volume_fraction
    return gibbs_free_energy_density_pa(
        deformation_gradient, parameters
    ) + parameters.thermal_energy_density_pa * (
        parameters.bath_chemical_potential_over_kbt * conjugate
    )


def independent_helmholtz_free_energy_density_pa(
    deformation_gradient: np.ndarray,
    solvent_concentration_per_m3: np.ndarray,
    parameters: HydrogelParameters,
) -> np.ndarray:
    """Reject an unsupported independent ``Psi(F, C)`` closure."""

    del deformation_gradient, solvent_concentration_per_m3, parameters
    raise NotImplementedError(
        "the supplied source eliminates C through the swelling constraint and "
        "does not uniquely define an independent Helmholtz Psi(F, C)"
    )
