"""Chemical conjugacy and reference-volume distinctions for Model II."""

from __future__ import annotations

import numpy as np

from .constitutive import deformation_invariants
from .parameters import HydrogelParameters


def gibbs_conjugate_solvent_content_dimensionless(
    deformation_gradient: np.ndarray,
    parameters: HydrogelParameters,
) -> np.ndarray:
    """Return ``-d(Delta G)/d(Delta mu/kT)=J-phi0``.

    This is the solvent content conjugate to the displayed page-6 Gibbs
    density on its initial-swelling reference-volume convention.
    """

    jacobian = deformation_invariants(deformation_gradient, parameters).jacobian
    return jacobian - parameters.initial_polymer_volume_fraction


def source_dry_solvent_content_times_molecular_volume(
    deformation_gradient: np.ndarray,
    parameters: HydrogelParameters,
) -> np.ndarray:
    """Return PDF page-8 ``nu*C_dry=(J-phi0)/phi0``.

    It differs from the Gibbs conjugate above by ``1/phi0`` because the source
    uses a dry-reference concentration on page 8.  The molecular volume ``nu``
    is not supplied separately, so this API intentionally remains dimensionless.
    """

    return (
        gibbs_conjugate_solvent_content_dimensionless(
            deformation_gradient, parameters
        )
        / parameters.initial_polymer_volume_fraction
    )


def independent_helmholtz_chemical_potential_over_kbt(
    deformation_gradient: np.ndarray,
    solvent_concentration_per_m3: np.ndarray,
    parameters: HydrogelParameters,
) -> np.ndarray:
    """Reject ``mu=partial_C Psi`` without an independent source closure."""

    del deformation_gradient, solvent_concentration_per_m3, parameters
    raise NotImplementedError(
        "the supplied source does not uniquely define independent F and C, so "
        "partial_C Psi cannot be evaluated"
    )
