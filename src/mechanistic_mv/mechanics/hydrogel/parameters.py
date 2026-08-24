"""Explicit, source-labelled parameters for Hong/Guo gel Model II."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TEST_ONLY_NOT_CALIBRATED = "TEST_ONLY_NOT_CALIBRATED"


@dataclass(frozen=True, slots=True)
class HydrogelParameters:
    """Material inputs appearing in the supplied Model II formula.

    Every value is required from the caller.  No numerical default in the
    supplied slides is a calibrated material value.

    ``network_density_times_solvent_volume`` is :math:`N\nu` (dimensionless),
    ``flory_huggins_chi`` is :math:`\chi` (dimensionless),
    ``initial_polymer_volume_fraction`` is :math:`\phi_0` (dimensionless),
    ``delta_chemical_potential_over_kbt`` is :math:`\Delta\mu/(kT)`
    (dimensionless), and ``thermal_energy_density_pa`` is :math:`kT/\nu`
    in Pa = J/m^3.

    ``calibration_status`` is provenance metadata, not a material parameter.
    Tests must pass ``TEST_ONLY_NOT_CALIBRATED`` explicitly.
    """

    network_density_times_solvent_volume: float
    flory_huggins_chi: float
    initial_polymer_volume_fraction: float
    delta_chemical_potential_over_kbt: float
    thermal_energy_density_pa: float
    calibration_status: str

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.network_density_times_solvent_volume,
                self.flory_huggins_chi,
                self.initial_polymer_volume_fraction,
                self.delta_chemical_potential_over_kbt,
                self.thermal_energy_density_pa,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("all Hydrogel parameters must be finite")
        if self.network_density_times_solvent_volume <= 0.0:
            raise ValueError("N*nu must be positive")
        if not 0.0 < self.initial_polymer_volume_fraction < 1.0:
            raise ValueError("phi0 must lie strictly between zero and one")
        if self.thermal_energy_density_pa <= 0.0:
            raise ValueError("kT/nu must be positive in Pa")
        if not isinstance(self.calibration_status, str) or not self.calibration_status.strip():
            raise ValueError("calibration_status must be a non-empty string")

    @property
    def initial_chemical_potential_over_kbt(self) -> float:
        """Return the stress-free :math:`\mu_0/(kT)` from PDF page 6."""

        n_nu = self.network_density_times_solvent_volume
        phi0 = self.initial_polymer_volume_fraction
        return float(
            n_nu * phi0 ** (1.0 / 3.0)
            + self.flory_huggins_chi * phi0**2
            + (1.0 - n_nu) * phi0
            + np.log1p(-phi0)
        )

    @property
    def bath_chemical_potential_over_kbt(self) -> float:
        """Return :math:`\mu/(kT)=\mu_0/(kT)+\Delta\mu/(kT)`."""

        return (
            self.initial_chemical_potential_over_kbt
            + self.delta_chemical_potential_over_kbt
        )
