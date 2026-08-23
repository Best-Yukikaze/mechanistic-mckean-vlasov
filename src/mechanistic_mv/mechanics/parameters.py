"""SI-valued physical parameters and overdamped consistency checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


BOLTZMANN_CONSTANT_J_PER_K = 1.380649e-23


@dataclass(frozen=True, slots=True)
class PhysicalParameters:
    """Physical parameters in SI units.

    The defaults are representative numerical-validation values, not a final
    experimental calibration. The Einstein relation is enforced by deriving
    diffusion from thermal energy and drag rather than storing an unrelated
    tunable coefficient.
    """

    particle_mass_kg: float = 1.0e-15
    drag_coefficient_kg_per_s: float = 1.0e-8
    temperature_kelvin: float = 298.15
    boltzmann_constant_joule_per_kelvin: float = BOLTZMANN_CONSTANT_J_PER_K

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.particle_mass_kg,
                self.drag_coefficient_kg_per_s,
                self.temperature_kelvin,
                self.boltzmann_constant_joule_per_kelvin,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("all physical parameters must be finite and positive")

    @property
    def mobility_m_per_newton_second(self) -> float:
        """Overdamped mobility ``M=1/gamma`` in m/(N s)."""

        return 1.0 / self.drag_coefficient_kg_per_s

    @property
    def thermal_energy_joule(self) -> float:
        """Return ``k_B T`` in joules."""

        return self.boltzmann_constant_joule_per_kelvin * self.temperature_kelvin

    @property
    def diffusion_m2_per_s(self) -> float:
        """Einstein diffusion ``D=M k_B T`` in m^2/s."""

        return self.mobility_m_per_newton_second * self.thermal_energy_joule

    @property
    def velocity_relaxation_time_s(self) -> float:
        """Inertial relaxation time ``tau_m=m/gamma`` in seconds."""

        return self.particle_mass_kg / self.drag_coefficient_kg_per_s

    def overdamped_ratio(self, characteristic_time_s: float) -> float:
        """Return ``tau_m/T_c`` after validating the characteristic time."""

        time = float(characteristic_time_s)
        if not np.isfinite(time) or time <= 0.0:
            raise ValueError("characteristic_time_s must be finite and positive")
        return self.velocity_relaxation_time_s / time

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values.update(
            {
                "mobility_m_per_newton_second": self.mobility_m_per_newton_second,
                "thermal_energy_joule": self.thermal_energy_joule,
                "diffusion_m2_per_s": self.diffusion_m2_per_s,
                "velocity_relaxation_time_s": self.velocity_relaxation_time_s,
            }
        )
        return values

