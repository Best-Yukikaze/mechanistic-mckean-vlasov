"""Discrete McKean--Vlasov free-energy diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..mechanics.density_scaling import DensityConvention, expected_density_mass
from ..mechanics.geometry import CartesianGrid


@dataclass(frozen=True, slots=True)
class FreeEnergyComponents:
    entropy_joule: float
    external_joule: float
    interaction_joule: float

    @property
    def total_joule(self) -> float:
        return self.entropy_joule + self.external_joule + self.interaction_joule


def free_energy_components(
    density_per_m2: np.ndarray,
    grid: CartesianGrid,
    thermal_energy_joule: float,
    reference_density_per_m2: float,
    external_potential_joule: np.ndarray,
    pair_convolution_joule: np.ndarray,
    *,
    fluid_mask: np.ndarray | None = None,
    density_convention: DensityConvention = DensityConvention.PROBABILITY,
    population_count: int | None = None,
) -> FreeEnergyComponents:
    """Discretize the free energy for probability or number density.

    For ``n=N*rho`` and ``W_Kac=N*W_pair``, using
    ``n_ref=N*rho_ref`` gives ``F_number=N*F_probability``.
    """

    density = np.asarray(density_per_m2, dtype=np.float64)
    external = np.asarray(external_potential_joule, dtype=np.float64)
    interaction = np.asarray(pair_convolution_joule, dtype=np.float64)
    shape = (grid.ny, grid.nx)
    if density.shape != shape or external.shape != shape or interaction.shape != shape:
        raise ValueError("energy arrays must have grid shape")
    if np.any(density < 0.0) or not np.all(np.isfinite(density)):
        raise ValueError("density must be finite and non-negative")
    expected_density_mass(density_convention, population_count)
    scalars = np.asarray(
        [thermal_energy_joule, reference_density_per_m2],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(scalars)) or np.any(scalars <= 0.0):
        raise ValueError("thermal energy and reference density must be positive")
    if not np.all(np.isfinite(external)) or not np.all(np.isfinite(interaction)):
        raise ValueError("external and interaction potentials must be finite")
    mask = np.ones(shape, dtype=bool) if fluid_mask is None else np.asarray(fluid_mask)
    if mask.shape != shape or mask.dtype != bool:
        raise ValueError("fluid_mask must be boolean with grid shape")
    if np.any(density[~mask] != 0.0):
        raise ValueError("solid cells must contain zero density")
    positive = (density > 0.0) & mask
    entropy_integrand = np.zeros_like(density)
    entropy_integrand[positive] = density[positive] * (
        np.log(density[positive] / reference_density_per_m2) - 1.0
    )
    area = grid.cell_area_m2
    return FreeEnergyComponents(
        entropy_joule=float(thermal_energy_joule * np.sum(entropy_integrand) * area),
        external_joule=float(np.sum(density[mask] * external[mask]) * area),
        interaction_joule=float(
            0.5 * np.sum(density[mask] * interaction[mask]) * area
        ),
    )
