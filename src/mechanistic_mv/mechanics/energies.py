"""Discrete particle energies for mechanics validation."""

from __future__ import annotations

import numpy as np

from .controlled_potential import ControlledPotentialBackend, ZeroControlledPotential
from .density_scaling import (
    DensityConvention,
    validate_pair_potential_scaling,
)
from .external_force import ExternalPotential, ZeroExternalPotential
from .pair_potential import PairPotential
from .parameters import PhysicalParameters
from .particle_model import ParticleMechanicalState


def discrete_particle_energy_joule(
    state: ParticleMechanicalState,
    parameters: PhysicalParameters,
    pair_potential: PairPotential,
    *,
    external: ExternalPotential | None = None,
    controlled_potential: ControlledPotentialBackend | None = None,
    control: np.ndarray | None = None,
    density_convention: DensityConvention = DensityConvention.PROBABILITY,
) -> float:
    """Return total particle energy in J under one interaction convention.

    Pair energy is ``sum(W_Kac)/(2N)`` for probability density and
    ``sum(W_pair)/2`` for number density. With ``W_Kac=N*W_pair`` the two
    total particle energies are identical.
    """

    selected_external = external or ZeroExternalPotential()
    selected_controlled = controlled_potential or ZeroControlledPotential()
    kinetic = 0.5 * parameters.particle_mass_kg * np.sum(
        state.velocities_m_per_s**2
    )
    external_energy = np.sum(selected_external.potential_joule(state.positions_m))
    controlled_energy = np.sum(
        selected_controlled.potential_joule(state.positions_m, control)
    )
    count = state.positions_m.shape[0]
    validate_pair_potential_scaling(
        pair_potential,
        density_convention,
        population_count=count,
    )
    displacement = state.positions_m[:, None, :] - state.positions_m[None, :, :]
    off_diagonal = ~np.eye(count, dtype=bool)
    pair_values = pair_potential.potential_joule(displacement[off_diagonal])
    divisor = count if density_convention is DensityConvention.PROBABILITY else 1
    pair_energy = 0.5 * np.sum(pair_values) / divisor
    return float(kinetic + external_energy + controlled_energy + pair_energy)
