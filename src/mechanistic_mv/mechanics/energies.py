"""Discrete particle energies for mechanics validation."""

from __future__ import annotations

import numpy as np

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
) -> float:
    """Return kinetic + external + mean-field pair energy in joules."""

    selected_external = external or ZeroExternalPotential()
    kinetic = 0.5 * parameters.particle_mass_kg * np.sum(
        state.velocities_m_per_s**2
    )
    external_energy = np.sum(selected_external.potential_joule(state.positions_m))
    displacement = state.positions_m[:, None, :] - state.positions_m[None, :, :]
    pair_matrix = pair_potential.potential_joule(displacement)
    np.fill_diagonal(pair_matrix, 0.0)
    pair_energy = 0.5 * np.sum(pair_matrix) / state.positions_m.shape[0]
    return float(kinetic + external_energy + pair_energy)

