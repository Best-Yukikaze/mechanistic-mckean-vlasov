"""Discrete particle energies for mechanics validation."""

from __future__ import annotations

import numpy as np

from .controlled_potential import ControlledPotentialBackend, ZeroControlledPotential
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
) -> float:
    """Return kinetic, passive/controlled potential, and pair energy in J."""

    selected_external = external or ZeroExternalPotential()
    selected_controlled = controlled_potential or ZeroControlledPotential()
    kinetic = 0.5 * parameters.particle_mass_kg * np.sum(
        state.velocities_m_per_s**2
    )
    external_energy = np.sum(selected_external.potential_joule(state.positions_m))
    controlled_energy = np.sum(
        selected_controlled.potential_joule(state.positions_m, control)
    )
    displacement = state.positions_m[:, None, :] - state.positions_m[None, :, :]
    off_diagonal = ~np.eye(state.positions_m.shape[0], dtype=bool)
    pair_values = pair_potential.potential_joule(displacement[off_diagonal])
    pair_energy = 0.5 * np.sum(pair_values) / state.positions_m.shape[0]
    return float(kinetic + external_energy + controlled_energy + pair_energy)
