"""N-particle second-order mother model before overdamped reduction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controlled_potential import ControlledPotentialBackend, ZeroControlledPotential
from .density_scaling import DensityConvention
from .external_force import ExternalPotential, ZeroExternalPotential
from .pair_potential import PairPotential, mean_field_pair_force_newton
from .parameters import PhysicalParameters


@dataclass(frozen=True, slots=True)
class ParticleMechanicalState:
    positions_m: np.ndarray
    velocities_m_per_s: np.ndarray

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_m, dtype=np.float64)
        velocities = np.asarray(self.velocities_m_per_s, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions_m must have shape (N, 2)")
        if velocities.shape != positions.shape:
            raise ValueError("velocities must match positions")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            raise ValueError("mechanical state must be finite")
        object.__setattr__(self, "positions_m", positions.copy())
        object.__setattr__(self, "velocities_m_per_s", velocities.copy())


def deterministic_acceleration_m_per_s2(
    state: ParticleMechanicalState,
    parameters: PhysicalParameters,
    pair_potential: PairPotential,
    *,
    external: ExternalPotential | None = None,
    controlled_potential: ControlledPotentialBackend | None = None,
    control: np.ndarray | None = None,
    pair_chunk_size: int = 64,
    density_convention: DensityConvention = DensityConvention.PROBABILITY,
) -> np.ndarray:
    """Evaluate ``m dV/dt=-gamma V+F_ext+F_pair+F_control``."""

    selected_external = external or ZeroExternalPotential()
    selected_controlled_potential = controlled_potential or ZeroControlledPotential()
    total_force = (
        -parameters.drag_coefficient_kg_per_s * state.velocities_m_per_s
        + selected_external.force_newton(state.positions_m)
        + mean_field_pair_force_newton(
            state.positions_m,
            pair_potential,
            chunk_size=pair_chunk_size,
            density_convention=density_convention,
        )
        + selected_controlled_potential.force_newton(state.positions_m, control)
    )
    return total_force / parameters.particle_mass_kg


def semi_implicit_euler_step(
    state: ParticleMechanicalState,
    acceleration_m_per_s2: np.ndarray,
    dt_s: float,
) -> ParticleMechanicalState:
    """Advance the deterministic mother model for interface validation."""

    acceleration = np.asarray(acceleration_m_per_s2, dtype=np.float64)
    if acceleration.shape != state.positions_m.shape or not np.all(
        np.isfinite(acceleration)
    ):
        raise ValueError("acceleration must match the finite particle state")
    dt = float(dt_s)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    velocity = state.velocities_m_per_s + dt * acceleration
    position = state.positions_m + dt * velocity
    return ParticleMechanicalState(position, velocity)
