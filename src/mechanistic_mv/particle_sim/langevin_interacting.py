"""Overdamped interacting Langevin dynamics in SI units."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..mechanics.controlled_potential import (
    ControlledPotentialBackend,
    ZeroControlledPotential,
)
from ..mechanics.density_scaling import DensityConvention
from ..mechanics.external_force import ExternalPotential, ZeroExternalPotential
from ..mechanics.geometry import (
    RectangleObstacle,
    RectangularDomain,
    enforce_particle_no_flux,
)
from ..mechanics.pair_potential import PairPotential, mean_field_pair_force_newton
from ..mechanics.parameters import PhysicalParameters


@dataclass(frozen=True, slots=True)
class LangevinStepDiagnostics:
    collision_count: int
    max_deterministic_displacement_m: float
    rms_brownian_displacement_m: float


def overdamped_langevin_step(
    positions_m: np.ndarray,
    parameters: PhysicalParameters,
    pair_potential: PairPotential,
    domain: RectangularDomain,
    *,
    dt_s: float,
    rng: np.random.Generator,
    obstacles: Iterable[RectangleObstacle] = (),
    external: ExternalPotential | None = None,
    controlled_potential: ControlledPotentialBackend | None = None,
    control: np.ndarray | None = None,
    pair_chunk_size: int = 64,
    density_convention: DensityConvention = DensityConvention.PROBABILITY,
) -> tuple[np.ndarray, LangevinStepDiagnostics]:
    """Euler-Maruyama step for the overdamped interacting system.

    ``dX=M(F_ext+F_pair+F_control)dt+sqrt(2D) dB``. The pair term is
    ``(1/N) sum F_Kac`` in probability mode and ``sum F_pair`` in number mode.
    """

    positions = np.asarray(positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2 or positions.shape[0] < 2:
        raise ValueError("positions_m must have shape (N, 2), N >= 2")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions_m must be finite")
    dt = float(dt_s)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")
    selected_external = external or ZeroExternalPotential()
    selected_controlled_potential = controlled_potential or ZeroControlledPotential()
    force = (
        selected_external.force_newton(positions)
        + mean_field_pair_force_newton(
            positions,
            pair_potential,
            chunk_size=pair_chunk_size,
            density_convention=density_convention,
        )
        + selected_controlled_potential.force_newton(positions, control)
    )
    deterministic = parameters.mobility_m_per_newton_second * force * dt
    noise_scale = np.sqrt(2.0 * parameters.diffusion_m2_per_s * dt)
    proposed = positions + deterministic + noise_scale * rng.standard_normal(
        positions.shape
    )
    constrained, collisions = enforce_particle_no_flux(
        positions, proposed, domain, obstacles
    )
    diagnostics = LangevinStepDiagnostics(
        collision_count=collisions,
        max_deterministic_displacement_m=float(
            np.max(np.linalg.norm(deterministic, axis=1))
        ),
        rms_brownian_displacement_m=float(np.sqrt(2.0) * noise_scale),
    )
    return constrained, diagnostics
