"""Positivity-aware conservative finite-volume McKean--Vlasov solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..mechanics.controlled_potential import (
    ControlledPotentialBackend,
    ZeroControlledPotential,
)
from ..mechanics.external_force import ExternalPotential, ZeroExternalPotential
from ..mechanics.geometry import CartesianGrid, RectangleObstacle
from ..mechanics.pair_potential import PairPotential
from ..mechanics.parameters import PhysicalParameters
from .convolution import FFTPairConvolver
from .flux import FaceFluxes, build_face_masks, compute_face_fluxes
from .free_energy import FreeEnergyComponents, free_energy_components


@dataclass(frozen=True, slots=True)
class StepDiagnostics:
    substeps: int
    initial_mass: float
    final_mass: float
    minimum_stable_dt_s: float
    minimum_density_per_m2: float
    clipped_negative_mass: float
    maximum_abs_flux_per_m_s: float

    @property
    def absolute_mass_error(self) -> float:
        return abs(self.final_mass - self.initial_mass)


def conservative_update(
    density_per_m2: np.ndarray,
    fluxes: FaceFluxes,
    grid: CartesianGrid,
    dt_s: float,
) -> np.ndarray:
    """Apply one conservative cell-balance update without normalization."""

    density = np.asarray(density_per_m2, dtype=np.float64)
    divergence = (
        (fluxes.x_per_m_s[:, 1:] - fluxes.x_per_m_s[:, :-1]) / grid.dx_m
        + (fluxes.y_per_m_s[1:, :] - fluxes.y_per_m_s[:-1, :]) / grid.dy_m
    )
    return density - float(dt_s) * divergence


class McKeanVlasovSolver:
    """Reusable solver with cached geometry, potential, and FFT kernel."""

    def __init__(
        self,
        grid: CartesianGrid,
        parameters: PhysicalParameters,
        pair_potential: PairPotential,
        *,
        obstacles: Iterable[RectangleObstacle] = (),
        external: ExternalPotential | None = None,
        controlled_potential: ControlledPotentialBackend | None = None,
        cfl_safety: float = 0.9,
    ) -> None:
        self.grid = grid
        self.parameters = parameters
        self.pair_potential = pair_potential
        self.obstacles = tuple(obstacles)
        self.external = external or ZeroExternalPotential()
        self.controlled_potential = controlled_potential or ZeroControlledPotential()
        self.cfl_safety = float(cfl_safety)
        if not np.isfinite(self.cfl_safety) or not 0.0 < self.cfl_safety <= 1.0:
            raise ValueError("cfl_safety must lie in (0, 1]")

        self.fluid_mask = grid.fluid_mask(self.obstacles)
        self.face_masks = build_face_masks(self.fluid_mask)
        x, y = grid.mesh_m()
        self.cell_positions_m = np.stack((x, y), axis=-1)
        self.external_potential_joule = np.asarray(
            self.external.potential_joule(self.cell_positions_m), dtype=np.float64
        )
        if self.external_potential_joule.shape != (grid.ny, grid.nx):
            raise ValueError("external potential must return one value per grid cell")
        self.convolver = FFTPairConvolver(grid, pair_potential)
        fluid_area = np.sum(self.fluid_mask) * grid.cell_area_m2
        self.reference_density_per_m2 = 1.0 / fluid_area

    def pair_convolution_joule(self, density_per_m2: np.ndarray) -> np.ndarray:
        density = self._validated_density(density_per_m2)
        return self.convolver.convolve_joule(density)

    def face_fluxes(
        self, density_per_m2: np.ndarray, control: np.ndarray | None = None
    ) -> tuple[FaceFluxes, np.ndarray]:
        density = self._validated_density(density_per_m2)
        interaction = self.convolver.convolve_joule(density)
        flat_positions = self.cell_positions_m.reshape(-1, 2)
        controlled_potential = np.asarray(
            self.controlled_potential.potential_joule(flat_positions, control),
            dtype=np.float64,
        ).reshape(self.grid.ny, self.grid.nx)
        effective_potential = (
            self.external_potential_joule + controlled_potential + interaction
        )
        fluxes = compute_face_fluxes(
            density,
            effective_potential,
            self.grid,
            diffusion_m2_per_s=self.parameters.diffusion_m2_per_s,
            mobility_m_per_newton_second=(
                self.parameters.mobility_m_per_newton_second
            ),
            face_masks=self.face_masks,
        )
        return fluxes, interaction

    def stable_dt_s(self, fluxes: FaceFluxes) -> float:
        diffusion_rate = 2.0 * self.parameters.diffusion_m2_per_s * (
            1.0 / self.grid.dx_m**2 + 1.0 / self.grid.dy_m**2
        )
        advection_rate = (
            fluxes.max_abs_velocity_x_m_per_s / self.grid.dx_m
            + fluxes.max_abs_velocity_y_m_per_s / self.grid.dy_m
        )
        total_rate = diffusion_rate + advection_rate
        if total_rate <= 0.0:
            return np.inf
        return self.cfl_safety / total_rate

    def step(
        self,
        density_per_m2: np.ndarray,
        dt_s: float,
        *,
        control: np.ndarray | None = None,
    ) -> tuple[np.ndarray, StepDiagnostics]:
        """Advance a requested interval using automatically selected substeps."""

        density = self._validated_density(density_per_m2).copy()
        requested_dt = float(dt_s)
        if not np.isfinite(requested_dt) or requested_dt <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        initial_mass = self.mass(density)
        remaining = requested_dt
        substeps = 0
        minimum_stable_dt = np.inf
        clipped_mass = 0.0
        maximum_abs_flux = 0.0
        while remaining > max(np.finfo(np.float64).eps * requested_dt, 0.0):
            fluxes, _ = self.face_fluxes(density, control)
            maximum_abs_flux = max(
                maximum_abs_flux,
                float(np.max(np.abs(fluxes.x_per_m_s))),
                float(np.max(np.abs(fluxes.y_per_m_s))),
            )
            stable = self.stable_dt_s(fluxes)
            local_dt = min(remaining, stable)
            if not np.isfinite(local_dt) or local_dt <= 0.0:
                raise FloatingPointError("non-positive finite-volume substep")
            candidate = conservative_update(density, fluxes, self.grid, local_dt)
            candidate[~self.fluid_mask] = 0.0
            tolerance = 5.0e-13 * max(float(np.max(density)), 1.0)
            minimum = float(np.min(candidate[self.fluid_mask]))
            if minimum < -tolerance:
                raise FloatingPointError(
                    "finite-volume positivity failure; reduce CFL or inspect forces "
                    f"(minimum density={minimum:.6e})"
                )
            negative = candidate < 0.0
            if np.any(negative):
                clipped_mass += float(
                    -np.sum(candidate[negative]) * self.grid.cell_area_m2
                )
                candidate[negative] = 0.0
            density = candidate
            remaining -= local_dt
            substeps += 1
            minimum_stable_dt = min(minimum_stable_dt, stable)
            if substeps > 1_000_000:
                raise RuntimeError("adaptive step exceeded one million substeps")
        final_mass = self.mass(density)
        mass_tolerance = 1.0e-11 * max(abs(initial_mass), 1.0)
        if abs(final_mass - initial_mass) > mass_tolerance:
            raise FloatingPointError(
                "finite-volume mass conservation failed without normalization"
            )
        return density, StepDiagnostics(
            substeps=substeps,
            initial_mass=initial_mass,
            final_mass=final_mass,
            minimum_stable_dt_s=float(minimum_stable_dt),
            minimum_density_per_m2=float(np.min(density[self.fluid_mask])),
            clipped_negative_mass=clipped_mass,
            maximum_abs_flux_per_m_s=maximum_abs_flux,
        )

    def mass(self, density_per_m2: np.ndarray) -> float:
        density = self._validated_density(density_per_m2)
        return float(np.sum(density) * self.grid.cell_area_m2)

    def free_energy(
        self, density_per_m2: np.ndarray, control: np.ndarray | None = None
    ) -> FreeEnergyComponents:
        density = self._validated_density(density_per_m2)
        interaction = self.convolver.convolve_joule(density)
        controlled = np.asarray(
            self.controlled_potential.potential_joule(
                self.cell_positions_m.reshape(-1, 2), control
            ),
            dtype=np.float64,
        ).reshape(self.grid.ny, self.grid.nx)
        return free_energy_components(
            density,
            self.grid,
            self.parameters.thermal_energy_joule,
            self.reference_density_per_m2,
            self.external_potential_joule + controlled,
            interaction,
            fluid_mask=self.fluid_mask,
        )

    def _validated_density(self, values: np.ndarray) -> np.ndarray:
        density = np.asarray(values, dtype=np.float64)
        if density.shape != (self.grid.ny, self.grid.nx):
            raise ValueError("density must have grid shape")
        if not np.all(np.isfinite(density)) or np.any(density < 0.0):
            raise ValueError("density must be finite and non-negative")
        solid_values = density[~self.fluid_mask]
        if np.any(solid_values != 0.0):
            raise ValueError("solid cells must contain exactly zero density")
        return density
