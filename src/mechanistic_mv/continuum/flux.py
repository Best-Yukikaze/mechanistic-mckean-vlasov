"""Conservative face fluxes with impermeable outer and solid boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..mechanics.geometry import CartesianGrid


@dataclass(frozen=True, slots=True)
class FaceMasks:
    open_x: np.ndarray
    open_y: np.ndarray


@dataclass(frozen=True, slots=True)
class FaceFluxes:
    x_per_m_s: np.ndarray
    y_per_m_s: np.ndarray
    max_abs_velocity_x_m_per_s: float
    max_abs_velocity_y_m_per_s: float


def build_face_masks(fluid_mask: np.ndarray) -> FaceMasks:
    """Cache which cell faces connect two fluid cells."""

    fluid = np.asarray(fluid_mask)
    if fluid.ndim != 2 or fluid.dtype != bool or not np.any(fluid):
        raise ValueError("fluid_mask must be a non-empty two-dimensional boolean array")
    ny, nx = fluid.shape
    open_x = np.zeros((ny, nx + 1), dtype=bool)
    open_y = np.zeros((ny + 1, nx), dtype=bool)
    open_x[:, 1:nx] = fluid[:, :-1] & fluid[:, 1:]
    open_y[1:ny, :] = fluid[:-1, :] & fluid[1:, :]
    return FaceMasks(open_x=open_x, open_y=open_y)


def compute_face_fluxes(
    density_per_m2: np.ndarray,
    effective_potential_joule: np.ndarray,
    grid: CartesianGrid,
    *,
    diffusion_m2_per_s: float,
    mobility_m_per_newton_second: float,
    face_masks: FaceMasks,
) -> FaceFluxes:
    """Return ``J=-D grad(rho)+rho M(-grad Phi+F_control)`` on faces.

    The advective density is first-order upwind, while the diffusive gradient
    is centred. Closed outer/solid faces are set exactly to zero.
    """

    density = np.asarray(density_per_m2, dtype=np.float64)
    potential = np.asarray(effective_potential_joule, dtype=np.float64)
    shape = (grid.ny, grid.nx)
    if density.shape != shape or potential.shape != shape:
        raise ValueError("density and effective potential must match the grid")
    if np.any(density < 0.0) or not np.all(np.isfinite(density)):
        raise ValueError("density must be finite and non-negative")
    if not np.all(np.isfinite(potential)):
        raise ValueError("effective potential must be finite")

    flux_x = np.zeros((grid.ny, grid.nx + 1), dtype=np.float64)
    flux_y = np.zeros((grid.ny + 1, grid.nx), dtype=np.float64)

    passive_force_x = -(potential[:, 1:] - potential[:, :-1]) / grid.dx_m
    velocity_x = mobility_m_per_newton_second * passive_force_x
    upwind_x = np.where(velocity_x >= 0.0, density[:, :-1], density[:, 1:])
    interior_x = (
        -diffusion_m2_per_s * (density[:, 1:] - density[:, :-1]) / grid.dx_m
        + velocity_x * upwind_x
    )
    flux_x[:, 1:grid.nx] = np.where(
        face_masks.open_x[:, 1:grid.nx], interior_x, 0.0
    )

    passive_force_y = -(potential[1:, :] - potential[:-1, :]) / grid.dy_m
    velocity_y = mobility_m_per_newton_second * passive_force_y
    upwind_y = np.where(velocity_y >= 0.0, density[:-1, :], density[1:, :])
    interior_y = (
        -diffusion_m2_per_s * (density[1:, :] - density[:-1, :]) / grid.dy_m
        + velocity_y * upwind_y
    )
    flux_y[1:grid.ny, :] = np.where(
        face_masks.open_y[1:grid.ny, :], interior_y, 0.0
    )

    open_vx = velocity_x[face_masks.open_x[:, 1:grid.nx]]
    open_vy = velocity_y[face_masks.open_y[1:grid.ny, :]]
    return FaceFluxes(
        x_per_m_s=flux_x,
        y_per_m_s=flux_y,
        max_abs_velocity_x_m_per_s=float(np.max(np.abs(open_vx))) if open_vx.size else 0.0,
        max_abs_velocity_y_m_per_s=float(np.max(np.abs(open_vy))) if open_vy.size else 0.0,
    )
