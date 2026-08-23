"""Mass-normalized initial conditions for continuum validation."""

from __future__ import annotations

import numpy as np

from ..mechanics.geometry import CartesianGrid


def gaussian_density(
    grid: CartesianGrid,
    centre_m: tuple[float, float],
    standard_deviation_m: tuple[float, float] | float,
    *,
    fluid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Create a unit-mass Gaussian density in 1/m^2.

    Normalization is performed once while constructing the initial condition;
    the time integrator must conserve this mass without later renormalization.
    """

    centre = np.asarray(centre_m, dtype=np.float64)
    scale = np.asarray(standard_deviation_m, dtype=np.float64)
    if scale.ndim == 0:
        scale = np.repeat(scale, 2)
    if centre.shape != (2,) or scale.shape != (2,):
        raise ValueError("centre and standard deviation must have two components")
    if not np.all(np.isfinite(centre)) or not np.all(np.isfinite(scale)):
        raise ValueError("Gaussian parameters must be finite")
    if np.any(scale <= 0.0):
        raise ValueError("standard deviations must be positive")
    x, y = grid.mesh_m()
    exponent = -0.5 * (
        ((x - centre[0]) / scale[0]) ** 2
        + ((y - centre[1]) / scale[1]) ** 2
    )
    density = np.exp(exponent)
    mask = _validated_mask(fluid_mask, density.shape)
    density[~mask] = 0.0
    mass = float(np.sum(density) * grid.cell_area_m2)
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("Gaussian has no mass in the fluid region")
    return density / mass


def uniform_density(
    grid: CartesianGrid, *, fluid_mask: np.ndarray | None = None
) -> np.ndarray:
    """Return a unit-mass uniform density over fluid cells."""

    mask = _validated_mask(fluid_mask, (grid.ny, grid.nx))
    density = mask.astype(np.float64)
    return density / (np.sum(density) * grid.cell_area_m2)


def _validated_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    values = np.asarray(mask)
    if values.shape != shape or values.dtype != bool:
        raise ValueError("fluid_mask must be boolean with grid shape")
    if not np.any(values):
        raise ValueError("fluid_mask must contain at least one fluid cell")
    return values.copy()
