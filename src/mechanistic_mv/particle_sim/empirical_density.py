"""Mass-conserving empirical particle density on a finite-volume grid."""

from __future__ import annotations

import numpy as np

from ..mechanics.density_scaling import DensityConvention, validate_density_convention
from ..mechanics.geometry import CartesianGrid


def empirical_density(
    positions_m: np.ndarray,
    grid: CartesianGrid,
    *,
    fluid_mask: np.ndarray | None = None,
    density_convention: DensityConvention = DensityConvention.PROBABILITY,
) -> np.ndarray:
    """Deposit counts as ``rho`` with integral 1 or ``n`` with integral N."""

    positions = np.asarray(positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2 or positions.shape[0] == 0:
        raise ValueError("positions_m must have shape (N, 2), N > 0")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions_m must be finite")
    convention = validate_density_convention(density_convention)
    counts, _, _ = np.histogram2d(
        positions[:, 1],
        positions[:, 0],
        bins=(grid.ny, grid.nx),
        range=(grid.domain.y_limits_m, grid.domain.x_limits_m),
    )
    if int(np.sum(counts)) != positions.shape[0]:
        raise ValueError("one or more particles lie outside the grid domain")
    normalization = (
        positions.shape[0]
        if convention is DensityConvention.PROBABILITY
        else 1
    )
    density = counts / (normalization * grid.cell_area_m2)
    if fluid_mask is not None:
        mask = np.asarray(fluid_mask)
        if mask.shape != density.shape or mask.dtype != bool:
            raise ValueError("fluid_mask must be boolean with grid shape")
        if np.any(density[~mask] != 0.0):
            raise ValueError("particles were deposited inside a solid cell")
    return density
