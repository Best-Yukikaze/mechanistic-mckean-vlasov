"""Discrete weak-form checks for finite-volume conservation."""

from __future__ import annotations

import numpy as np

from ..mechanics.geometry import CartesianGrid
from .flux import FaceFluxes


def weak_form_residual(
    old_density_per_m2: np.ndarray,
    new_density_per_m2: np.ndarray,
    test_function: np.ndarray,
    fluxes: FaceFluxes,
    grid: CartesianGrid,
    dt_s: float,
) -> float:
    """Return the finite-volume summation-by-parts residual in 1/s."""

    old = np.asarray(old_density_per_m2, dtype=np.float64)
    new = np.asarray(new_density_per_m2, dtype=np.float64)
    eta = np.asarray(test_function, dtype=np.float64)
    shape = (grid.ny, grid.nx)
    if old.shape != shape or new.shape != shape or eta.shape != shape:
        raise ValueError("density and test function arrays must have grid shape")
    dt = float(dt_s)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    temporal = np.sum(eta * (new - old)) * grid.cell_area_m2 / dt
    x_term = np.sum(
        (eta[:, 1:] - eta[:, :-1]) * fluxes.x_per_m_s[:, 1:grid.nx]
    ) * grid.dy_m
    y_term = np.sum(
        (eta[1:, :] - eta[:-1, :]) * fluxes.y_per_m_s[1:grid.ny, :]
    ) * grid.dx_m
    return float(temporal - x_term - y_term)
