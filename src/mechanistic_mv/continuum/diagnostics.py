"""Distribution diagnostics used by tests and validation experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..mechanics.geometry import CartesianGrid


@dataclass(frozen=True, slots=True)
class DensityMoments:
    mass: float
    mean_m: np.ndarray
    covariance_m2: np.ndarray


def density_moments(density_per_m2: np.ndarray, grid: CartesianGrid) -> DensityMoments:
    density = np.asarray(density_per_m2, dtype=np.float64)
    if density.shape != (grid.ny, grid.nx):
        raise ValueError("density must have grid shape")
    if np.any(density < 0.0) or not np.all(np.isfinite(density)):
        raise ValueError("density must be finite and non-negative")
    weights = density * grid.cell_area_m2
    mass = float(np.sum(weights))
    if mass <= 0.0:
        raise ValueError("density must have positive mass")
    x, y = grid.mesh_m()
    points = np.stack((x, y), axis=-1)
    mean = np.sum(points * weights[..., None], axis=(0, 1)) / mass
    displacement = points - mean
    flat_displacement = displacement.reshape(-1, 2)
    flat_weights = weights.ravel()
    covariance = np.einsum(
        "ni,nj,n->ij", flat_displacement, flat_displacement, flat_weights
    ) / mass
    return DensityMoments(mass=mass, mean_m=mean, covariance_m2=covariance)


def relative_l2_error(
    candidate_per_m2: np.ndarray,
    reference_per_m2: np.ndarray,
    grid: CartesianGrid,
) -> float:
    candidate = np.asarray(candidate_per_m2, dtype=np.float64)
    reference = np.asarray(reference_per_m2, dtype=np.float64)
    if candidate.shape != reference.shape or candidate.shape != (grid.ny, grid.nx):
        raise ValueError("densities must share the grid shape")
    numerator = np.sum((candidate - reference) ** 2) * grid.cell_area_m2
    denominator = np.sum(reference**2) * grid.cell_area_m2
    if denominator <= 0.0:
        raise ValueError("reference density must have positive L2 norm")
    return float(np.sqrt(numerator / denominator))


def jensen_shannon_divergence(
    first_per_m2: np.ndarray,
    second_per_m2: np.ndarray,
    grid: CartesianGrid,
) -> float:
    """Return the natural-log Jensen--Shannon divergence of cell masses."""

    first = np.asarray(first_per_m2, dtype=np.float64).ravel()
    second = np.asarray(second_per_m2, dtype=np.float64).ravel()
    if first.shape != second.shape or first.size != grid.nx * grid.ny:
        raise ValueError("densities must share the grid shape")
    if np.any(first < 0.0) or np.any(second < 0.0):
        raise ValueError("densities must be non-negative")
    p = first * grid.cell_area_m2
    q = second * grid.cell_area_m2
    p /= np.sum(p)
    q /= np.sum(q)
    midpoint = 0.5 * (p + q)
    p_positive = p > 0.0
    q_positive = q > 0.0
    first_kl = np.sum(p[p_positive] * np.log(p[p_positive] / midpoint[p_positive]))
    second_kl = np.sum(q[q_positive] * np.log(q[q_positive] / midpoint[q_positive]))
    return float(0.5 * (first_kl + second_kl))
