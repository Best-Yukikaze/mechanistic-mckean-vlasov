"""Distribution diagnostics used by tests and validation experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..mechanics.geometry import CartesianGrid


_MASS_COMPARISON_RELATIVE_TOLERANCE = 1.0e-10


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
    """Return the cell-area-weighted relative L2 error of equal-mass densities."""

    candidate, reference = _validated_comparable_density_pair(
        candidate_per_m2,
        reference_per_m2,
        grid,
        first_name="candidate",
        second_name="reference",
    )
    with np.errstate(over="ignore", invalid="ignore"):
        numerator = np.sum((candidate - reference) ** 2) * grid.cell_area_m2
        denominator = np.sum(reference**2) * grid.cell_area_m2
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        raise ValueError("relative L2 calculation overflowed for finite densities")
    if denominator <= 0.0:
        raise ValueError("reference density must have positive L2 norm")
    return float(np.sqrt(numerator / denominator))


def jensen_shannon_divergence(
    first_per_m2: np.ndarray,
    second_per_m2: np.ndarray,
    grid: CartesianGrid,
) -> float:
    """Return the natural-log Jensen--Shannon divergence of cell masses."""

    first, second = _validated_comparable_density_pair(
        first_per_m2,
        second_per_m2,
        grid,
        first_name="first",
        second_name="second",
    )
    first = first.ravel()
    second = second.ravel()
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


def _validated_comparable_density_pair(
    first_per_m2: np.ndarray,
    second_per_m2: np.ndarray,
    grid: CartesianGrid,
    *,
    first_name: str,
    second_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate two physical densities before comparing their shapes.

    These diagnostics compare densities representing the same population.  They
    accept either unit-mass probability densities or equal-mass number
    densities, but never silently rescale one density to hide a mass mismatch.
    The relative tolerance is explicit and scale-relative, so it permits
    floating-point summation roundoff without silently accepting a material
    population mismatch.
    """

    first = _validated_density(first_per_m2, grid, first_name)
    second = _validated_density(second_per_m2, grid, second_name)
    first_mass = _discrete_mass(first, grid, first_name)
    second_mass = _discrete_mass(second, grid, second_name)
    allowed_difference = _MASS_COMPARISON_RELATIVE_TOLERANCE * max(
        first_mass, second_mass
    )
    if abs(first_mass - second_mass) > allowed_difference:
        raise ValueError(
            "density discrete masses must agree within relative tolerance "
            f"{_MASS_COMPARISON_RELATIVE_TOLERANCE:.1e}; "
            f"got {first_name}={first_mass:.16e}, "
            f"{second_name}={second_mass:.16e}"
        )
    return first, second


def _validated_density(
    density_per_m2: np.ndarray,
    grid: CartesianGrid,
    name: str,
) -> np.ndarray:
    density = np.asarray(density_per_m2, dtype=np.float64)
    expected_shape = (grid.ny, grid.nx)
    if density.shape != expected_shape:
        raise ValueError(
            f"{name} density must have grid shape {expected_shape}; "
            f"got {density.shape}"
        )
    if not np.all(np.isfinite(density)):
        raise ValueError(f"{name} density must contain only finite values")
    if np.any(density < 0.0):
        raise ValueError(f"{name} density must be non-negative")
    return density


def _discrete_mass(
    density_per_m2: np.ndarray,
    grid: CartesianGrid,
    name: str,
) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        mass = float(np.sum(density_per_m2) * grid.cell_area_m2)
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError(f"{name} density must have positive finite discrete mass")
    return mass
