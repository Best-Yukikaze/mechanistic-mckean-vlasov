"""Conservative face fluxes with impermeable outer and solid boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..mechanics.geometry import CartesianGrid


class DriftFluxScheme(str, Enum):
    """Selectable spatial fluxes for ``J=-D grad(rho)-M rho grad(Phi)``."""

    FIRST_ORDER_UPWIND = "FIRST_ORDER_UPWIND"
    SECOND_ORDER_SCHARFETTER_GUMMEL = "SECOND_ORDER_SCHARFETTER_GUMMEL"


@dataclass(frozen=True, slots=True)
class FaceMasks:
    open_x: np.ndarray
    open_y: np.ndarray


@dataclass(frozen=True, slots=True)
class FaceFluxes:
    """One shared flux per face and the explicit positivity row sum.

    ``outgoing_rate_per_s[i, j]`` is the sum of non-negative coefficients
    multiplying that cell's density in the donor/SG forward-Euler operator.
    Thus ``dt <= 1 / max(outgoing_rate_per_s)`` is sufficient for positivity
    before the solver applies its CFL safety factor.
    """

    x_per_m_s: np.ndarray
    y_per_m_s: np.ndarray
    max_abs_velocity_x_m_per_s: float
    max_abs_velocity_y_m_per_s: float
    # ``None`` preserves the four-argument constructor used before row-sum
    # CFL metadata existed.  The solver then takes its explicit legacy-safe
    # bound; fluxes produced by ``compute_face_fluxes`` always provide an array.
    outgoing_rate_per_s: np.ndarray | None = None
    drift_flux_scheme: DriftFluxScheme = DriftFluxScheme.FIRST_ORDER_UPWIND

    @property
    def maximum_outgoing_rate_per_s(self) -> float | None:
        if self.outgoing_rate_per_s is None:
            return None
        return float(np.max(self.outgoing_rate_per_s))


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
    drift_flux_scheme: DriftFluxScheme = DriftFluxScheme.FIRST_ORDER_UPWIND,
) -> FaceFluxes:
    """Return a shared-face flux and its exact monotone-row CFL coefficients.

    ``FIRST_ORDER_UPWIND`` is the donor-cell reference.  The second option is
    the Scharfetter--Gummel (Chang--Cooper) flux

    ``J = D/h [B(q) rho_L - B(-q) rho_R]``,
    ``q = (M/D) (Phi_R-Phi_L)``, ``B(q)=q/expm1(q)``.

    ``B`` is evaluated stably near zero and at large signed ``q``.  For
    positive ``D`` its two density coefficients are non-negative, so the
    returned per-cell row sum gives an explicit positivity CFL.  At the
    discrete Gibbs state ``rho_R=rho_L exp(-q)``, every open SG face has zero
    flux up to floating-point rounding.
    """

    density = np.asarray(density_per_m2, dtype=np.float64)
    potential = np.asarray(effective_potential_joule, dtype=np.float64)
    shape = (grid.ny, grid.nx)
    diffusion = float(diffusion_m2_per_s)
    mobility = float(mobility_m_per_newton_second)
    if density.shape != shape or potential.shape != shape:
        raise ValueError("density and effective potential must match the grid")
    if np.any(density < 0.0) or not np.all(np.isfinite(density)):
        raise ValueError("density must be finite and non-negative")
    if not np.all(np.isfinite(potential)):
        raise ValueError("effective potential must be finite")
    if not np.isfinite(diffusion) or diffusion < 0.0:
        raise ValueError("diffusion_m2_per_s must be finite and non-negative")
    if not np.isfinite(mobility) or mobility < 0.0:
        raise ValueError("mobility_m_per_newton_second must be finite and non-negative")
    if not isinstance(drift_flux_scheme, DriftFluxScheme):
        raise TypeError("drift_flux_scheme must be DriftFluxScheme")
    _validate_face_masks(face_masks, grid)

    delta_x = potential[:, 1:] - potential[:, :-1]
    delta_y = potential[1:, :] - potential[:-1, :]
    velocity_x = -mobility * delta_x / grid.dx_m
    velocity_y = -mobility * delta_y / grid.dy_m
    open_x = face_masks.open_x[:, 1:grid.nx]
    open_y = face_masks.open_y[1:grid.ny, :]

    if drift_flux_scheme is DriftFluxScheme.FIRST_ORDER_UPWIND:
        interior_x, left_x, right_x = _first_order_x(
            density, velocity_x, diffusion, grid.dx_m
        )
        interior_y, lower_y, upper_y = _first_order_y(
            density, velocity_y, diffusion, grid.dy_m
        )
    else:
        if diffusion <= 0.0:
            raise ValueError(
                "SECOND_ORDER_SCHARFETTER_GUMMEL requires diffusion_m2_per_s > 0"
            )
        interior_x, left_x, right_x = _scharfetter_gummel_x(
            density, delta_x, diffusion, mobility, grid.dx_m
        )
        interior_y, lower_y, upper_y = _scharfetter_gummel_y(
            density, delta_y, diffusion, mobility, grid.dy_m
        )

    flux_x = np.zeros((grid.ny, grid.nx + 1), dtype=np.float64)
    flux_y = np.zeros((grid.ny + 1, grid.nx), dtype=np.float64)
    flux_x[:, 1:grid.nx] = np.where(open_x, interior_x, 0.0)
    flux_y[1:grid.ny, :] = np.where(open_y, interior_y, 0.0)
    outgoing_rate = _outgoing_rate(
        grid,
        open_x,
        open_y,
        left_x,
        right_x,
        lower_y,
        upper_y,
    )
    open_vx = velocity_x[open_x]
    open_vy = velocity_y[open_y]
    return FaceFluxes(
        x_per_m_s=flux_x,
        y_per_m_s=flux_y,
        max_abs_velocity_x_m_per_s=(
            float(np.max(np.abs(open_vx))) if open_vx.size else 0.0
        ),
        max_abs_velocity_y_m_per_s=(
            float(np.max(np.abs(open_vy))) if open_vy.size else 0.0
        ),
        outgoing_rate_per_s=outgoing_rate,
        drift_flux_scheme=drift_flux_scheme,
    )


def _first_order_x(
    density: np.ndarray,
    velocity: np.ndarray,
    diffusion: float,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_density = density[:, :-1]
    right_density = density[:, 1:]
    flux = -diffusion * (right_density - left_density) / spacing
    flux += velocity * np.where(velocity >= 0.0, left_density, right_density)
    left_rate = diffusion / spacing**2 + np.maximum(velocity, 0.0) / spacing
    right_rate = diffusion / spacing**2 + np.maximum(-velocity, 0.0) / spacing
    return flux, left_rate, right_rate


def _first_order_y(
    density: np.ndarray,
    velocity: np.ndarray,
    diffusion: float,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower_density = density[:-1, :]
    upper_density = density[1:, :]
    flux = -diffusion * (upper_density - lower_density) / spacing
    flux += velocity * np.where(velocity >= 0.0, lower_density, upper_density)
    lower_rate = diffusion / spacing**2 + np.maximum(velocity, 0.0) / spacing
    upper_rate = diffusion / spacing**2 + np.maximum(-velocity, 0.0) / spacing
    return flux, lower_rate, upper_rate


def _scharfetter_gummel_x(
    density: np.ndarray,
    potential_jump: np.ndarray,
    diffusion: float,
    mobility: float,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = _dimensionless_potential_jump(potential_jump, mobility, diffusion)
    left_coefficient = diffusion / spacing * _bernoulli(q)
    right_coefficient = diffusion / spacing * _bernoulli(-q)
    flux = left_coefficient * density[:, :-1] - right_coefficient * density[:, 1:]
    return flux, left_coefficient / spacing, right_coefficient / spacing


def _scharfetter_gummel_y(
    density: np.ndarray,
    potential_jump: np.ndarray,
    diffusion: float,
    mobility: float,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = _dimensionless_potential_jump(potential_jump, mobility, diffusion)
    lower_coefficient = diffusion / spacing * _bernoulli(q)
    upper_coefficient = diffusion / spacing * _bernoulli(-q)
    flux = lower_coefficient * density[:-1, :] - upper_coefficient * density[1:, :]
    return flux, lower_coefficient / spacing, upper_coefficient / spacing


def _dimensionless_potential_jump(
    potential_jump_joule: np.ndarray, mobility: float, diffusion: float
) -> np.ndarray:
    q = (mobility / diffusion) * potential_jump_joule
    if not np.all(np.isfinite(q)):
        raise FloatingPointError("non-finite Scharfetter-Gummel potential jump")
    return q


def _bernoulli(q: np.ndarray) -> np.ndarray:
    """Evaluate ``q / expm1(q)`` without cancellation or exponential overflow."""

    values = np.asarray(q, dtype=np.float64)
    result = np.empty_like(values)
    small = np.abs(values) < 1.0e-5
    positive_large = values > 50.0
    negative_large = values < -50.0
    regular = ~(small | positive_large | negative_large)
    q_small = values[small]
    result[small] = 1.0 - 0.5 * q_small + q_small**2 / 12.0 - q_small**4 / 720.0
    q_positive = values[positive_large]
    result[positive_large] = (
        q_positive * np.exp(-q_positive) / -np.expm1(-q_positive)
    )
    q_negative = values[negative_large]
    result[negative_large] = -q_negative / -np.expm1(q_negative)
    result[regular] = values[regular] / np.expm1(values[regular])
    # For extremely uphill jumps the exact positive value may underflow to
    # zero in float64; zero remains a valid non-negative outflow coefficient.
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise FloatingPointError("invalid Scharfetter-Gummel Bernoulli coefficient")
    return result


def _outgoing_rate(
    grid: CartesianGrid,
    open_x: np.ndarray,
    open_y: np.ndarray,
    left_x: np.ndarray,
    right_x: np.ndarray,
    lower_y: np.ndarray,
    upper_y: np.ndarray,
) -> np.ndarray:
    """Return the exact donor/outgoing row sum for the open-face operator."""

    rate = np.zeros((grid.ny, grid.nx), dtype=np.float64)
    rate[:, :-1] += np.where(open_x, left_x, 0.0)
    rate[:, 1:] += np.where(open_x, right_x, 0.0)
    rate[:-1, :] += np.where(open_y, lower_y, 0.0)
    rate[1:, :] += np.where(open_y, upper_y, 0.0)
    if not np.all(np.isfinite(rate)) or np.any(rate < 0.0):
        raise FloatingPointError("invalid explicit outgoing-rate row sum")
    return rate


def _validate_face_masks(face_masks: FaceMasks, grid: CartesianGrid) -> None:
    if (
        face_masks.open_x.shape != (grid.ny, grid.nx + 1)
        or face_masks.open_y.shape != (grid.ny + 1, grid.nx)
        or face_masks.open_x.dtype != bool
        or face_masks.open_y.dtype != bool
    ):
        raise ValueError("face masks must be boolean arrays matching the grid")
