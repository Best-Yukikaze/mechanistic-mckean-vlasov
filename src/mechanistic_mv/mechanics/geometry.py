"""Rectangular fluid geometry and no-penetration particle operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class RectangularDomain:
    """Two-dimensional SI domain with axis-aligned limits in metres."""

    x_limits_m: tuple[float, float] = (0.0, 20.0e-6)
    y_limits_m: tuple[float, float] = (0.0, 20.0e-6)

    def __post_init__(self) -> None:
        for name, limits in (
            ("x_limits_m", self.x_limits_m),
            ("y_limits_m", self.y_limits_m),
        ):
            values = np.asarray(limits, dtype=np.float64)
            if (
                values.shape != (2,)
                or not np.all(np.isfinite(values))
                or values[0] >= values[1]
            ):
                raise ValueError(f"{name} must be a finite increasing pair")


@dataclass(frozen=True, slots=True)
class RectangleObstacle:
    """Closed solid rectangle, expressed in metres."""

    x_limits_m: tuple[float, float]
    y_limits_m: tuple[float, float]

    def __post_init__(self) -> None:
        RectangularDomain(self.x_limits_m, self.y_limits_m)

    def contains(self, points: np.ndarray) -> np.ndarray:
        values = _points(points)
        return (
            (values[:, 0] >= self.x_limits_m[0])
            & (values[:, 0] <= self.x_limits_m[1])
            & (values[:, 1] >= self.y_limits_m[0])
            & (values[:, 1] <= self.y_limits_m[1])
        )


@dataclass(frozen=True, slots=True)
class CartesianGrid:
    """Cell-centred finite-volume grid on a rectangular SI domain."""

    domain: RectangularDomain = RectangularDomain()
    nx: int = 48
    ny: int = 48

    def __post_init__(self) -> None:
        if isinstance(self.nx, bool) or isinstance(self.ny, bool):
            raise ValueError("grid sizes must be integers")
        if not isinstance(self.nx, int) or not isinstance(self.ny, int):
            raise ValueError("grid sizes must be integers")
        if self.nx < 4 or self.ny < 4:
            raise ValueError("grid sizes must be at least four")

    @property
    def dx_m(self) -> float:
        return (self.domain.x_limits_m[1] - self.domain.x_limits_m[0]) / self.nx

    @property
    def dy_m(self) -> float:
        return (self.domain.y_limits_m[1] - self.domain.y_limits_m[0]) / self.ny

    @property
    def cell_area_m2(self) -> float:
        return self.dx_m * self.dy_m

    @property
    def x_centres_m(self) -> np.ndarray:
        low, high = self.domain.x_limits_m
        return np.linspace(low + 0.5 * self.dx_m, high - 0.5 * self.dx_m, self.nx)

    @property
    def y_centres_m(self) -> np.ndarray:
        low, high = self.domain.y_limits_m
        return np.linspace(low + 0.5 * self.dy_m, high - 0.5 * self.dy_m, self.ny)

    def mesh_m(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.x_centres_m, self.y_centres_m, indexing="xy")

    def fluid_mask(self, obstacles: Iterable[RectangleObstacle] = ()) -> np.ndarray:
        x, y = self.mesh_m()
        fluid = np.ones((self.ny, self.nx), dtype=bool)
        for obstacle in obstacles:
            fluid &= ~(
                (x >= obstacle.x_limits_m[0])
                & (x <= obstacle.x_limits_m[1])
                & (y >= obstacle.y_limits_m[0])
                & (y <= obstacle.y_limits_m[1])
            )
        if not np.any(fluid):
            raise ValueError("obstacles leave no fluid cells")
        return fluid


def _points(values: np.ndarray) -> np.ndarray:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must be finite")
    return points


def reflect_outer_walls(points: np.ndarray, domain: RectangularDomain) -> np.ndarray:
    """Reflect arbitrary overshoots at impermeable outer walls."""

    reflected = _points(points).copy()
    for axis, limits in enumerate((domain.x_limits_m, domain.y_limits_m)):
        lower, upper = limits
        span = upper - lower
        folded = np.mod(reflected[:, axis] - lower, 2.0 * span)
        reflected[:, axis] = np.where(
            folded <= span, lower + folded, upper - (folded - span)
        )
    return reflected


def _first_rectangle_hit(
    start: np.ndarray,
    end: np.ndarray,
    obstacle: RectangleObstacle,
) -> tuple[float, np.ndarray] | None:
    direction = end - start
    t_enter, t_exit = 0.0, 1.0
    entry_normal = np.zeros(2, dtype=np.float64)
    for axis, limits in enumerate((obstacle.x_limits_m, obstacle.y_limits_m)):
        low, high = limits
        if abs(direction[axis]) <= np.finfo(np.float64).tiny:
            if start[axis] < low or start[axis] > high:
                return None
            continue
        near = (low - start[axis]) / direction[axis]
        far = (high - start[axis]) / direction[axis]
        normal = np.zeros(2, dtype=np.float64)
        normal[axis] = -1.0 if direction[axis] > 0.0 else 1.0
        if near > far:
            near, far = far, near
        if near > t_enter:
            t_enter = near
            entry_normal = normal
        t_exit = min(t_exit, far)
        if t_enter > t_exit:
            return None
    if 0.0 <= t_enter <= 1.0 and t_enter <= t_exit:
        return t_enter, entry_normal
    return None


def enforce_particle_no_flux(
    previous: np.ndarray,
    proposed: np.ndarray,
    domain: RectangularDomain,
    obstacles: Iterable[RectangleObstacle] = (),
) -> tuple[np.ndarray, int]:
    """Reflect outer walls and block swept paths through solid rectangles.

    Collision response stops at first contact. This is robust and conservative
    for validation, while higher-fidelity tangential/reflected contact remains
    a documented future backend.
    """

    old = _points(previous)
    new = _points(proposed)
    if old.shape != new.shape:
        raise ValueError("previous and proposed particle arrays must match")
    result = reflect_outer_walls(new, domain)
    collision_count = 0
    clearance = 16.0 * np.finfo(np.float64).eps * max(
        domain.x_limits_m[1] - domain.x_limits_m[0],
        domain.y_limits_m[1] - domain.y_limits_m[0],
    )
    for index in range(old.shape[0]):
        candidate = result[index]
        for obstacle in obstacles:
            hit = _first_rectangle_hit(old[index], candidate, obstacle)
            if hit is None and not obstacle.contains(candidate[None, :])[0]:
                continue
            collision_count += 1
            if hit is None:
                distances = np.asarray(
                    [
                        abs(candidate[0] - obstacle.x_limits_m[0]),
                        abs(candidate[0] - obstacle.x_limits_m[1]),
                        abs(candidate[1] - obstacle.y_limits_m[0]),
                        abs(candidate[1] - obstacle.y_limits_m[1]),
                    ]
                )
                side = int(np.argmin(distances))
                normals = np.asarray([[-1, 0], [1, 0], [0, -1], [0, 1]])
                normal = normals[side].astype(np.float64)
                contact = candidate.copy()
                contact[side // 2] = (
                    obstacle.x_limits_m[side]
                    if side < 2
                    else obstacle.y_limits_m[side - 2]
                )
            else:
                fraction, normal = hit
                contact = old[index] + fraction * (candidate - old[index])
            candidate = contact + clearance * normal
        result[index] = candidate
    return reflect_outer_walls(result, domain), collision_count

