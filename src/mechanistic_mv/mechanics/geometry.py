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
) -> tuple[float, tuple[np.ndarray, ...]] | None:
    direction = end - start
    t_enter, t_exit = 0.0, 1.0
    near_faces: list[tuple[float, np.ndarray]] = []
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
        near_faces.append((near, normal))
        t_enter = max(t_enter, near)
        t_exit = min(t_exit, far)
        if t_enter > t_exit:
            return None
    if 0.0 <= t_enter <= 1.0 and t_enter <= t_exit:
        tolerance = 64.0 * np.finfo(np.float64).eps
        entry_normals = tuple(
            normal
            for near, normal in near_faces
            if abs(near - t_enter) <= tolerance
        )
        if entry_normals:
            return t_enter, entry_normals
    return None


def _first_outer_wall_hit(
    start: np.ndarray,
    end: np.ndarray,
    domain: RectangularDomain,
) -> tuple[float, tuple[np.ndarray, ...]] | None:
    """Return the first crossed outer face and its inward normal(s)."""

    direction = end - start
    candidates: list[tuple[float, np.ndarray]] = []
    for axis, limits in enumerate((domain.x_limits_m, domain.y_limits_m)):
        low, high = limits
        if direction[axis] < 0.0 and end[axis] < low:
            normal = np.zeros(2, dtype=np.float64)
            normal[axis] = 1.0
            candidates.append(((low - start[axis]) / direction[axis], normal))
        elif direction[axis] > 0.0 and end[axis] > high:
            normal = np.zeros(2, dtype=np.float64)
            normal[axis] = -1.0
            candidates.append(((high - start[axis]) / direction[axis], normal))
    if not candidates:
        return None
    first_fraction = min(fraction for fraction, _ in candidates)
    tolerance = 64.0 * np.finfo(np.float64).eps
    normals = tuple(
        normal
        for fraction, normal in candidates
        if abs(fraction - first_fraction) <= tolerance
    )
    return first_fraction, normals


def _inside_domain(points: np.ndarray, domain: RectangularDomain) -> np.ndarray:
    return (
        (points[:, 0] >= domain.x_limits_m[0])
        & (points[:, 0] <= domain.x_limits_m[1])
        & (points[:, 1] >= domain.y_limits_m[0])
        & (points[:, 1] <= domain.y_limits_m[1])
    )


def _unique_normals(normals: Iterable[np.ndarray]) -> tuple[np.ndarray, ...]:
    unique: list[np.ndarray] = []
    for normal in normals:
        if not any(np.array_equal(normal, existing) for existing in unique):
            unique.append(normal)
    return tuple(unique)


def enforce_particle_no_flux(
    previous: np.ndarray,
    proposed: np.ndarray,
    domain: RectangularDomain,
    obstacles: Iterable[RectangleObstacle] = (),
) -> tuple[np.ndarray, int]:
    """Apply specular no-flux reflection to a complete proposed displacement.

    The untravelled part of a displacement is reflected at each outer wall or
    solid rectangle face.  Thus oblique impacts preserve tangential motion and
    multiple impacts within one integration step are resolved in time order.
    The returned count is the number of reflected boundary faces.
    """

    old = _points(previous)
    new = _points(proposed)
    if old.shape != new.shape:
        raise ValueError("previous and proposed particle arrays must match")
    if not np.all(_inside_domain(old, domain)):
        raise ValueError("a particle starts outside the fluid domain")
    solid_geometry = tuple(obstacles)
    for obstacle in solid_geometry:
        if np.any(obstacle.contains(old)):
            raise ValueError("a particle starts inside a solid obstacle")
    if not solid_geometry and np.all(_inside_domain(new, domain)):
        return new.copy(), 0
    result = np.empty_like(new)
    collision_count = 0
    clearance = 16.0 * np.finfo(np.float64).eps * max(
        domain.x_limits_m[1] - domain.x_limits_m[0],
        domain.y_limits_m[1] - domain.y_limits_m[0],
    )
    for index in range(old.shape[0]):
        position = old[index].copy()
        remaining = new[index] - old[index]
        for _ in range(256):
            candidate = position + remaining
            hits: list[tuple[float, tuple[np.ndarray, ...]]] = []
            outer_hit = _first_outer_wall_hit(position, candidate, domain)
            if outer_hit is not None:
                hits.append(outer_hit)
            for obstacle in solid_geometry:
                obstacle_hit = _first_rectangle_hit(position, candidate, obstacle)
                if obstacle_hit is not None:
                    hits.append(obstacle_hit)
            if not hits:
                position = candidate
                break

            first_fraction = min(fraction for fraction, _ in hits)
            tolerance = 64.0 * np.finfo(np.float64).eps
            normals = _unique_normals(
                normal
                for fraction, hit_normals in hits
                if abs(fraction - first_fraction) <= tolerance
                for normal in hit_normals
            )
            contact = position + first_fraction * remaining
            remaining = (1.0 - first_fraction) * remaining
            for normal in normals:
                remaining -= 2.0 * np.dot(remaining, normal) * normal
            clearance_offset = clearance * np.sum(normals, axis=0)
            position = contact + clearance_offset
            remaining -= clearance_offset
            collision_count += len(normals)
            if np.linalg.norm(remaining, ord=np.inf) <= clearance:
                break
        else:
            raise RuntimeError(
                "particle displacement exceeded 256 no-flux reflections"
            )
        result[index] = position

    if not np.all(_inside_domain(result, domain)):
        raise RuntimeError("no-flux reflection left a particle outside the domain")
    for obstacle in solid_geometry:
        if np.any(obstacle.contains(result)):
            raise RuntimeError("no-flux reflection left a particle inside a solid")
    return result, collision_count
