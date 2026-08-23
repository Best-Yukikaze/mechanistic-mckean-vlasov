"""Control interfaces that map an input to a conservative potential."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ControlledPotentialBackend(Protocol):
    """Contract for the preferred control path ``u -> V(x; u)``."""

    name: str
    physical_status: str

    def potential_joule(
        self, positions_m: np.ndarray, control: np.ndarray | None
    ) -> np.ndarray: ...

    def force_newton(
        self, positions_m: np.ndarray, control: np.ndarray | None
    ) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class ZeroControlledPotential:
    name: str = "zero_controlled_potential"
    physical_status: str = "physical null control input"

    def potential_joule(
        self, positions_m: np.ndarray, control: np.ndarray | None = None
    ) -> np.ndarray:
        del control
        points = _points(positions_m)
        return np.zeros(points.shape[:-1], dtype=np.float64)

    def force_newton(
        self, positions_m: np.ndarray, control: np.ndarray | None = None
    ) -> np.ndarray:
        del control
        return np.zeros_like(_points(positions_m))


@dataclass(frozen=True, slots=True)
class TestOnlyUniformFieldPotential:
    """TEST_ONLY_NOT_FINAL_PHYSICS linear potential for direction checks.

    ``V(x;u)=-F_max u dot (x-x_ref)`` and therefore
    ``-grad(V)=F_max u``. No claim is made that this is a calibrated actuator.
    """

    maximum_force_newton: float
    reference_position_m: tuple[float, float] = (0.0, 0.0)
    name: str = "test_only_not_final_physics_uniform_field_potential"
    physical_status: str = "TEST_ONLY_NOT_FINAL_PHYSICS"

    def __post_init__(self) -> None:
        reference = np.asarray(self.reference_position_m, dtype=np.float64)
        if reference.shape != (2,) or not np.all(np.isfinite(reference)):
            raise ValueError("reference_position_m must be a finite two-vector")
        if not np.isfinite(self.maximum_force_newton) or self.maximum_force_newton <= 0:
            raise ValueError("maximum_force_newton must be finite and positive")

    def potential_joule(
        self, positions_m: np.ndarray, control: np.ndarray | None
    ) -> np.ndarray:
        points = _points(positions_m)
        force = self._force_vector(control)
        displacement = points - np.asarray(self.reference_position_m)
        return -np.sum(displacement * force, axis=-1)

    def force_newton(
        self, positions_m: np.ndarray, control: np.ndarray | None
    ) -> np.ndarray:
        points = _points(positions_m)
        return np.broadcast_to(self._force_vector(control), points.shape).copy()

    def _force_vector(self, control: np.ndarray | None) -> np.ndarray:
        if control is None:
            vector = np.zeros(2, dtype=np.float64)
        else:
            vector = np.asarray(control, dtype=np.float64)
            if vector.shape != (2,) or not np.all(np.isfinite(vector)):
                raise ValueError("control must be a finite two-vector")
            if np.linalg.norm(vector) > 1.0 + 1.0e-12:
                raise ValueError("test control norm cannot exceed one")
        return self.maximum_force_newton * vector


def _points(values: np.ndarray) -> np.ndarray:
    points = np.asarray(values, dtype=np.float64)
    if points.shape[-1:] != (2,) or not np.all(np.isfinite(points)):
        raise ValueError("positions must be finite and end in two components")
    return points
