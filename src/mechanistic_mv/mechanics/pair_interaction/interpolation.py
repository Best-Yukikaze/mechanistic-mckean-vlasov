"""Shape-preserving force interpolation and exact PCHIP antiderivative."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import PchipInterpolator

from .data_model import PairForceTable


@dataclass(frozen=True, slots=True)
class PchipIntegratedForceLaw:
    """Represent ``F(r)`` and ``W(r)=-integral[r_ref->r] F(s) ds``.

    Interpolation below the smallest supplied distance is forbidden.  At and
    above ``r_ref`` both force and potential are zero because the data contract
    explicitly validates that the reference force is negligible.
    """

    force_data: PairForceTable
    _force: PchipInterpolator = field(init=False, repr=False, compare=False)
    _antiderivative: PchipInterpolator = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        force = PchipInterpolator(
            self.force_data.center_distance_m,
            self.force_data.radial_force_newton,
            extrapolate=False,
        )
        object.__setattr__(self, "_force", force)
        object.__setattr__(self, "_antiderivative", force.antiderivative())

    @property
    def minimum_distance_m(self) -> float:
        return float(self.force_data.center_distance_m[0])

    @property
    def reference_distance_m(self) -> float:
        return self.force_data.metadata.reference_distance_m

    def radial_force_newton(self, radius_m: np.ndarray | float) -> np.ndarray:
        radius, flat = self._validated_radius(radius_m)
        values = np.zeros_like(flat)
        inside = flat < self.reference_distance_m
        values[inside] = self._force(flat[inside])
        return values.reshape(radius.shape)

    def potential_joule(self, radius_m: np.ndarray | float) -> np.ndarray:
        radius, flat = self._validated_radius(radius_m)
        values = np.zeros_like(flat)
        inside = flat < self.reference_distance_m
        reference_value = float(self._antiderivative(self.reference_distance_m))
        values[inside] = reference_value - self._antiderivative(flat[inside])
        return values.reshape(radius.shape)

    def potential_at_nodes_by_trapezoid_joule(self) -> np.ndarray:
        """Direct discrete-integration baseline evaluated at source nodes."""

        distance = self.force_data.center_distance_m
        force = self.force_data.radial_force_newton
        segment_integral = 0.5 * (force[:-1] + force[1:]) * np.diff(distance)
        values = np.zeros_like(distance)
        values[:-1] = np.cumsum(segment_integral[::-1])[::-1]
        return values

    def _validated_radius(
        self, radius_m: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        radius = np.asarray(radius_m, dtype=np.float64)
        if not np.all(np.isfinite(radius)) or np.any(radius < 0.0):
            raise ValueError("radii must be finite and non-negative")
        flat = radius.reshape(-1)
        if np.any(flat < self.minimum_distance_m):
            raise ValueError(
                "radius lies below the validated force table; short-range "
                "extrapolation is not physically specified"
            )
        return radius, flat
