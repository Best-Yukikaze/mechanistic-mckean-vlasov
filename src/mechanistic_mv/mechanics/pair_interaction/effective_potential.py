"""Validation-gated Hydrogel effective pair potential for particle and MV use."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..hydrogel.equilibrium import TimeScaleStatus
from ..hydrogel.parameters import TEST_ONLY_NOT_CALIBRATED
from .data_model import (
    PairDataValidationStatus,
    PairForceScaling,
    PairForceTable,
)
from .interpolation import PchipIntegratedForceLaw


@dataclass(frozen=True, slots=True)
class ForcePotentialValidationReport:
    passed: bool
    evaluated_point_count: int
    maximum_absolute_error_newton: float
    maximum_relative_error: float
    pchip_vs_trapezoid_maximum_difference_joule: float


def validate_force_potential_consistency(
    law: PchipIntegratedForceLaw,
    *,
    absolute_tolerance_newton: float,
    relative_tolerance: float,
    finite_difference_step_fraction: float,
) -> ForcePotentialValidationReport:
    """Numerically verify ``F=-dW/dr`` away from interpolation endpoints."""

    tolerances = np.asarray(
        [absolute_tolerance_newton, relative_tolerance, finite_difference_step_fraction],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(tolerances)) or np.any(tolerances < 0.0):
        raise ValueError("validation tolerances must be finite and non-negative")
    if not 0.0 < finite_difference_step_fraction < 0.25:
        raise ValueError("finite_difference_step_fraction must lie in (0, 0.25)")

    nodes = law.force_data.center_distance_m
    widths = np.diff(nodes)
    points = 0.5 * (nodes[:-1] + nodes[1:])
    steps = finite_difference_step_fraction * widths
    force = law.radial_force_newton(points)
    force_from_potential = -(
        law.potential_joule(points + steps) - law.potential_joule(points - steps)
    ) / (2.0 * steps)
    absolute_error = np.abs(force_from_potential - force)
    scale = np.maximum(
        np.maximum(np.abs(force), np.abs(force_from_potential)),
        np.finfo(np.float64).tiny,
    )
    relative_error = absolute_error / scale
    allowed = absolute_tolerance_newton + relative_tolerance * np.abs(force)
    pchip_nodes = law.potential_joule(nodes)
    trapezoid_nodes = law.potential_at_nodes_by_trapezoid_joule()
    return ForcePotentialValidationReport(
        passed=bool(np.all(absolute_error <= allowed)),
        evaluated_point_count=int(points.size),
        maximum_absolute_error_newton=float(np.max(absolute_error)),
        maximum_relative_error=float(np.max(relative_error)),
        pchip_vs_trapezoid_maximum_difference_joule=float(
            np.max(np.abs(pchip_nodes - trapezoid_nodes))
        ),
    )


@dataclass(frozen=True, slots=True)
class HydrogelEffectivePairPotential:
    """Use one validated radial law in particles and continuum convolution.

    The current particle model uses ``(1/N) sum F`` and the continuum density
    integrates to one.  Consequently this backend rejects unscaled single-pair
    data instead of silently changing the physical interaction strength.
    """

    force_data: PairForceTable
    derivative_absolute_tolerance_newton: float
    derivative_relative_tolerance: float
    finite_difference_step_fraction: float
    name: str = field(init=False)
    physical_status: str = field(init=False)
    scaling_semantics: str = field(init=False)
    validation_report: ForcePotentialValidationReport = field(init=False)
    _law: PchipIntegratedForceLaw = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        metadata = self.force_data.metadata
        if metadata.validation_status is not PairDataValidationStatus.PASSED:
            raise ValueError("pair-force data must have validation_status=PASSED")
        if (
            metadata.physical_status != TEST_ONLY_NOT_CALIBRATED
            and metadata.time_scale_status is not TimeScaleStatus.SATISFIED
        ):
            raise ValueError(
                "a physical Hydrogel pair reduction requires a verified "
                "tau_gel << tau_swarm assessment"
            )
        if self.force_data.metadata.scaling is not PairForceScaling.KAC_NORMALIZED_PROBABILITY:
            raise ValueError(
                "current MV rho has unit mass and particle forces use 1/N; "
                "an unscaled physical single-pair table cannot be used directly"
            )
        if self.force_data.center_distance_m[0] != 0.0:
            raise ValueError(
                "the continuum convolution evaluates W at zero displacement; "
                "the force table must cover r=0 because no short-range closure "
                "or extrapolation was supplied"
            )
        law = PchipIntegratedForceLaw(self.force_data)
        report = validate_force_potential_consistency(
            law,
            absolute_tolerance_newton=self.derivative_absolute_tolerance_newton,
            relative_tolerance=self.derivative_relative_tolerance,
            finite_difference_step_fraction=self.finite_difference_step_fraction,
        )
        if not report.passed:
            raise ValueError(
                "pair force/potential derivative validation failed: "
                f"max_abs={report.maximum_absolute_error_newton:.6e} N, "
                f"max_rel={report.maximum_relative_error:.6e}"
            )
        object.__setattr__(self, "_law", law)
        object.__setattr__(self, "validation_report", report)
        object.__setattr__(
            self, "name", f"hydrogel_effective_pair:{self.force_data.metadata.dataset_id}"
        )
        object.__setattr__(self, "physical_status", self.force_data.metadata.physical_status)
        object.__setattr__(
            self, "scaling_semantics", self.force_data.metadata.scaling.value
        )

    def radial_potential_joule(self, radius_m: np.ndarray | float) -> np.ndarray:
        return self._law.potential_joule(radius_m)

    def radial_force_newton(self, radius_m: np.ndarray | float) -> np.ndarray:
        return self._law.radial_force_newton(radius_m)

    def radial_derivative_joule_per_m(
        self, radius_m: np.ndarray | float
    ) -> np.ndarray:
        return -self.radial_force_newton(radius_m)

    def potential_joule(self, displacement_m: np.ndarray) -> np.ndarray:
        displacement = _validated_displacement(displacement_m)
        return self.radial_potential_joule(np.linalg.norm(displacement, axis=-1))

    def force_newton(self, displacement_m: np.ndarray) -> np.ndarray:
        displacement = _validated_displacement(displacement_m)
        radius = np.linalg.norm(displacement, axis=-1)
        flat_radius = radius.reshape(-1)
        flat_radial_force = np.zeros_like(flat_radius)
        nonzero = flat_radius > 0.0
        if np.any(nonzero):
            flat_radial_force[nonzero] = self.radial_force_newton(
                flat_radius[nonzero]
            )
        flat_factor = np.zeros_like(flat_radius)
        np.divide(
            flat_radial_force,
            flat_radius,
            out=flat_factor,
            where=nonzero,
        )
        factor = flat_factor.reshape(radius.shape)
        return factor[..., None] * displacement


def _validated_displacement(values: np.ndarray) -> np.ndarray:
    displacement = np.asarray(values, dtype=np.float64)
    if displacement.shape[-1:] != (2,):
        raise ValueError("displacements must end with two Cartesian components")
    if not np.all(np.isfinite(displacement)):
        raise ValueError("displacements must be finite")
    return displacement
