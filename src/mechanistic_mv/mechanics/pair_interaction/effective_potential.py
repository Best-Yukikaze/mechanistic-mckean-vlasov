"""Validation-gated Hydrogel effective pair potential for particle and MV use."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..density_scaling import (
    DensityConvention,
    PairForceScaling,
    expected_pair_force_scaling,
    validate_density_convention,
)
from ..hydrogel.equilibrium import TimeScaleStatus
from ..hydrogel.parameters import TEST_ONLY_NOT_CALIBRATED
from .data_model import (
    PairDataValidationStatus,
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
    """Use one validated radial law with an explicit density convention.

    The default remains probability density with Kac-scaled force data. Number
    density explicitly accepts only unscaled single-pair data. A table starting
    at ``r_min>0`` is particle-only: interpolation stays fail-closed below
    ``r_min`` and continuum constructors reject it before building a kernel.
    """

    force_data: PairForceTable
    derivative_absolute_tolerance_newton: float
    derivative_relative_tolerance: float
    finite_difference_step_fraction: float
    density_convention: DensityConvention = DensityConvention.PROBABILITY
    name: str = field(init=False)
    physical_status: str = field(init=False)
    pair_force_scaling: PairForceScaling = field(init=False)
    scaling_semantics: str = field(init=False)
    scaling_population_count: int | None = field(init=False)
    minimum_supported_distance_m: float = field(init=False)
    continuum_ready: bool = field(init=False)
    validation_report: ForcePotentialValidationReport = field(init=False)
    _law: PchipIntegratedForceLaw = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        validate_density_convention(self.density_convention)
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
        expected_scaling = expected_pair_force_scaling(self.density_convention)
        if metadata.scaling is not expected_scaling:
            raise ValueError(
                "density/potential scaling mismatch: "
                f"{self.density_convention.name} requires {expected_scaling.name}, "
                f"got {metadata.scaling.name}"
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
        object.__setattr__(self, "pair_force_scaling", metadata.scaling)
        object.__setattr__(
            self, "scaling_semantics", metadata.scaling.value
        )
        evidence = metadata.scaling_conversion
        object.__setattr__(
            self,
            "scaling_population_count",
            None if evidence is None else evidence.population_count,
        )
        minimum_distance = float(self.force_data.center_distance_m[0])
        object.__setattr__(
            self,
            "minimum_supported_distance_m",
            minimum_distance,
        )
        object.__setattr__(self, "continuum_ready", minimum_distance == 0.0)

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
        flat_radial_force = self.radial_force_newton(flat_radius)
        nonzero = flat_radius > 0.0
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
