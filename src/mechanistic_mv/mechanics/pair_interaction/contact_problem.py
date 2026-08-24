"""Strongly typed inputs and readiness states for real two-gel contact solves.

This module defines inputs only.  It does not solve contact, generate force
samples, or validate an effective pair potential.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, ClassVar

import numpy as np

from ..density_scaling import DensityConvention, validate_population_count
from ..hydrogel.equilibrium import (
    TimeScaleAssessment,
    TimeScaleStatus,
    assess_time_scale_separation,
)
from ..hydrogel.parameters import HydrogelParameters
from .geometry import TwoSphereGeometry


_VAGUE_EXACT_TEXT = (
    "n/a",
    "not applicable",
    "none",
    "tbd",
    "todo",
    "unknown",
    "unspecified",
    "calibrated",
    "measured",
    "verified",
    "validated",
    "physical",
    "real data",
    "literature",
)
_VAGUE_PHRASES = (
    "provided later",
    "provide later",
    "to be provided",
    "placeholder",
    "default value",
)
_TEST_ONLY_MARKERS = (
    "test_only",
    "test-only",
    "test only",
    "not_calibrated",
    "not-calibrated",
    "not calibrated",
)
_UNVERIFIED_MARKERS = (
    "unverified",
    "not verified",
    "pending verification",
    "uncalibrated",
    "pending calibration",
    "calibration pending",
)
_VERIFIED_CALIBRATION_MARKERS = ("calibrated", "measured", "certified")
_FORBIDDEN_PHYSICAL_CLOSURE_MARKERS = (
    "hertz",
    "gaussian",
    "morse",
    "quadratic",
    "constant force",
    "force clipping",
    "clipped force",
)


def _specific_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty, specific string")
    cleaned = value.strip()
    folded = " ".join(cleaned.casefold().split())
    if folded in _VAGUE_EXACT_TEXT or any(
        phrase in folded for phrase in _VAGUE_PHRASES
    ):
        raise ValueError(f"{name} must not contain a vague placeholder")
    return cleaned


def _contains_marker(value: str, markers: tuple[str, ...]) -> bool:
    folded = value.casefold()
    return any(marker in folded for marker in markers)


def _text_leaves(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if is_dataclass(value) and not isinstance(value, type):
        return tuple(
            text
            for item in fields(value)
            for text in _text_leaves(getattr(value, item.name))
        )
    if isinstance(value, (tuple, list)):
        return tuple(text for item in value for text in _text_leaves(item))
    return ()


def _finite_float(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real scalar") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if strictly_positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _strict_real_scalar(value: object, name: str) -> float:
    """Validate nested legacy scalar fields without coercing their stored type."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real scalar, not a coercible value")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


class ContactInputPurpose(str, Enum):
    """Whether an input bundle may make physical claims."""

    PHYSICAL = "PHYSICAL_CONTACT_PROBLEM"
    TEST_ONLY_CONTRACT = "TEST_ONLY_CONTRACT_NOT_PHYSICAL"


class InputVerificationStatus(str, Enum):
    """Verification state attached to every input group."""

    VERIFIED = "VERIFIED_SOURCE_RECORD"
    UNVERIFIED = "UNVERIFIED"
    TEST_ONLY_NOT_CALIBRATED = "TEST_ONLY_NOT_CALIBRATED"


class SolverInputStatus(str, Enum):
    BLOCKED = "BLOCKED"
    READY_FOR_CONTACT_SOLVER = "READY_FOR_CONTACT_SOLVER"


class ContactResultsStatus(str, Enum):
    NOT_GENERATED = "NOT_GENERATED"


class ShortRangeInputStatus(str, Enum):
    BLOCKED = "BLOCKED"
    PARTICLE_ONLY = "PARTICLE_ONLY_SHORT_RANGE"
    CONTINUUM_INPUT_READY = "CONTINUUM_INPUT_READY"


class RequiredInputStatus(str, Enum):
    MISSING = "MISSING"
    UNVERIFIED = "UNVERIFIED"
    TEST_ONLY = "TEST_ONLY_NOT_CALIBRATED"
    READY = "READY"


class ShortRangeClosureKind(str, Enum):
    NONE = "NO_SHORT_RANGE_CLOSURE"
    EXTERNAL_VALIDATED_HYDROGEL_MECHANICS_CLOSURE = (
        "EXTERNAL_VALIDATED_HYDROGEL_MECHANICS_SHORT_RANGE_CLOSURE"
    )


class MeanFieldScalingMode(str, Enum):
    KAC_FROM_SINGLE_PAIR_POPULATION = "KAC_FROM_SINGLE_PAIR_POPULATION"
    RAW_SINGLE_PAIR_NUMBER_DENSITY = "RAW_SINGLE_PAIR_NUMBER_DENSITY"


class NormalContactLaw(str, Enum):
    """Contact mechanics admitted by the current non-adhesive result model."""

    UNILATERAL_IMPENETRABILITY = (
        "frictionless non-adhesive unilateral impenetrability"
    )


class PairAxisDefinition(str, Enum):
    """Canonical radial axis used by ``PairContactSweep`` force records."""

    SECOND_TO_FIRST_CENTER = (
        "unit vector from second sphere center to first sphere center"
    )


class ReactionForceSignConvention(str, Enum):
    """Canonical sign accepted by the non-adhesive pair-force contract."""

    POSITIVE_REPULSION = "positive radial force separates the two sphere centers"


class TotalFreeEnergyReference(str, Enum):
    """Energy zero used when converting a completed sweep to ``W_eff``."""

    ZERO_AT_FINAL_REFERENCE_DISTANCE = (
        "zero at the final separated reference distance"
    )


@dataclass(frozen=True, slots=True)
class InputProvenance:
    """Source and verification record applying to every field in one group."""

    source_id: str
    source_description: str
    verification_status: InputVerificationStatus
    verification_record_id: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _specific_text(self.source_id, "source_id"))
        object.__setattr__(
            self,
            "source_description",
            _specific_text(self.source_description, "source_description"),
        )
        if not isinstance(self.verification_status, InputVerificationStatus):
            raise TypeError("verification_status must be InputVerificationStatus")
        if self.verification_record_id is not None:
            object.__setattr__(
                self,
                "verification_record_id",
                _specific_text(
                    self.verification_record_id, "verification_record_id"
                ),
            )
        combined = " ".join(
            value
            for value in (
                self.source_id,
                self.source_description,
                self.verification_record_id,
            )
            if value is not None
        )
        contains_test_marker = _contains_marker(combined, _TEST_ONLY_MARKERS)
        contains_unverified_marker = _contains_marker(
            combined, _UNVERIFIED_MARKERS
        )
        if self.verification_status is InputVerificationStatus.VERIFIED:
            if self.verification_record_id is None:
                raise ValueError("verified input requires verification_record_id")
            if contains_test_marker:
                raise ValueError("TEST_ONLY input cannot claim VERIFIED provenance")
            if contains_unverified_marker:
                raise ValueError("UNVERIFIED input cannot claim VERIFIED provenance")
        elif self.verification_status is InputVerificationStatus.TEST_ONLY_NOT_CALIBRATED:
            if not contains_test_marker:
                raise ValueError("test-only provenance must be visibly labelled TEST_ONLY")


def _require_provenance(value: object, owner: str) -> InputProvenance:
    if not isinstance(value, InputProvenance):
        raise TypeError(f"{owner}.provenance must be InputProvenance")
    return value


@dataclass(frozen=True, slots=True)
class MechanicalBoundaryConditions:
    center_separation_control: str
    rigid_body_constraint: str
    noncontact_surface_traction: str
    loading_protocol: str
    provenance: InputProvenance

    def __post_init__(self) -> None:
        for name in (
            "center_separation_control",
            "rigid_body_constraint",
            "noncontact_surface_traction",
            "loading_protocol",
        ):
            object.__setattr__(self, name, _specific_text(getattr(self, name), name))
        _require_provenance(self.provenance, type(self).__name__)


@dataclass(frozen=True, slots=True)
class SolventBathBoundaryConditions:
    bath_chemical_potential_over_kbt: float
    exposed_surface_exchange_condition: str
    contact_surface_transport_condition: str
    initial_solvent_state: str
    provenance: InputProvenance

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "bath_chemical_potential_over_kbt",
            _finite_float(
                self.bath_chemical_potential_over_kbt,
                "bath_chemical_potential_over_kbt",
            ),
        )
        for name in (
            "exposed_surface_exchange_condition",
            "contact_surface_transport_condition",
            "initial_solvent_state",
        ):
            object.__setattr__(self, name, _specific_text(getattr(self, name), name))
        _require_provenance(self.provenance, type(self).__name__)


@dataclass(frozen=True, slots=True)
class ContactLawAndTolerances:
    normal_contact_model: NormalContactLaw
    contact_enforcement_method: str
    frictionless: bool
    adhesive: bool
    normal_gap_tolerance_m: float
    force_balance_tolerance_newton: float
    provenance: InputProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.normal_contact_model, NormalContactLaw):
            raise TypeError("normal_contact_model must be NormalContactLaw")
        object.__setattr__(
            self,
            "contact_enforcement_method",
            _specific_text(
                self.contact_enforcement_method, "contact_enforcement_method"
            ),
        )
        if not isinstance(self.frictionless, bool) or not isinstance(self.adhesive, bool):
            raise TypeError("frictionless and adhesive must be bool")
        if not self.frictionless or self.adhesive:
            raise ValueError(
                "current PairContactSweep supports frictionless, non-adhesive "
                "radial contact only"
            )
        for name in ("normal_gap_tolerance_m", "force_balance_tolerance_newton"):
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), name, minimum=0.0),
            )
        _require_provenance(self.provenance, type(self).__name__)


@dataclass(frozen=True, slots=True)
class ContactSolverSpecification:
    solver_name: str
    solver_version: str
    implementation_id: str
    configuration_id: str
    nonlinear_algorithm: str
    linear_solver: str
    provenance: InputProvenance

    def __post_init__(self) -> None:
        for name in (
            "solver_name",
            "solver_version",
            "implementation_id",
            "configuration_id",
            "nonlinear_algorithm",
            "linear_solver",
        ):
            object.__setattr__(self, name, _specific_text(getattr(self, name), name))
        _require_provenance(self.provenance, type(self).__name__)


@dataclass(frozen=True, slots=True)
class MeshConvergencePlan:
    discretization_method: str
    element_family: str
    characteristic_lengths_m: tuple[float, ...]
    nonlinear_residual_tolerance_dimensionless: float
    relative_force_convergence_tolerance_dimensionless: float
    relative_energy_convergence_tolerance_dimensionless: float
    relative_stress_convergence_tolerance_dimensionless: float
    maximum_nonlinear_iterations: int
    provenance: InputProvenance

    def __post_init__(self) -> None:
        for name in ("discretization_method", "element_family"):
            object.__setattr__(self, name, _specific_text(getattr(self, name), name))
        lengths = tuple(
            _finite_float(value, "characteristic_lengths_m", strictly_positive=True)
            for value in self.characteristic_lengths_m
        )
        if len(lengths) < 2 or np.any(np.diff(lengths) >= 0.0):
            raise ValueError(
                "characteristic_lengths_m must contain at least two strictly "
                "decreasing mesh scales"
            )
        object.__setattr__(self, "characteristic_lengths_m", lengths)
        for name in (
            "nonlinear_residual_tolerance_dimensionless",
            "relative_force_convergence_tolerance_dimensionless",
            "relative_energy_convergence_tolerance_dimensionless",
            "relative_stress_convergence_tolerance_dimensionless",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), name, strictly_positive=True),
            )
        object.__setattr__(
            self,
            "maximum_nonlinear_iterations",
            _positive_integer(
                self.maximum_nonlinear_iterations, "maximum_nonlinear_iterations"
            ),
        )
        _require_provenance(self.provenance, type(self).__name__)


@dataclass(frozen=True, slots=True)
class DistanceScanPlan:
    center_distances_m: tuple[float, ...]
    reference_distance_m: float
    reference_force_tolerance_newton: float
    provenance: InputProvenance

    def __post_init__(self) -> None:
        distances = tuple(
            _finite_float(value, "center_distances_m", strictly_positive=True)
            for value in self.center_distances_m
        )
        if len(distances) < 4 or np.any(np.diff(distances) <= 0.0):
            raise ValueError(
                "center_distances_m must contain at least four strictly increasing "
                "positive distances"
            )
        reference = _finite_float(
            self.reference_distance_m,
            "reference_distance_m",
            strictly_positive=True,
        )
        if not np.isclose(
            distances[-1],
            reference,
            rtol=16.0 * np.finfo(np.float64).eps,
            atol=0.0,
        ):
            raise ValueError("reference_distance_m must equal the final scan distance")
        object.__setattr__(self, "center_distances_m", distances)
        object.__setattr__(self, "reference_distance_m", reference)
        object.__setattr__(
            self,
            "reference_force_tolerance_newton",
            _finite_float(
                self.reference_force_tolerance_newton,
                "reference_force_tolerance_newton",
                minimum=0.0,
            ),
        )
        _require_provenance(self.provenance, type(self).__name__)


@dataclass(frozen=True, slots=True)
class ReactionForceConvention:
    integration_boundary: str
    traction_quantity: str
    pair_axis_definition: PairAxisDefinition
    positive_sign_convention: ReactionForceSignConvention
    total_free_energy_reference: TotalFreeEnergyReference
    integration_tolerance_newton: float
    provenance: InputProvenance

    def __post_init__(self) -> None:
        for name in (
            "integration_boundary",
            "traction_quantity",
        ):
            object.__setattr__(self, name, _specific_text(getattr(self, name), name))
        if not isinstance(self.pair_axis_definition, PairAxisDefinition):
            raise TypeError("pair_axis_definition must be PairAxisDefinition")
        if not isinstance(
            self.positive_sign_convention, ReactionForceSignConvention
        ):
            raise TypeError(
                "positive_sign_convention must be ReactionForceSignConvention"
            )
        if not isinstance(
            self.total_free_energy_reference, TotalFreeEnergyReference
        ):
            raise TypeError(
                "total_free_energy_reference must be TotalFreeEnergyReference"
            )
        object.__setattr__(
            self,
            "integration_tolerance_newton",
            _finite_float(
                self.integration_tolerance_newton,
                "integration_tolerance_newton",
                minimum=0.0,
            ),
        )
        _require_provenance(self.provenance, type(self).__name__)


@dataclass(frozen=True, slots=True)
class MeanFieldScalingInput:
    """Population/scaling source; ``scaling_mode`` is the authoritative rule.

    ``scaling_definition`` is a source-facing explanatory record and cannot
    override the enum or the numeric population consistency checks.
    """

    density_convention: DensityConvention
    scaling_mode: MeanFieldScalingMode
    population_count: int
    source_population_count: int
    areal_number_density_per_m2: float | None
    representative_area_m2: float | None
    scaling_definition: str
    provenance: InputProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.density_convention, DensityConvention):
            raise TypeError("density_convention must be DensityConvention")
        if not isinstance(self.scaling_mode, MeanFieldScalingMode):
            raise TypeError("scaling_mode must be MeanFieldScalingMode")
        expected_mode = (
            MeanFieldScalingMode.KAC_FROM_SINGLE_PAIR_POPULATION
            if self.density_convention is DensityConvention.PROBABILITY
            else MeanFieldScalingMode.RAW_SINGLE_PAIR_NUMBER_DENSITY
        )
        if self.scaling_mode is not expected_mode:
            raise ValueError(
                "density convention disagrees with the declared mean-field "
                "scaling mode"
            )
        count = validate_population_count(self.population_count, required=True)
        source_count = validate_population_count(
            self.source_population_count, required=True
        )
        object.__setattr__(self, "population_count", count)
        object.__setattr__(self, "source_population_count", source_count)
        if source_count != count:
            raise ValueError(
                "population_count disagrees with the scaling source population"
            )
        supplied_density = self.areal_number_density_per_m2 is not None
        supplied_area = self.representative_area_m2 is not None
        if supplied_density != supplied_area:
            raise ValueError(
                "areal_number_density_per_m2 and representative_area_m2 must "
                "be supplied together"
            )
        if supplied_density:
            density = _finite_float(
                self.areal_number_density_per_m2,
                "areal_number_density_per_m2",
                strictly_positive=True,
            )
            area = _finite_float(
                self.representative_area_m2,
                "representative_area_m2",
                strictly_positive=True,
            )
            inferred = density * area
            if not np.isclose(inferred, count, rtol=1.0e-12, atol=0.0):
                raise ValueError(
                    "population_count disagrees with areal number density times "
                    "representative area"
                )
            object.__setattr__(self, "areal_number_density_per_m2", density)
            object.__setattr__(self, "representative_area_m2", area)
        object.__setattr__(
            self,
            "scaling_definition",
            _specific_text(self.scaling_definition, "scaling_definition"),
        )
        _require_provenance(self.provenance, type(self).__name__)


@dataclass(frozen=True, slots=True)
class ShortRangeClosureInput:
    minimum_supported_distance_m: float
    closure_kind: ShortRangeClosureKind
    closure_description: str | None
    validation_method: str | None
    force_match_tolerance_newton: float | None
    energy_match_tolerance_joule: float | None
    provenance: InputProvenance

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_supported_distance_m",
            _finite_float(
                self.minimum_supported_distance_m,
                "minimum_supported_distance_m",
                minimum=0.0,
            ),
        )
        if not isinstance(self.closure_kind, ShortRangeClosureKind):
            raise TypeError("closure_kind must be ShortRangeClosureKind")
        if self.closure_kind is ShortRangeClosureKind.NONE:
            if any(
                value is not None
                for value in (
                    self.closure_description,
                    self.validation_method,
                    self.force_match_tolerance_newton,
                    self.energy_match_tolerance_joule,
                )
            ):
                raise ValueError("NO_SHORT_RANGE_CLOSURE must not carry closure values")
        else:
            object.__setattr__(
                self,
                "closure_description",
                _specific_text(self.closure_description, "closure_description"),
            )
            object.__setattr__(
                self,
                "validation_method",
                _specific_text(self.validation_method, "validation_method"),
            )
            for name in (
                "force_match_tolerance_newton",
                "energy_match_tolerance_joule",
            ):
                if getattr(self, name) is None:
                    raise ValueError(f"{name} is required for a short-range closure")
                object.__setattr__(
                    self,
                    name,
                    _finite_float(getattr(self, name), name, minimum=0.0),
                )
        _require_provenance(self.provenance, type(self).__name__)


@dataclass(frozen=True, slots=True)
class RequiredInputReport:
    requirement_id: str
    input_field_paths: tuple[str, ...]
    status: RequiredInputStatus
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "requirement_id",
            _specific_text(self.requirement_id, "requirement_id"),
        )
        paths = tuple(
            _specific_text(path, "input_field_paths") for path in self.input_field_paths
        )
        if not paths or len(set(paths)) != len(paths):
            raise ValueError("input_field_paths must be non-empty and unique")
        object.__setattr__(self, "input_field_paths", paths)
        if not isinstance(self.status, RequiredInputStatus):
            raise TypeError("status must be RequiredInputStatus")
        object.__setattr__(self, "reason", _specific_text(self.reason, "reason"))


@dataclass(frozen=True, slots=True)
class ContactProblemReadiness:
    solver_input_status: SolverInputStatus
    short_range_status: ShortRangeInputStatus
    required_inputs: tuple[RequiredInputReport, ...]
    blocking_reasons: tuple[str, ...]
    physical_inputs_verified: bool
    contact_results_status: ContactResultsStatus = field(
        default=ContactResultsStatus.NOT_GENERATED, init=False
    )
    effective_pair_potential_validated: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.solver_input_status, SolverInputStatus):
            raise TypeError("solver_input_status must be SolverInputStatus")
        if not isinstance(self.short_range_status, ShortRangeInputStatus):
            raise TypeError("short_range_status must be ShortRangeInputStatus")
        reports = tuple(self.required_inputs)
        if any(not isinstance(item, RequiredInputReport) for item in reports):
            raise TypeError("required_inputs must contain RequiredInputReport values")
        requirement_ids = tuple(item.requirement_id for item in reports)
        if requirement_ids != ContactProblemInput.REQUIRED_INPUT_NAMES:
            raise ValueError(
                "required_inputs must cover every contact requirement exactly once "
                "in schema order"
            )
        reasons = tuple(
            _specific_text(reason, "blocking_reasons")
            for reason in self.blocking_reasons
        )
        if len(set(reasons)) != len(reasons):
            raise ValueError("blocking_reasons must be unique")
        if not isinstance(self.physical_inputs_verified, bool):
            raise TypeError("physical_inputs_verified must be bool")
        blocking_reports = tuple(
            item
            for item in reports
            if item.status
            in (RequiredInputStatus.MISSING, RequiredInputStatus.UNVERIFIED)
        )
        if len(reasons) != len(blocking_reports) or any(
            not any(
                reason.startswith(f"{item.requirement_id}:")
                for reason in reasons
            )
            for item in blocking_reports
        ):
            raise ValueError(
                "blocking_reasons must correspond one-to-one with missing or "
                "unverified requirements"
            )
        if (
            self.solver_input_status is SolverInputStatus.READY_FOR_CONTACT_SOLVER
        ) != (not reasons):
            raise ValueError(
                "READY_FOR_CONTACT_SOLVER must be equivalent to having no blockers"
            )
        if self.physical_inputs_verified and (
            self.solver_input_status is not SolverInputStatus.READY_FOR_CONTACT_SOLVER
            or any(item.status is not RequiredInputStatus.READY for item in reports)
        ):
            raise ValueError(
                "physical_inputs_verified requires fully verified solver inputs"
            )
        if (
            self.short_range_status is ShortRangeInputStatus.CONTINUUM_INPUT_READY
            and not self.physical_inputs_verified
        ):
            raise ValueError(
                "CONTINUUM_INPUT_READY requires verified physical inputs"
            )
        object.__setattr__(self, "required_inputs", reports)
        object.__setattr__(self, "blocking_reasons", reasons)

    @property
    def can_submit_to_solver(self) -> bool:
        return self.solver_input_status is SolverInputStatus.READY_FOR_CONTACT_SOLVER

    @property
    def physical_continuum_input_ready(self) -> bool:
        return (
            self.physical_inputs_verified
            and self.short_range_status
            is ShortRangeInputStatus.CONTINUUM_INPUT_READY
        )

    @property
    def missing_inputs(self) -> tuple[str, ...]:
        return tuple(
            item.requirement_id
            for item in self.required_inputs
            if item.status is RequiredInputStatus.MISSING
        )

    @property
    def unverified_inputs(self) -> tuple[str, ...]:
        return tuple(
            item.requirement_id
            for item in self.required_inputs
            if item.status is RequiredInputStatus.UNVERIFIED
        )

    def to_dict(self) -> dict[str, Any]:
        payload = _native(self)
        payload["physical_continuum_input_ready"] = (
            self.physical_continuum_input_ready
        )
        return payload


@dataclass(frozen=True, slots=True)
class ContactProblemInput:
    """Complete or partially specified input to a future real contact solver.

    No numerical field has a default.  Missing groups remain ``None`` and are
    reported as BLOCKED; no value is inferred or fabricated.
    """

    REQUIRED_INPUT_FIELD_PATHS: ClassVar[
        tuple[tuple[str, tuple[str, ...]], ...]
    ] = (
        (
            "material.hydrogel_parameters",
            (
                "hydrogel_parameters.network_density_times_solvent_volume",
                "hydrogel_parameters.flory_huggins_chi",
                "hydrogel_parameters.initial_polymer_volume_fraction",
                "hydrogel_parameters.delta_chemical_potential_over_kbt",
                "hydrogel_parameters.thermal_energy_density_pa",
                "hydrogel_parameters.calibration_status",
                "hydrogel_parameters_provenance.source_id",
                "hydrogel_parameters_provenance.source_description",
                "hydrogel_parameters_provenance.verification_status",
                "hydrogel_parameters_provenance.verification_record_id",
            ),
        ),
        (
            "geometry.two_sphere",
            (
                "geometry.first_radius_m",
                "geometry.second_radius_m",
                "geometry.calibration_status",
                "geometry_provenance.source_id",
                "geometry_provenance.source_description",
                "geometry_provenance.verification_status",
                "geometry_provenance.verification_record_id",
            ),
        ),
        (
            "boundary.mechanical",
            (
                "mechanical_boundary_conditions.center_separation_control",
                "mechanical_boundary_conditions.rigid_body_constraint",
                "mechanical_boundary_conditions.noncontact_surface_traction",
                "mechanical_boundary_conditions.loading_protocol",
                "mechanical_boundary_conditions.provenance.source_id",
                "mechanical_boundary_conditions.provenance.source_description",
                "mechanical_boundary_conditions.provenance.verification_status",
                "mechanical_boundary_conditions.provenance.verification_record_id",
            ),
        ),
        (
            "boundary.solvent_bath",
            (
                "solvent_bath_boundary_conditions.bath_chemical_potential_over_kbt",
                "solvent_bath_boundary_conditions.exposed_surface_exchange_condition",
                "solvent_bath_boundary_conditions.contact_surface_transport_condition",
                "solvent_bath_boundary_conditions.initial_solvent_state",
                "solvent_bath_boundary_conditions.provenance.source_id",
                "solvent_bath_boundary_conditions.provenance.source_description",
                "solvent_bath_boundary_conditions.provenance.verification_status",
                "solvent_bath_boundary_conditions.provenance.verification_record_id",
            ),
        ),
        (
            "contact.law_and_tolerances",
            (
                "contact_law.normal_contact_model",
                "contact_law.contact_enforcement_method",
                "contact_law.frictionless",
                "contact_law.adhesive",
                "contact_law.normal_gap_tolerance_m",
                "contact_law.force_balance_tolerance_newton",
                "contact_law.provenance.source_id",
                "contact_law.provenance.source_description",
                "contact_law.provenance.verification_status",
                "contact_law.provenance.verification_record_id",
            ),
        ),
        (
            "solver.identity_and_configuration",
            (
                "solver.solver_name",
                "solver.solver_version",
                "solver.implementation_id",
                "solver.configuration_id",
                "solver.nonlinear_algorithm",
                "solver.linear_solver",
                "solver.provenance.source_id",
                "solver.provenance.source_description",
                "solver.provenance.verification_status",
                "solver.provenance.verification_record_id",
            ),
        ),
        (
            "mesh.convergence_plan",
            (
                "mesh_convergence.discretization_method",
                "mesh_convergence.element_family",
                "mesh_convergence.characteristic_lengths_m",
                "mesh_convergence.nonlinear_residual_tolerance_dimensionless",
                "mesh_convergence.relative_force_convergence_tolerance_dimensionless",
                "mesh_convergence.relative_energy_convergence_tolerance_dimensionless",
                "mesh_convergence.relative_stress_convergence_tolerance_dimensionless",
                "mesh_convergence.maximum_nonlinear_iterations",
                "mesh_convergence.provenance.source_id",
                "mesh_convergence.provenance.source_description",
                "mesh_convergence.provenance.verification_status",
                "mesh_convergence.provenance.verification_record_id",
            ),
        ),
        (
            "distance.scan_and_reference_force_tolerance",
            (
                "distance_scan.center_distances_m",
                "distance_scan.reference_distance_m",
                "distance_scan.reference_force_tolerance_newton",
                "distance_scan.provenance.source_id",
                "distance_scan.provenance.source_description",
                "distance_scan.provenance.verification_status",
                "distance_scan.provenance.verification_record_id",
            ),
        ),
        (
            "reaction.integration_and_sign",
            (
                "reaction_force.integration_boundary",
                "reaction_force.traction_quantity",
                "reaction_force.pair_axis_definition",
                "reaction_force.positive_sign_convention",
                "reaction_force.total_free_energy_reference",
                "reaction_force.integration_tolerance_newton",
                "reaction_force.provenance.source_id",
                "reaction_force.provenance.source_description",
                "reaction_force.provenance.verification_status",
                "reaction_force.provenance.verification_record_id",
            ),
        ),
        (
            "timescale.separation_and_source",
            (
                "time_scale.tau_gel_s",
                "time_scale.tau_swarm_s",
                "time_scale.required_max_ratio",
                "time_scale_provenance.source_id",
                "time_scale_provenance.source_description",
                "time_scale_provenance.verification_status",
                "time_scale_provenance.verification_record_id",
            ),
        ),
        (
            "mean_field.population_or_kac_scaling_source",
            (
                "mean_field_scaling.density_convention",
                "mean_field_scaling.scaling_mode",
                "mean_field_scaling.population_count",
                "mean_field_scaling.source_population_count",
                "mean_field_scaling.areal_number_density_per_m2",
                "mean_field_scaling.representative_area_m2",
                "mean_field_scaling.scaling_definition",
                "mean_field_scaling.provenance.source_id",
                "mean_field_scaling.provenance.source_description",
                "mean_field_scaling.provenance.verification_status",
                "mean_field_scaling.provenance.verification_record_id",
            ),
        ),
        (
            "short_range.minimum_distance_or_zero_closure",
            (
                "short_range.minimum_supported_distance_m",
                "short_range.closure_kind",
                "short_range.closure_description",
                "short_range.validation_method",
                "short_range.force_match_tolerance_newton",
                "short_range.energy_match_tolerance_joule",
                "short_range.provenance.source_id",
                "short_range.provenance.source_description",
                "short_range.provenance.verification_status",
                "short_range.provenance.verification_record_id",
            ),
        ),
    )
    REQUIRED_INPUT_NAMES: ClassVar[tuple[str, ...]] = tuple(
        requirement_id for requirement_id, _ in REQUIRED_INPUT_FIELD_PATHS
    )

    purpose: ContactInputPurpose
    hydrogel_parameters: HydrogelParameters | None = None
    hydrogel_parameters_provenance: InputProvenance | None = None
    geometry: TwoSphereGeometry | None = None
    geometry_provenance: InputProvenance | None = None
    mechanical_boundary_conditions: MechanicalBoundaryConditions | None = None
    solvent_bath_boundary_conditions: SolventBathBoundaryConditions | None = None
    contact_law: ContactLawAndTolerances | None = None
    solver: ContactSolverSpecification | None = None
    mesh_convergence: MeshConvergencePlan | None = None
    distance_scan: DistanceScanPlan | None = None
    reaction_force: ReactionForceConvention | None = None
    time_scale: TimeScaleAssessment | None = None
    time_scale_provenance: InputProvenance | None = None
    mean_field_scaling: MeanFieldScalingInput | None = None
    short_range: ShortRangeClosureInput | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.purpose, ContactInputPurpose):
            raise TypeError("purpose must be ContactInputPurpose")
        type_pairs = (
            ("hydrogel_parameters", HydrogelParameters),
            ("hydrogel_parameters_provenance", InputProvenance),
            ("geometry", TwoSphereGeometry),
            ("geometry_provenance", InputProvenance),
            ("mechanical_boundary_conditions", MechanicalBoundaryConditions),
            ("solvent_bath_boundary_conditions", SolventBathBoundaryConditions),
            ("contact_law", ContactLawAndTolerances),
            ("solver", ContactSolverSpecification),
            ("mesh_convergence", MeshConvergencePlan),
            ("distance_scan", DistanceScanPlan),
            ("reaction_force", ReactionForceConvention),
            ("time_scale", TimeScaleAssessment),
            ("time_scale_provenance", InputProvenance),
            ("mean_field_scaling", MeanFieldScalingInput),
            ("short_range", ShortRangeClosureInput),
        )
        for name, expected_type in type_pairs:
            value = getattr(self, name)
            if value is not None and not isinstance(value, expected_type):
                raise TypeError(f"{name} must be {expected_type.__name__} or None")

        if self.hydrogel_parameters is not None:
            for name in (
                "network_density_times_solvent_volume",
                "flory_huggins_chi",
                "initial_polymer_volume_fraction",
                "delta_chemical_potential_over_kbt",
                "thermal_energy_density_pa",
            ):
                _strict_real_scalar(
                    getattr(self.hydrogel_parameters, name),
                    f"hydrogel_parameters.{name}",
                )
            calibration = _specific_text(
                self.hydrogel_parameters.calibration_status,
                "hydrogel_parameters.calibration_status",
            )
            self._validate_physical_calibration_claim(calibration)
        if self.geometry is not None:
            for name in ("first_radius_m", "second_radius_m"):
                _strict_real_scalar(
                    getattr(self.geometry, name), f"geometry.{name}"
                )
            calibration = _specific_text(
                self.geometry.calibration_status,
                "geometry.calibration_status",
            )
            self._validate_physical_calibration_claim(calibration)

        for provenance in self._all_provenance():
            if (
                self.purpose is ContactInputPurpose.PHYSICAL
                and provenance.verification_status
                is InputVerificationStatus.TEST_ONLY_NOT_CALIBRATED
            ):
                raise ValueError("TEST_ONLY input cannot claim a physical contact problem")
            if (
                self.purpose is ContactInputPurpose.TEST_ONLY_CONTRACT
                and provenance.verification_status
                is InputVerificationStatus.VERIFIED
            ):
                raise ValueError(
                    "test-only contact problems cannot attach VERIFIED physical provenance"
                )

        if self.purpose is ContactInputPurpose.PHYSICAL:
            physical_groups = tuple(
                group_part
                for _, group_value, provenance in self._group_values()
                for group_part in (group_value, provenance)
                if group_part is not None
            )
            if any(
                _contains_marker(text, _TEST_ONLY_MARKERS)
                for group in physical_groups
                for text in _text_leaves(group)
            ):
                raise ValueError(
                    "TEST_ONLY values cannot claim a physical contact problem"
                )

        if self.purpose is ContactInputPurpose.TEST_ONLY_CONTRACT:
            for name, value in (
                (
                    "hydrogel_parameters.calibration_status",
                    None
                    if self.hydrogel_parameters is None
                    else self.hydrogel_parameters.calibration_status,
                ),
                (
                    "geometry.calibration_status",
                    None if self.geometry is None else self.geometry.calibration_status,
                ),
            ):
                if value is not None and not _contains_marker(value, _TEST_ONLY_MARKERS):
                    raise ValueError(f"{name} must be visibly TEST_ONLY")

        if self.time_scale is not None:
            for name in ("tau_gel_s", "tau_swarm_s", "required_max_ratio"):
                value = getattr(self.time_scale, name)
                if value is not None:
                    _strict_real_scalar(value, f"time_scale.{name}")
            recomputed = assess_time_scale_separation(
                tau_gel_s=self.time_scale.tau_gel_s,
                tau_swarm_s=self.time_scale.tau_swarm_s,
                required_max_ratio=self.time_scale.required_max_ratio,
            )
            object.__setattr__(self, "time_scale", recomputed)

        if (
            self.hydrogel_parameters is not None
            and self.solvent_bath_boundary_conditions is not None
            and not np.isclose(
                self.hydrogel_parameters.bath_chemical_potential_over_kbt,
                self.solvent_bath_boundary_conditions.bath_chemical_potential_over_kbt,
                rtol=64.0 * np.finfo(np.float64).eps,
                atol=0.0,
            )
        ):
            raise ValueError(
                "bath boundary chemical potential disagrees with HydrogelParameters"
            )
        if self.geometry is not None and self.distance_scan is not None:
            separated_distance = self.geometry.first_radius_m + self.geometry.second_radius_m
            if self.distance_scan.reference_distance_m <= separated_distance:
                raise ValueError(
                    "reference_distance_m must exceed the sum of undeformed radii"
                )
        if self.short_range is not None and self.distance_scan is not None:
            if (
                self.short_range.minimum_supported_distance_m
                > self.distance_scan.center_distances_m[0]
            ):
                raise ValueError(
                    "minimum_supported_distance_m cannot exceed the first scan distance"
                )
            if (
                self.short_range.closure_kind is ShortRangeClosureKind.NONE
                and self.short_range.minimum_supported_distance_m > 0.0
                and not np.isclose(
                    self.short_range.minimum_supported_distance_m,
                    self.distance_scan.center_distances_m[0],
                    rtol=16.0 * np.finfo(np.float64).eps,
                    atol=0.0,
                )
            ):
                raise ValueError(
                    "without a short-range closure, minimum_supported_distance_m "
                    "must equal the first scan distance"
                )
        if (
            self.purpose is ContactInputPurpose.PHYSICAL
            and self.short_range is not None
            and self.short_range.closure_kind
            is not ShortRangeClosureKind.NONE
        ):
            closure_record = " ".join(
                value
                for value in (
                    self.short_range.closure_description,
                    self.short_range.validation_method,
                    self.short_range.provenance.source_id,
                    self.short_range.provenance.source_description,
                    self.short_range.provenance.verification_record_id,
                )
                if value is not None
            ).casefold()
            if any(
                marker in closure_record
                for marker in _FORBIDDEN_PHYSICAL_CLOSURE_MARKERS
            ):
                raise ValueError(
                    "Hertz, Gaussian, Morse, constant, or clipped short-range "
                    "closures cannot claim physical Hydrogel readiness"
                )

    def _validate_physical_calibration_claim(self, value: str) -> None:
        if self.purpose is not ContactInputPurpose.PHYSICAL:
            return
        if _contains_marker(value, _TEST_ONLY_MARKERS):
            raise ValueError(
                "TEST_ONLY calibration cannot claim physical completion"
            )

    def _all_provenance(self) -> tuple[InputProvenance, ...]:
        direct = (
            self.hydrogel_parameters_provenance,
            self.geometry_provenance,
            self.time_scale_provenance,
        )
        embedded = tuple(
            value.provenance
            for value in (
                self.mechanical_boundary_conditions,
                self.solvent_bath_boundary_conditions,
                self.contact_law,
                self.solver,
                self.mesh_convergence,
                self.distance_scan,
                self.reaction_force,
                self.mean_field_scaling,
                self.short_range,
            )
            if value is not None
        )
        return tuple(value for value in direct if value is not None) + embedded

    def _group_values(self) -> tuple[tuple[str, object | None, InputProvenance | None], ...]:
        return (
            (
                self.REQUIRED_INPUT_NAMES[0],
                self.hydrogel_parameters,
                self.hydrogel_parameters_provenance,
            ),
            (
                self.REQUIRED_INPUT_NAMES[1],
                self.geometry,
                self.geometry_provenance,
            ),
            *tuple(
                (name, value, None if value is None else value.provenance)
                for name, value in zip(
                    self.REQUIRED_INPUT_NAMES[2:9],
                    (
                        self.mechanical_boundary_conditions,
                        self.solvent_bath_boundary_conditions,
                        self.contact_law,
                        self.solver,
                        self.mesh_convergence,
                        self.distance_scan,
                        self.reaction_force,
                    ),
                    strict=True,
                )
            ),
            (
                self.REQUIRED_INPUT_NAMES[9],
                self.time_scale,
                self.time_scale_provenance,
            ),
            (
                self.REQUIRED_INPUT_NAMES[10],
                self.mean_field_scaling,
                None if self.mean_field_scaling is None else self.mean_field_scaling.provenance,
            ),
            (
                self.REQUIRED_INPUT_NAMES[11],
                self.short_range,
                None if self.short_range is None else self.short_range.provenance,
            ),
        )

    def evaluate_readiness(self) -> ContactProblemReadiness:
        reports: list[RequiredInputReport] = []
        blocking_reasons: list[str] = []
        field_paths_by_group = dict(self.REQUIRED_INPUT_FIELD_PATHS)
        for field_name, value, provenance in self._group_values():
            if value is None or provenance is None:
                reports.append(
                    RequiredInputReport(
                        field_name,
                        field_paths_by_group[field_name],
                        RequiredInputStatus.MISSING,
                        "input or provenance is missing",
                    )
                )
                blocking_reasons.append(f"{field_name}: missing")
                continue
            status = RequiredInputStatus.READY
            reason = "input and source record are verified"
            if provenance.verification_status is InputVerificationStatus.UNVERIFIED:
                status = RequiredInputStatus.UNVERIFIED
                reason = "source record is unverified"
            elif (
                provenance.verification_status
                is InputVerificationStatus.TEST_ONLY_NOT_CALIBRATED
            ):
                status = RequiredInputStatus.TEST_ONLY
                reason = "input is explicitly test-only and not calibrated"

            if field_name == self.REQUIRED_INPUT_NAMES[0]:
                calibration = self.hydrogel_parameters.calibration_status
                if (
                    provenance.verification_status
                    is not InputVerificationStatus.TEST_ONLY_NOT_CALIBRATED
                    and (
                        _contains_marker(calibration, _UNVERIFIED_MARKERS)
                        or not _contains_marker(
                            calibration, _VERIFIED_CALIBRATION_MARKERS
                        )
                    )
                ):
                    status = RequiredInputStatus.UNVERIFIED
                    reason = "Hydrogel calibration is unverified"
            elif field_name == self.REQUIRED_INPUT_NAMES[1]:
                calibration = self.geometry.calibration_status
                if (
                    provenance.verification_status
                    is not InputVerificationStatus.TEST_ONLY_NOT_CALIBRATED
                    and (
                        _contains_marker(calibration, _UNVERIFIED_MARKERS)
                        or not _contains_marker(
                            calibration, _VERIFIED_CALIBRATION_MARKERS
                        )
                    )
                ):
                    status = RequiredInputStatus.UNVERIFIED
                    reason = "geometry calibration is unverified"
            elif field_name == self.REQUIRED_INPUT_NAMES[9]:
                if self.time_scale.status is not TimeScaleStatus.SATISFIED:
                    status = RequiredInputStatus.UNVERIFIED
                    reason = f"time-scale status is {self.time_scale.status.value}"
            elif field_name == self.REQUIRED_INPUT_NAMES[11]:
                if (
                    self.short_range.minimum_supported_distance_m == 0.0
                    and self.short_range.closure_kind is ShortRangeClosureKind.NONE
                ):
                    status = RequiredInputStatus.UNVERIFIED
                    reason = "zero minimum distance requires a validated closure"

            reports.append(
                RequiredInputReport(
                    field_name,
                    field_paths_by_group[field_name],
                    status,
                    reason,
                )
            )
            if status in (RequiredInputStatus.MISSING, RequiredInputStatus.UNVERIFIED):
                blocking_reasons.append(f"{field_name}: {reason}")

        solver_status = (
            SolverInputStatus.READY_FOR_CONTACT_SOLVER
            if not blocking_reasons
            else SolverInputStatus.BLOCKED
        )
        all_provenance_verified = all(
            item.verification_status is InputVerificationStatus.VERIFIED
            for item in self._all_provenance()
        )
        physical_inputs_verified = (
            self.purpose is ContactInputPurpose.PHYSICAL
            and solver_status is SolverInputStatus.READY_FOR_CONTACT_SOLVER
            and all_provenance_verified
        )
        short_status = ShortRangeInputStatus.BLOCKED
        if (
            self.short_range is not None
            and self.short_range.provenance.verification_status
            is not InputVerificationStatus.UNVERIFIED
        ):
            if self.short_range.minimum_supported_distance_m > 0.0:
                short_status = ShortRangeInputStatus.PARTICLE_ONLY
            elif (
                self.short_range.closure_kind
                is ShortRangeClosureKind.EXTERNAL_VALIDATED_HYDROGEL_MECHANICS_CLOSURE
                and physical_inputs_verified
            ):
                short_status = ShortRangeInputStatus.CONTINUUM_INPUT_READY
        return ContactProblemReadiness(
            solver_input_status=solver_status,
            short_range_status=short_status,
            required_inputs=tuple(reports),
            blocking_reasons=tuple(blocking_reasons),
            physical_inputs_verified=physical_inputs_verified,
        )

    def required_inputs_report(self) -> tuple[RequiredInputReport, ...]:
        return self.evaluate_readiness().required_inputs

    @classmethod
    def required_input_field_paths(cls) -> dict[str, tuple[str, ...]]:
        """Return stable field paths for parameter collection UIs or schemas."""

        return dict(cls.REQUIRED_INPUT_FIELD_PATHS)

    def to_input_dict(self) -> dict[str, Any]:
        payload = _native(self)
        payload["schema_name"] = "mechanistic_mv.contact_problem_input"
        payload["schema_version"] = 1
        payload["readiness"] = self.evaluate_readiness().to_dict()
        return payload


def _native(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _native(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, tuple):
        return [_native(item) for item in value]
    if isinstance(value, list):
        return [_native(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    return value
