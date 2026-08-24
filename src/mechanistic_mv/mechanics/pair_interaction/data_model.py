"""Validated, immutable contracts for two-gel contact sweeps and pair forces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..hydrogel.equilibrium import TimeScaleAssessment, TimeScaleStatus
from ..hydrogel.parameters import HydrogelParameters
from .geometry import TwoSphereGeometry


_VAGUE_REQUIRED_TEXT = {
    "n/a",
    "na",
    "none",
    "not applicable",
    "tbd",
    "todo",
    "unknown",
    "unspecified",
}


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    cleaned = value.strip()
    if cleaned.casefold() in _VAGUE_REQUIRED_TEXT:
        raise ValueError(f"{name} must be specific, not {cleaned!r}")
    return cleaned


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
        number = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real scalar") from error
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if strictly_positive and number <= 0.0:
        raise ValueError(f"{name} must be positive")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return number


class PairDataValidationStatus(str, Enum):
    """Whether a complete external dataset passed its upstream checks."""

    PASSED = "PASSED"
    FAILED = "FAILED"
    UNVERIFIED = "UNVERIFIED"


class ContactSolveStatus(str, Enum):
    """Per-distance nonlinear/contact solve outcome, separate from data validation."""

    CONVERGED = "CONVERGED"
    FAILED = "FAILED"
    NOT_RUN = "NOT_RUN"


class PairForceScaling(str, Enum):
    """Scaling meaning of the tabulated force and its integrated potential."""

    KAC_NORMALIZED_PROBABILITY = (
        "KAC_EFFECTIVE_FOR_UNIT_MASS_RHO_AND_ONE_OVER_N_PARTICLE_FORCE"
    )
    UNSCALED_SINGLE_PAIR = "UNSCALED_PHYSICAL_SINGLE_PAIR"


@dataclass(frozen=True, slots=True)
class QuantityDefinition:
    """Definition and unit for an otherwise source-dependent scalar quantity.

    ``si_unit_or_one`` must name the SI unit used by every sample, or be ``"1"``
    for a dimensionless quantity. This avoids silently deciding whether the
    source's generic ``contact_measure`` or ``max_deformation`` means an area,
    length, strain, or another norm.
    """

    quantity_name: str
    si_unit_or_one: str
    description: str

    def __post_init__(self) -> None:
        for name in ("quantity_name", "si_unit_or_one", "description"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class ScalarDiagnostic:
    """One finite solvent-state diagnostic with an explicit SI unit or ``1``."""

    quantity_name: str
    value_si_or_dimensionless: float
    si_unit_or_one: str
    description: str

    def __post_init__(self) -> None:
        for name in ("quantity_name", "si_unit_or_one", "description"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        object.__setattr__(
            self,
            "value_si_or_dimensionless",
            _finite_float(
                self.value_si_or_dimensionless,
                "value_si_or_dimensionless",
            ),
        )


@dataclass(frozen=True, slots=True)
class MeshOrResolutionMetadata:
    """Per-distance discretization record; no anonymous mesh label is accepted."""

    method: str
    resolution_id: str
    solver_configuration_id: str
    degrees_of_freedom: int | None
    characteristic_length_m: float | None
    solver_iterations: int | None
    final_residual_dimensionless: float | None

    def __post_init__(self) -> None:
        for name in ("method", "resolution_id", "solver_configuration_id"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        for name in ("degrees_of_freedom", "solver_iterations"):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, (bool, np.bool_)) or not isinstance(
                    value, (int, np.integer)
                ):
                    raise TypeError(f"{name} must be an integer when supplied")
                if value <= 0:
                    raise ValueError(f"{name} must be positive when supplied")
                object.__setattr__(self, name, int(value))
        if self.characteristic_length_m is not None:
            object.__setattr__(
                self,
                "characteristic_length_m",
                _finite_float(
                    self.characteristic_length_m,
                    "characteristic_length_m",
                    strictly_positive=True,
                ),
            )
        if self.final_residual_dimensionless is not None:
            object.__setattr__(
                self,
                "final_residual_dimensionless",
                _finite_float(
                    self.final_residual_dimensionless,
                    "final_residual_dimensionless",
                    minimum=0.0,
                ),
            )


@dataclass(frozen=True, slots=True)
class PairContactSample:
    """Complete record at one center distance.

    Distance, radial force, energy, and maximum stress use metres, newtons,
    joules, and pascals. ``contact_measure_si_or_dimensionless`` and
    ``maximum_deformation_si_or_dimensionless`` use the definitions stored once
    in :class:`PairContactSweepMetadata`. Each solvent diagnostic carries its
    own unit.

    A converged sample must contain every physical result. For ``FAILED`` or
    ``NOT_RUN``, every physical result must be ``None`` and ``failure_reason``
    is mandatory; zero and NaN are never used as missing-value sentinels.
    Resolution metadata remains mandatory for every requested distance.
    """

    center_distance_m: float
    radial_reaction_force_newton: float | None
    total_free_energy_joule: float | None
    contact_measure_si_or_dimensionless: float | None
    maximum_deformation_si_or_dimensionless: float | None
    maximum_stress_pascal: float | None
    solvent_state_summary: tuple[ScalarDiagnostic, ...] | None
    solver_status: ContactSolveStatus
    mesh_or_resolution_metadata: MeshOrResolutionMetadata
    failure_reason: str | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "center_distance_m",
            _finite_float(
                self.center_distance_m,
                "center_distance_m",
                strictly_positive=True,
            ),
        )
        if not isinstance(self.solver_status, ContactSolveStatus):
            raise TypeError("solver_status must be a ContactSolveStatus")
        if not isinstance(
            self.mesh_or_resolution_metadata, MeshOrResolutionMetadata
        ):
            raise TypeError(
                "mesh_or_resolution_metadata must be MeshOrResolutionMetadata"
            )

        physical_names = (
            "radial_reaction_force_newton",
            "total_free_energy_joule",
            "contact_measure_si_or_dimensionless",
            "maximum_deformation_si_or_dimensionless",
            "maximum_stress_pascal",
        )
        if self.solver_status is ContactSolveStatus.CONVERGED:
            if self.failure_reason is not None:
                raise ValueError("a converged sample cannot have failure_reason")
            if any(getattr(self, name) is None for name in physical_names):
                raise ValueError("a converged sample requires every physical output")
            if (
                not isinstance(self.solvent_state_summary, tuple)
                or not self.solvent_state_summary
            ):
                raise ValueError(
                    "a converged sample requires a non-empty solvent_state_summary tuple"
                )
            names: set[str] = set()
            for diagnostic in self.solvent_state_summary:
                if not isinstance(diagnostic, ScalarDiagnostic):
                    raise TypeError(
                        "solvent_state_summary entries must be ScalarDiagnostic"
                    )
                if diagnostic.quantity_name in names:
                    raise ValueError(
                        "solvent_state_summary quantity names must be unique"
                    )
                names.add(diagnostic.quantity_name)
            resolution = self.mesh_or_resolution_metadata
            if (
                resolution.degrees_of_freedom is None
                and resolution.characteristic_length_m is None
            ):
                raise ValueError(
                    "a converged sample requires degrees_of_freedom or "
                    "characteristic_length_m"
                )
            if (
                resolution.solver_iterations is None
                or resolution.final_residual_dimensionless is None
            ):
                raise ValueError(
                    "a converged sample requires solver iterations and final residual"
                )
        else:
            if any(getattr(self, name) is not None for name in physical_names):
                raise ValueError(
                    "failed or unrun samples must not expose partial physical outputs"
                )
            if self.solvent_state_summary is not None:
                raise ValueError(
                    "failed or unrun samples must use None for solvent_state_summary"
                )
            object.__setattr__(
                self,
                "failure_reason",
                _required_text(self.failure_reason, "failure_reason"),
            )
            return

        numerical_rules = (
            ("radial_reaction_force_newton", 0.0),
            ("total_free_energy_joule", None),
            ("contact_measure_si_or_dimensionless", 0.0),
            ("maximum_deformation_si_or_dimensionless", 0.0),
            ("maximum_stress_pascal", 0.0),
        )
        for name, minimum in numerical_rules:
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), name, minimum=minimum),
            )


@dataclass(frozen=True, slots=True)
class PairContactSweepMetadata:
    """One immutable provenance block shared by every sample in a sweep.

    The raw two-particle FEM result has ``UNSCALED_SINGLE_PAIR`` semantics.
    Converting it to Kac/normalized-density scaling requires a separate,
    physically documented concentration or effective-particle-number map; this
    contract deliberately cannot relabel the raw force as Kac-scaled.
    """

    dataset_id: str
    source: str
    physical_status: str
    geometry: TwoSphereGeometry
    material_parameters: HydrogelParameters
    time_scale: TimeScaleAssessment
    solver_name: str
    solver_version: str
    solver_configuration_id: str
    resolution_method: str
    mechanical_boundary_conditions: str
    solvent_bath_boundary_conditions: str
    contact_law: str
    radial_reaction_force_sign_convention: str
    total_free_energy_reference: str
    contact_measure_definition: QuantityDefinition
    maximum_deformation_definition: QuantityDefinition
    maximum_stress_definition: QuantityDefinition
    validation_status: PairDataValidationStatus
    reference_distance_m: float
    reference_force_tolerance_newton: float

    def __post_init__(self) -> None:
        text_fields = (
            "dataset_id",
            "source",
            "physical_status",
            "solver_name",
            "solver_version",
            "solver_configuration_id",
            "resolution_method",
            "mechanical_boundary_conditions",
            "solvent_bath_boundary_conditions",
            "contact_law",
            "radial_reaction_force_sign_convention",
            "total_free_energy_reference",
        )
        for name in text_fields:
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        if not isinstance(self.geometry, TwoSphereGeometry):
            raise TypeError("geometry must be TwoSphereGeometry")
        if not isinstance(self.material_parameters, HydrogelParameters):
            raise TypeError("material_parameters must be HydrogelParameters")
        if not isinstance(self.time_scale, TimeScaleAssessment):
            raise TypeError("time_scale must be TimeScaleAssessment")
        if not isinstance(self.validation_status, PairDataValidationStatus):
            raise TypeError("validation_status must be PairDataValidationStatus")
        for name in (
            "contact_measure_definition",
            "maximum_deformation_definition",
            "maximum_stress_definition",
        ):
            if not isinstance(getattr(self, name), QuantityDefinition):
                raise TypeError(f"{name} must be QuantityDefinition")
        if self.maximum_stress_definition.si_unit_or_one != "Pa":
            raise ValueError("maximum_stress_definition must use SI unit 'Pa'")
        object.__setattr__(
            self,
            "reference_distance_m",
            _finite_float(
                self.reference_distance_m,
                "reference_distance_m",
                strictly_positive=True,
            ),
        )
        object.__setattr__(
            self,
            "reference_force_tolerance_newton",
            _finite_float(
                self.reference_force_tolerance_newton,
                "reference_force_tolerance_newton",
                minimum=0.0,
            ),
        )


@dataclass(frozen=True, slots=True)
class PairForceMetadata:
    """Required provenance and validation semantics; units are fixed to SI."""

    dataset_id: str
    source: str
    physical_status: str
    solver_status: str
    validation_status: PairDataValidationStatus
    time_scale_status: TimeScaleStatus
    scaling: PairForceScaling
    reference_distance_m: float
    reference_force_tolerance_newton: float

    def __post_init__(self) -> None:
        for name in ("dataset_id", "source", "physical_status", "solver_status"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        if not isinstance(self.scaling, PairForceScaling):
            raise TypeError("scaling must be a PairForceScaling value")
        if not isinstance(self.validation_status, PairDataValidationStatus):
            raise TypeError("validation_status must be a PairDataValidationStatus")
        if not isinstance(self.time_scale_status, TimeScaleStatus):
            raise TypeError("time_scale_status must be a TimeScaleStatus")
        object.__setattr__(
            self,
            "reference_distance_m",
            _finite_float(
                self.reference_distance_m,
                "reference_distance_m",
                strictly_positive=True,
            ),
        )
        object.__setattr__(
            self,
            "reference_force_tolerance_newton",
            _finite_float(
                self.reference_force_tolerance_newton,
                "reference_force_tolerance_newton",
                minimum=0.0,
            ),
        )


@dataclass(frozen=True, slots=True)
class PairForceTable:
    """Strict SI table of radial force ``F(r)`` with positive repulsion sign."""

    center_distance_m: np.ndarray
    radial_force_newton: np.ndarray
    metadata: PairForceMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, PairForceMetadata):
            raise TypeError("metadata must be PairForceMetadata")
        distance = np.asarray(self.center_distance_m, dtype=np.float64)
        force = np.asarray(self.radial_force_newton, dtype=np.float64)
        if distance.ndim != 1 or force.ndim != 1 or distance.shape != force.shape:
            raise ValueError("distance and force must be matching one-dimensional arrays")
        if distance.size < 4:
            raise ValueError("at least four force samples are required")
        if not np.all(np.isfinite(distance)) or not np.all(np.isfinite(force)):
            raise ValueError("pair-force data must be finite")
        if np.any(distance < 0.0) or np.any(np.diff(distance) <= 0.0):
            raise ValueError("center distances must be non-negative and strictly increasing")
        if np.any(force < 0.0):
            raise ValueError(
                "radial force must be non-negative for frictionless, non-adhesive contact"
            )
        if not np.isclose(
            distance[-1],
            self.metadata.reference_distance_m,
            rtol=16.0 * np.finfo(np.float64).eps,
            atol=0.0,
        ):
            raise ValueError("the final distance must equal the metadata reference distance")
        if abs(force[-1]) > self.metadata.reference_force_tolerance_newton:
            raise ValueError("force at r_ref exceeds the declared negligible-force tolerance")
        distance = distance.copy()
        force = force.copy()
        distance.setflags(write=False)
        force.setflags(write=False)
        object.__setattr__(self, "center_distance_m", distance)
        object.__setattr__(self, "radial_force_newton", force)


@dataclass(frozen=True, slots=True)
class PairContactSweep:
    """Ordered contact records plus one shared, immutable provenance block."""

    samples: tuple[PairContactSample, ...]
    metadata: PairContactSweepMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, PairContactSweepMetadata):
            raise TypeError("metadata must be PairContactSweepMetadata")
        if not isinstance(self.samples, tuple) or not self.samples:
            raise ValueError("samples must be a non-empty tuple")
        if not all(isinstance(sample, PairContactSample) for sample in self.samples):
            raise TypeError("samples must contain only PairContactSample values")
        distance = np.fromiter(
            (sample.center_distance_m for sample in self.samples),
            dtype=np.float64,
            count=len(self.samples),
        )
        if np.any(np.diff(distance) <= 0.0):
            raise ValueError("contact-sweep center distances must be strictly increasing")
        if not np.isclose(
            distance[-1],
            self.metadata.reference_distance_m,
            rtol=16.0 * np.finfo(np.float64).eps,
            atol=0.0,
        ):
            raise ValueError(
                "the final contact-sweep distance must equal reference_distance_m"
            )
        for sample in self.samples:
            if (
                sample.mesh_or_resolution_metadata.method
                != self.metadata.resolution_method
            ):
                raise ValueError("all samples must use the sweep resolution_method")
            if (
                sample.mesh_or_resolution_metadata.solver_configuration_id
                != self.metadata.solver_configuration_id
            ):
                raise ValueError(
                    "all samples must use the sweep solver_configuration_id"
                )
        if (
            self.metadata.validation_status is PairDataValidationStatus.PASSED
            and any(
                sample.solver_status is not ContactSolveStatus.CONVERGED
                for sample in self.samples
            )
        ):
            raise ValueError(
                "validation_status=PASSED is inconsistent with a failed or unrun sample"
            )

        solvent_schema: tuple[tuple[str, str, str], ...] | None = None
        for sample in self.samples:
            if sample.solvent_state_summary is None:
                continue
            schema = tuple(
                (
                    item.quantity_name,
                    item.si_unit_or_one,
                    item.description,
                )
                for item in sample.solvent_state_summary
            )
            if solvent_schema is None:
                solvent_schema = schema
            elif schema != solvent_schema:
                raise ValueError(
                    "solvent_state_summary schema must be consistent across the sweep"
                )

    def to_pair_force_table(self) -> PairForceTable:
        """Convert the whole validated sweep without dropping or filling samples."""

        if len(self.samples) < 4:
            raise ValueError("at least four converged contact samples are required")
        failed_indices = [
            index
            for index, sample in enumerate(self.samples)
            if sample.solver_status is not ContactSolveStatus.CONVERGED
        ]
        if failed_indices:
            raise ValueError(
                "contact sweep contains non-converged samples at indices "
                + ", ".join(str(index) for index in failed_indices)
            )
        if self.metadata.validation_status is not PairDataValidationStatus.PASSED:
            raise ValueError("contact sweep requires validation_status=PASSED")
        if self.metadata.time_scale.status is not TimeScaleStatus.SATISFIED:
            raise ValueError("contact sweep requires verified tau_gel << tau_swarm")
        distance = np.fromiter(
            (sample.center_distance_m for sample in self.samples),
            dtype=np.float64,
            count=len(self.samples),
        )
        force = np.fromiter(
            (float(sample.radial_reaction_force_newton) for sample in self.samples),
            dtype=np.float64,
            count=len(self.samples),
        )
        metadata = PairForceMetadata(
            dataset_id=self.metadata.dataset_id,
            source=self.metadata.source,
            physical_status=self.metadata.physical_status,
            solver_status=(
                f"ALL_SAMPLES_CONVERGED:{self.metadata.solver_name}@"
                f"{self.metadata.solver_version}"
            ),
            validation_status=self.metadata.validation_status,
            time_scale_status=self.metadata.time_scale.status,
            scaling=PairForceScaling.UNSCALED_SINGLE_PAIR,
            reference_distance_m=self.metadata.reference_distance_m,
            reference_force_tolerance_newton=(
                self.metadata.reference_force_tolerance_newton
            ),
        )
        return PairForceTable(distance, force, metadata)


def pair_force_table_from_contact_sweep(
    sweep: PairContactSweep,
) -> PairForceTable:
    """Explicit lossless conversion gate from contact records to a force table."""

    if not isinstance(sweep, PairContactSweep):
        raise TypeError("sweep must be PairContactSweep")
    return sweep.to_pair_force_table()
