"""Fail-closed source and 3-D-to-2-D closure gate for magnetic MV validation.

The current continuum grid has area elements ``dx dy``.  Its physically valid
magnetic state must therefore be a depth-integrated number density
``rho_2D`` [m^-2], with ``integral rho_2D dA = N``.  For a source-backed
uniform slab of thickness ``h`` [m],

``rho_3D(R,z) = rho_2D(R)/h`` [m^-3],

and the only compatible planar mean-field energy is

``W_2D(R) = h^-2 int int W_3D(R,z-z') dz dz'`` [J],
``(W_2D * rho_2D)(R) = int W_2D(R-R') rho_2D(R') dA'`` [J].

Thus ``V_mag`` [J] and ``W_2D*rho_2D`` [J] can enter the same chemical
potential.  A raw 3-D ``1/r^3`` function convolved with ``rho_2D`` is not
dimensionally closed and is rejected.  No source-backed slab thickness,
vertical distribution, contact rule, or zero-separation cell closure means no
physical 2-D simulation is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CURRENT_2D_MV_PHYSICAL_REDUCTION_INVALID = "CURRENT_2D_MV_PHYSICAL_REDUCTION_INVALID"
SOURCE_BACKED_2D_REDUCTION_READY = "SOURCE_BACKED_2D_REDUCTION_READY_NO_CONTINUUM_RUN"
SOURCE_PROVENANCE_RELATIVE_PATH = Path(
    "data/external/magnetic_particle/generic__source_provenance__r1.json"
)


class Magnetic2DClosureKind(str, Enum):
    """Only the explicitly stated reduction supported by the present 2-D grid."""

    DEPTH_INTEGRATED_UNIFORM_SLAB = "DEPTH_INTEGRATED_UNIFORM_SLAB"


class Current2DMVPhysicalReductionInvalid(RuntimeError):
    """Raised before any magnetic continuum solve when its SI closure is absent."""


@dataclass(frozen=True, slots=True)
class UniformThinSlabClosure:
    """Source-backed uniform-depth closure for an area-density 2-D solver."""

    thickness_m: float
    source_locator: str
    vertical_distribution: str
    contact_zero_separation_closure: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.thickness_m) or self.thickness_m <= 0.0:
            raise ValueError("slab thickness must be positive and finite in m")
        for value, name in (
            (self.source_locator, "source_locator"),
            (self.vertical_distribution, "vertical_distribution"),
            (self.contact_zero_separation_closure, "contact_zero_separation_closure"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty source-backed string")
        if self.vertical_distribution != "UNIFORM":
            raise ValueError("present 2-D closure supports only a source-declared UNIFORM vertical distribution")

    def as_jsonable(self) -> dict[str, object]:
        return {
            "kind": Magnetic2DClosureKind.DEPTH_INTEGRATED_UNIFORM_SLAB.value,
            "slab_thickness_m": self.thickness_m,
            "vertical_distribution": self.vertical_distribution,
            "rho_2D_unit": "m^-2",
            "rho_3D_relation": "rho_3D(R,z)=rho_2D(R)/h for a uniform slab [m^-3]",
            "W_2D_unit": "J",
            "W_2D_formula": "h^-2 integral integral W_3D(R,z-z') dz dz'",
            "convolution_unit": "(W_2D*rho_2D) [J]",
            "V_mag_unit": "J",
            "source_locator": self.source_locator,
            "contact_zero_separation_closure": self.contact_zero_separation_closure,
        }


@dataclass(frozen=True, slots=True)
class MagneticValidationSource:
    """Parsed source values; fields remain source-labelled rather than calibrated."""

    source_path: Path
    source_id: str
    provenance_class: str
    closure: UniformThinSlabClosure
    particle_diameter_m: float
    particle_volume_m3: float
    contact_diameter_m: float
    chi_v_dimensionless: float | None
    nonlinear_magnetization_table: tuple[tuple[float, float], ...] | None
    field_orientation_unit_vector_3d: tuple[float, float, float]
    diffusivity_m2_per_s: float
    mobility_m_per_newton_second: float
    phase_c_cases: tuple[Mapping[str, object], ...]


@dataclass(frozen=True, slots=True)
class Magnetic2DClosureAdmission:
    """Audit result that permits or blocks the magnetic continuum path."""

    status: str
    source_path: Path
    physical_simulation_allowed: bool
    required_inputs: tuple[str, ...]
    missing_or_invalid: tuple[str, ...]
    unit_audit: Mapping[str, str]
    source: MagneticValidationSource | None = None

    def require_runnable(self) -> MagneticValidationSource:
        if not self.physical_simulation_allowed or self.source is None:
            detail = "; ".join(self.missing_or_invalid) or "source-backed 2-D closure unavailable"
            raise Current2DMVPhysicalReductionInvalid(f"{CURRENT_2D_MV_PHYSICAL_REDUCTION_INVALID}: {detail}")
        return self.source

    def as_jsonable(self) -> dict[str, object]:
        return {
            "status": self.status,
            "source_path": str(self.source_path),
            "physical_simulation_allowed": self.physical_simulation_allowed,
            "required_inputs": list(self.required_inputs),
            "missing_or_invalid": list(self.missing_or_invalid),
            "unit_audit": dict(self.unit_audit),
            "closure": None if self.source is None else self.source.closure.as_jsonable(),
        }


def default_source_provenance_path() -> Path:
    """Return the sole Experiment-owned provenance location without creating it."""

    return Path(__file__).resolve().parents[4] / SOURCE_PROVENANCE_RELATIVE_PATH


def assess_magnetic_2d_closure(
    source_path: Path | None = None,
) -> Magnetic2DClosureAdmission:
    """Load only the designated provenance file and fail closed on any gap."""

    path = default_source_provenance_path() if source_path is None else Path(source_path)
    if not path.is_file():
        return _blocked(path, ("generic__source_provenance__r1.json is absent",))
    try:
        payload = _load_json_object(path)
        audit_blockers = _source_audit_blockers(payload)
        if audit_blockers:
            return _blocked(path, audit_blockers)
        source = _parse_source(path, payload)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        return _blocked(path, (str(error),))
    return Magnetic2DClosureAdmission(
        status=SOURCE_BACKED_2D_REDUCTION_READY,
        source_path=path,
        physical_simulation_allowed=True,
        required_inputs=_required_inputs(),
        missing_or_invalid=(),
        unit_audit=_unit_audit(),
        source=source,
    )


def _blocked(path: Path, issues: tuple[str, ...]) -> Magnetic2DClosureAdmission:
    return Magnetic2DClosureAdmission(
        status=CURRENT_2D_MV_PHYSICAL_REDUCTION_INVALID,
        source_path=path,
        physical_simulation_allowed=False,
        required_inputs=_required_inputs(),
        missing_or_invalid=issues,
        unit_audit=_unit_audit(),
        source=None,
    )


def _required_inputs() -> tuple[str, ...]:
    return (
        "source_id and non-empty provenance_class",
        "closure.kind=DEPTH_INTEGRATED_UNIFORM_SLAB",
        "closure.slab_thickness_m [m] with source_locator",
        "closure.vertical_distribution=UNIFORM with source_locator",
        "closure.contact_zero_separation_closure with source locator",
        "particle.diameter_m [m], volume_m3 [m^3], contact_diameter_m [m]",
        "particle either chi_v_dimensionless or sourced nonlinear magnetization table",
        "particle and contact source locators",
        "field.orientation_unit_vector_3d and source locator",
        "field source-backed B(x) and grad(B)(x), or a condition-specific equivalent with uncertainty",
        "transport.diffusivity_m2_per_s [m^2/s] and mobility_m_per_newton_second [m/(N s)]",
        "phase_c_cases carrying source-backed 10/40 nm, 1/2/4 mg/mL, and 40 mT conditions",
    )


def _unit_audit() -> dict[str, str]:
    return {
        "rho_2D": "number density [m^-2]; integral rho_2D dA = N",
        "rho_3D": "rho_2D/h [m^-3] for a source-backed uniform slab",
        "W_3D": "dipolar pair energy [J]",
        "W_2D": "vertical average of W_3D [J], not raw 1/r^3",
        "W_2D_convolution": "integral W_2D(R-R') rho_2D(R') dA' [J]",
        "V_mag": "-chi_v*V_m*B^2/(2*mu0) [J]",
        "F_mag": "chi_v*V_m*B*grad(B)/mu0 [N]",
    }


def _source_audit_blockers(payload: Mapping[str, object]) -> tuple[str, ...]:
    """Translate the Experiment-owned audit's explicit unresolved states into gates.

    Values marked as rejected transfer candidates remain unavailable.  This
    function never searches ``scalar_records`` for a convenient value because
    doing so would silently cross the provenance restrictions in the frozen
    source register.
    """

    issues: list[str] = []
    physical_status = payload.get("physical_status")
    if not isinstance(physical_status, str) or not physical_status.strip():
        issues.append("source physical_status is absent")
    elif "MISSING" in physical_status.upper() or "BLOCKED" in physical_status.upper():
        issues.append("source physical_status itself declares required same-system inputs missing")
    closure = payload.get("closure")
    if not isinstance(closure, Mapping):
        return ("closure object is absent",)
    if closure.get("kind") != Magnetic2DClosureKind.DEPTH_INTEGRATED_UNIFORM_SLAB.value:
        issues.append("current solver lacks a source-backed DEPTH_INTEGRATED_UNIFORM_SLAB closure")
    if not _is_positive_number(closure.get("slab_thickness_m")):
        issues.append("source-backed slab_thickness_m [m] is missing")
    if closure.get("vertical_distribution") != "UNIFORM":
        issues.append("source-backed uniform vertical distribution is missing")
    if _is_unresolved_text(closure.get("contact_zero_separation_closure")):
        issues.append("source-backed zero-separation/contact cell closure is missing")

    particle = payload.get("particle")
    if not isinstance(particle, Mapping):
        return tuple(issues + ["particle object is absent"])
    if not _is_positive_number(particle.get("volume_m3")):
        issues.append("same-system magnetic material volume [m^3] is missing")
    if not _is_positive_number(particle.get("contact_diameter_m")):
        issues.append("source-backed physical coated contact diameter [m] is missing")
    if not _is_positive_number(particle.get("chi_v_dimensionless")) and not _has_nonempty_table(particle.get("nonlinear_magnetization_table")):
        issues.append("same-system susceptibility or nonlinear moment law is missing")

    field = payload.get("field")
    if not isinstance(field, Mapping):
        return tuple(issues + ["field object is absent"])
    try:
        _unit_vector3(field.get("orientation_unit_vector_3d"), "field.orientation_unit_vector_3d")
    except (TypeError, ValueError):
        issues.append("source-backed magnetic field orientation is missing")
    # The source record intentionally names its orientation as a placeholder;
    # even a direction cannot supply the B*grad(B) force input by itself.
    if not _is_source_backed_field_force(field):
        issues.append("same-system B(x) and grad(B)(x), or B*grad(B), with uncertainty are missing")

    transport = payload.get("transport")
    if not isinstance(transport, Mapping):
        return tuple(issues + ["transport object is absent"])
    if not _is_positive_number(transport.get("diffusivity_m2_per_s")):
        issues.append("same-system in-gel diffusivity D [m^2/s] is missing")
    if not _is_positive_number(transport.get("mobility_m_per_newton_second")):
        issues.append("same-system in-gel mobility M [m/(N s)] is missing")

    cases = payload.get("phase_c_cases")
    if not isinstance(cases, list) or not cases:
        issues.append("Phase C cases are absent")
    elif any(not isinstance(case, Mapping) or case.get("status") != "SOURCE_BACKED_NUMBER_DENSITY_READY" for case in cases):
        issues.append("Phase C mass concentrations lack source-backed particle-number-density conversion")
    return tuple(issues)


def _is_positive_number(value: object) -> bool:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(result) and result > 0.0


def _has_nonempty_table(value: object) -> bool:
    return isinstance(value, list) and len(value) >= 2


def _is_unresolved_text(value: object) -> bool:
    return not isinstance(value, str) or not value.strip() or "UNRESOLVED" in value.upper() or "NOT_AVAILABLE" in value.upper()


def _is_source_backed_field_force(field: Mapping[str, object]) -> bool:
    """Accept either a sourced field-map descriptor or condition-specific force tuple."""

    locator = field.get("source_locator")
    if _is_unresolved_text(locator) or "PLACEHOLDER" in str(locator).upper():
        return False
    has_map = isinstance(field.get("spatial_field_map"), Mapping)
    has_tuple = _is_positive_number(field.get("flux_density_tesla")) and _is_positive_number(
        field.get("flux_density_gradient_tesla_per_m")
    )
    return has_map or has_tuple


def _load_json_object(path: Path) -> Mapping[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} is forbidden")

    def reject_duplicate(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} is forbidden")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=reject_constant, object_pairs_hook=reject_duplicate)
    if not isinstance(payload, Mapping):
        raise ValueError("source provenance must be a JSON object")
    return payload


def _parse_source(path: Path, payload: Mapping[str, object]) -> MagneticValidationSource:
    _require_keys(payload, {"source_id", "provenance_class", "closure", "particle", "field", "transport", "phase_c_cases"}, "root")
    source_id = _nonempty(payload["source_id"], "source_id")
    provenance = _nonempty(payload["provenance_class"], "provenance_class")
    closure_payload = _mapping(payload["closure"], "closure")
    _require_keys(
        closure_payload,
        {"kind", "slab_thickness_m", "vertical_distribution", "source_locator", "contact_zero_separation_closure"},
        "closure",
    )
    if closure_payload["kind"] != Magnetic2DClosureKind.DEPTH_INTEGRATED_UNIFORM_SLAB.value:
        raise ValueError("closure.kind must be DEPTH_INTEGRATED_UNIFORM_SLAB for the present 2-D solver")
    closure = UniformThinSlabClosure(
        _positive(closure_payload["slab_thickness_m"], "closure.slab_thickness_m"),
        _nonempty(closure_payload["source_locator"], "closure.source_locator"),
        _nonempty(closure_payload["vertical_distribution"], "closure.vertical_distribution"),
        _nonempty(closure_payload["contact_zero_separation_closure"], "closure.contact_zero_separation_closure"),
    )
    particle = _mapping(payload["particle"], "particle")
    _require_keys(
        particle,
        {"diameter_m", "volume_m3", "contact_diameter_m", "source_locator"},
        "particle",
    )
    chi = particle.get("chi_v_dimensionless")
    table = particle.get("nonlinear_magnetization_table")
    if (chi is None) == (table is None):
        raise ValueError("particle requires exactly one of chi_v_dimensionless or nonlinear_magnetization_table")
    chi_value = None if chi is None else _positive(chi, "particle.chi_v_dimensionless")
    table_value = None if table is None else _parse_magnetization_table(table)
    field = _mapping(payload["field"], "field")
    _require_keys(field, {"orientation_unit_vector_3d", "source_locator"}, "field")
    orientation = _unit_vector3(field["orientation_unit_vector_3d"], "field.orientation_unit_vector_3d")
    transport = _mapping(payload["transport"], "transport")
    _require_keys(transport, {"diffusivity_m2_per_s", "mobility_m_per_newton_second", "source_locator"}, "transport")
    cases = payload["phase_c_cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("phase_c_cases must be a non-empty source-backed list")
    if not all(isinstance(case, Mapping) for case in cases):
        raise ValueError("phase_c_cases must contain objects")
    return MagneticValidationSource(
        source_path=path,
        source_id=source_id,
        provenance_class=provenance,
        closure=closure,
        particle_diameter_m=_positive(particle["diameter_m"], "particle.diameter_m"),
        particle_volume_m3=_positive(particle["volume_m3"], "particle.volume_m3"),
        contact_diameter_m=_positive(particle["contact_diameter_m"], "particle.contact_diameter_m"),
        chi_v_dimensionless=chi_value,
        nonlinear_magnetization_table=table_value,
        field_orientation_unit_vector_3d=orientation,
        diffusivity_m2_per_s=_positive(transport["diffusivity_m2_per_s"], "transport.diffusivity_m2_per_s"),
        mobility_m_per_newton_second=_positive(transport["mobility_m_per_newton_second"], "transport.mobility_m_per_newton_second"),
        phase_c_cases=tuple(cases),
    )


def _parse_magnetization_table(value: object) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError("nonlinear_magnetization_table must be a list of at least two [B_T, M_A_per_m] rows")
    rows: list[tuple[float, float]] = []
    for index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(f"nonlinear_magnetization_table[{index}] must be [B_T, M_A_per_m]")
        rows.append((_nonnegative(row[0], f"nonlinear_magnetization_table[{index}][0]"), _nonnegative(row[1], f"nonlinear_magnetization_table[{index}][1]")))
    if rows[0][0] != 0.0 or any(right[0] <= left[0] for left, right in zip(rows, rows[1:], strict=True)):
        raise ValueError("nonlinear magnetization B samples must start at 0 and strictly increase")
    return tuple(rows)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _require_keys(mapping: Mapping[str, object], keys: set[str], name: str) -> None:
    missing = sorted(keys - set(mapping))
    if missing:
        raise ValueError(f"{name} is missing required fields: {', '.join(missing)}")


def _positive(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _nonnegative(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty source-backed string")
    return value


def _unit_vector3(value: object, name: str) -> tuple[float, float, float]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite length-three vector")
    if not math.isclose(float(np.linalg.norm(vector)), 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError(f"{name} must have unit norm")
    return (float(vector[0]), float(vector[1]), float(vector[2]))
