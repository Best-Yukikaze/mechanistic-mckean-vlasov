"""Shared strict I/O and provenance helpers for Phase 6 validation scripts.

This module deliberately lives in ``scripts``: it validates experiment
artifacts but does not alter any mechanics or continuum equation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import scipy

from mechanistic_mv.mechanics.hydrogel import (
    HydrogelParameters,
    TEST_ONLY_NOT_CALIBRATED,
)
from mechanistic_mv.mechanics.hydrogel.equilibrium import TimeScaleStatus
from mechanistic_mv.mechanics.pair_interaction import (
    HydrogelEffectivePairPotential,
    PairDataValidationStatus,
    PairForceMetadata,
    PairForceScaling,
    PairForceTable,
    PairScalingConversionEvidence,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIRECTORY = PROJECT_ROOT / "outputs" / "validation"
PAIR_VALIDATION_DIRECTORY = VALIDATION_DIRECTORY / "pair_interaction"

PHASE6_SCHEMA_VERSION = 1
FORCE_DERIVATIVE_ABSOLUTE_TOLERANCE_N = 1.0e-20
FORCE_DERIVATIVE_RELATIVE_TOLERANCE = 2.0e-7
FORCE_DERIVATIVE_STEP_FRACTION = 1.0e-4
CONTINUUM_READY_CLASSIFICATION = "CONTINUUM_READY"
PARTICLE_ONLY_SHORT_RANGE_UNRESOLVED = (
    "PARTICLE_ONLY_SHORT_RANGE_UNRESOLVED"
)
_VAGUE_TEXT = {"n/a", "na", "none", "tbd", "todo", "unknown", "unspecified"}
_NONPHYSICAL_LABEL_TOKENS = (
    "test_only",
    "placeholder",
    "not_calibrated",
    "unverified",
    "not_available",
    "failed",
    "pending",
    "unknown",
    "tbd",
    "todo",
    "unspecified",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={PROJECT_ROOT.as_posix()}",
                "rev-parse",
                "HEAD",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={PROJECT_ROOT.as_posix()}",
                "status",
                "--porcelain",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def runtime_metadata() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def provenance(command: list[str] | None = None) -> dict[str, object]:
    return {
        "generated_at_utc": utc_now_iso(),
        "git_revision_at_run": git_revision(),
        "git_dirty_at_run": git_dirty(),
        "command": list(sys.argv if command is None else command),
        "runtime": runtime_metadata(),
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json_object(path: Path) -> dict[str, Any]:
    def reject_nonfinite_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value!r} is forbidden")

    def reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} is forbidden")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_nonfinite_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except FileNotFoundError as error:
        raise ValueError(f"metadata file does not exist: {path}") from error
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"metadata is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError("metadata root must be a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def short_range_admission(minimum_supported_distance_m: float) -> dict[str, object]:
    """Classify measured table support without filling or extrapolating samples."""

    minimum = float(minimum_supported_distance_m)
    if not np.isfinite(minimum) or minimum < 0.0:
        raise ValueError(
            "minimum_supported_distance_m must be finite and non-negative"
        )
    continuum_ready = minimum == 0.0
    classification = (
        CONTINUUM_READY_CLASSIFICATION
        if continuum_ready
        else PARTICLE_ONLY_SHORT_RANGE_UNRESOLVED
    )
    return {
        "minimum_supported_distance_m": minimum,
        "continuum_ready": continuum_ready,
        "short_range_classification": classification,
    }


def _metadata_with_short_range_admission(
    metadata: Mapping[str, Any], minimum_supported_distance_m: float
) -> dict[str, Any]:
    admission = short_range_admission(minimum_supported_distance_m)
    result = dict(metadata)
    for name, expected in admission.items():
        if name in result:
            supplied = result[name]
            if name == "continuum_ready" and not isinstance(supplied, bool):
                raise ValueError("metadata continuum_ready must be boolean")
            if name == "minimum_supported_distance_m" and isinstance(
                supplied, bool
            ):
                raise ValueError(
                    "metadata minimum_supported_distance_m must be numeric"
                )
            if name == "short_range_classification" and not isinstance(
                supplied, str
            ):
                raise ValueError(
                    "metadata short_range_classification must be a string"
                )
            if supplied != expected:
                raise ValueError(
                    f"metadata field {name!r} contradicts the force-table r_min"
                )
        result[name] = expected
    derived_usage = {
        "continuum_admission_status": (
            "PASSED" if admission["continuum_ready"] else "BLOCKED"
        ),
        "usage_scope": (
            "PARTICLES_AND_CONTINUUM"
            if admission["continuum_ready"]
            else "PARTICLES_ONLY"
        ),
    }
    for name, expected in derived_usage.items():
        if name in result:
            supplied = result[name]
            if not isinstance(supplied, str):
                raise ValueError(f"metadata field {name!r} must be a string")
            if supplied != expected:
                raise ValueError(
                    f"metadata field {name!r} contradicts the force-table r_min"
                )
    return result


def test_only_hydrogel_parameters() -> HydrogelParameters:
    return HydrogelParameters(
        network_density_times_solvent_volume=0.025,
        flory_huggins_chi=0.31,
        initial_polymer_volume_fraction=0.18,
        delta_chemical_potential_over_kbt=0.12,
        thermal_energy_density_pa=2.4e5,
        calibration_status=TEST_ONLY_NOT_CALIBRATED,
    )


def test_only_pair_force_table(
    *, minimum_distance_m: float = 0.0
) -> PairForceTable:
    reference_distance_m = 4.0e-6
    minimum = float(minimum_distance_m)
    if (
        not np.isfinite(minimum)
        or minimum < 0.0
        or minimum >= reference_distance_m
    ):
        raise ValueError(
            "test-only minimum_distance_m must lie in [0, reference_distance_m)"
        )
    distance = np.linspace(minimum, reference_distance_m, 17)
    normalized = distance / reference_distance_m
    force = 2.0e-12 * normalized * (1.0 - normalized) ** 2
    metadata = PairForceMetadata(
        dataset_id="TEST_ONLY_NOT_CALIBRATED_phase6_smooth_radial_force",
        source="analytic Phase 6 regression fixture; not contact or physical data",
        physical_status=TEST_ONLY_NOT_CALIBRATED,
        solver_status="TEST_ONLY_NOT_A_CONTACT_SOLVER",
        validation_status=PairDataValidationStatus.PASSED,
        time_scale_status=TimeScaleStatus.UNVERIFIED,
        scaling=PairForceScaling.KAC_NORMALIZED_PROBABILITY,
        reference_distance_m=reference_distance_m,
        reference_force_tolerance_newton=0.0,
    )
    return PairForceTable(distance, force, metadata)


def test_only_scaling_provenance() -> dict[str, object]:
    return {
        "source_semantics": "TEST_ONLY_DIRECT_KAC_FIXTURE",
        "target_semantics": PairForceScaling.KAC_NORMALIZED_PROBABILITY.value,
        "formula": (
            "fixture is defined directly for unit-mass rho and the 1/N particle "
            "prefactor; no physical single-pair conversion was performed"
        ),
        "population_or_concentration_value": 1.0,
        "population_or_concentration_unit": "1 (test-only normalization marker)",
        "force_multiplier": 1.0,
        "energy_multiplier": 1.0,
        "source": "analytic Phase 6 regression fixture",
        "calibration_id": TEST_ONLY_NOT_CALIBRATED,
    }


def test_only_short_range_provenance(
    *, minimum_distance_m: float = 0.0
) -> dict[str, object]:
    admission = short_range_admission(minimum_distance_m)
    method = (
        "TEST_ONLY_ANALYTIC_DEFINITION_COVERS_R_EQUALS_ZERO"
        if admission["continuum_ready"]
        else "TEST_ONLY_SHORT_RANGE_UNRESOLVED_BELOW_TABLE_MINIMUM"
    )
    return {
        "method": method,
        "source": "analytic Phase 6 regression fixture",
        "calibration_id": TEST_ONLY_NOT_CALIBRATED,
        "minimum_distance_m": admission["minimum_supported_distance_m"],
    }


def pair_force_metadata_payload(
    table: PairForceTable,
    *,
    data_file_sha256: str,
    scaling_provenance: Mapping[str, object],
    short_range_closure_provenance: Mapping[str, object] | None,
    workflow_status: str,
    calibration_status: str,
    calibration_id: str | None = None,
    physical_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    metadata = table.metadata
    admission = short_range_admission(float(table.center_distance_m[0]))
    payload: dict[str, object] = {
        "schema_name": "mechanistic_mv.pair_force_table",
        "schema_version": PHASE6_SCHEMA_VERSION,
        "artifact_type": "PAIR_FORCE_TABLE",
        "artifact_id": metadata.dataset_id,
        "workflow_status": workflow_status,
        "physical_status": metadata.physical_status,
        "calibration_status": calibration_status,
        "validation_status": metadata.validation_status.value,
        "time_scale_status": metadata.time_scale_status.value,
        "solver_status": metadata.solver_status,
        "source": metadata.source,
        "scaling": metadata.scaling.value,
        "native_scaling": metadata.native_scaling.value,
        "reference_distance_m": metadata.reference_distance_m,
        "reference_force_tolerance_newton": (
            metadata.reference_force_tolerance_newton
        ),
        "data_file_sha256": data_file_sha256,
        "columns": {
            "center_distance_m": {"unit": "m"},
            "radial_force_newton": {"unit": "N"},
        },
        "scaling_provenance": dict(scaling_provenance),
        **admission,
    }
    if short_range_closure_provenance is not None:
        payload["short_range_closure_provenance"] = dict(
            short_range_closure_provenance
        )
    if metadata.scaling_conversion is not None:
        conversion = metadata.scaling_conversion
        payload["scaling_conversion"] = {
            "source_dataset_id": conversion.source_dataset_id,
            "population_count": conversion.population_count,
            "population_count_provenance": (
                conversion.population_count_provenance
            ),
            "source_scaling": conversion.source_scaling.value,
            "target_scaling": conversion.target_scaling.value,
        }
    if physical_provenance is not None:
        payload["physical_provenance"] = dict(physical_provenance)
    if calibration_id is not None:
        payload["calibration_id"] = calibration_id
    return payload


def write_pair_force_csv(path: Path, table: PairForceTable) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("center_distance_m", "radial_force_newton"))
        writer.writerows(
            zip(
                table.center_distance_m.tolist(),
                table.radial_force_newton.tolist(),
                strict=True,
            )
        )
    temporary.replace(path)


def _require_text(mapping: Mapping[str, Any], name: str) -> str:
    value = mapping.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"metadata field {name!r} must be a non-empty string")
    cleaned = value.strip()
    if cleaned.casefold() in _VAGUE_TEXT:
        raise ValueError(f"metadata field {name!r} must be specific")
    return cleaned


def _require_finite_number(
    mapping: Mapping[str, Any], name: str, *, positive: bool = False
) -> float:
    value = mapping.get(name)
    if isinstance(value, bool):
        raise ValueError(f"metadata field {name!r} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"metadata field {name!r} must be numeric") from error
    if not np.isfinite(number) or (positive and number <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"metadata field {name!r} must be {qualifier}")
    return number


def _require_sha256(mapping: Mapping[str, Any], name: str) -> str:
    value = _require_text(mapping, name)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"metadata field {name!r} must be a lowercase SHA256")
    return value


def _require_physical_artifact_labels(
    physical_status: str, calibration_status: str
) -> None:
    for name, value in (
        ("physical_status", physical_status),
        ("calibration_status", calibration_status),
    ):
        normalized = value.casefold().replace("-", "_").replace(" ", "_")
        if any(token in normalized for token in _NONPHYSICAL_LABEL_TOKENS):
            raise ValueError(f"physical metadata field {name!r} is nonphysical")
    physical_label_is_allowed = (
        physical_status == "FULL_PHYSICAL_FEM"
        or (
            physical_status.startswith("CALIBRATED_PHYSICAL_")
            and len(physical_status) > len("CALIBRATED_PHYSICAL_")
        )
    )
    calibration_label_is_allowed = (
        calibration_status.startswith("CALIBRATED_")
        and len(calibration_status) > len("CALIBRATED_")
    )
    if not physical_label_is_allowed or not calibration_label_is_allowed:
        raise ValueError(
            "physical artifacts require physical_status=FULL_PHYSICAL_FEM or "
            "CALIBRATED_PHYSICAL_* and calibration_status=CALIBRATED_*"
        )


def _require_physical_text(mapping: Mapping[str, Any], name: str) -> str:
    value = _require_text(mapping, name)
    normalized = value.casefold().replace("-", "_").replace(" ", "_")
    if any(token in normalized for token in _NONPHYSICAL_LABEL_TOKENS):
        raise ValueError(
            f"physical metadata field {name!r} contains a nonphysical label"
        )
    return value


def _require_complete_scaling_provenance(
    metadata: Mapping[str, Any],
    scaling: PairForceScaling,
    *,
    physical_status: str,
    calibration_id: str,
    current_force_table_sha256: str,
) -> dict[str, object]:
    value = metadata.get("scaling_provenance")
    if not isinstance(value, dict):
        raise ValueError("complete scaling_provenance metadata is required")
    for field in (
        "source_semantics",
        "target_semantics",
        "formula",
        "population_or_concentration_unit",
        "source",
        "calibration_id",
    ):
        _require_text(value, field)
    if value["target_semantics"] != scaling.value:
        raise ValueError("scaling_provenance target_semantics does not match scaling")
    _require_finite_number(value, "population_or_concentration_value", positive=True)
    force_multiplier = _require_finite_number(value, "force_multiplier", positive=True)
    energy_multiplier = _require_finite_number(
        value, "energy_multiplier", positive=True
    )
    if not np.isclose(
        force_multiplier,
        energy_multiplier,
        rtol=16.0 * np.finfo(np.float64).eps,
        atol=0.0,
    ):
        raise ValueError("force and energy Kac multipliers must be identical")
    if physical_status != TEST_ONLY_NOT_CALIBRATED:
        if value["source_semantics"] != PairForceScaling.UNSCALED_SINGLE_PAIR.value:
            raise ValueError(
                "physical Kac provenance source_semantics must identify the "
                "UNSCALED_SINGLE_PAIR source"
            )
        if value.get("scaling_applied_before_ingestion") is not True:
            raise ValueError(
                "physical Kac scaling must be explicitly applied before ingestion"
            )
        _require_sha256(value, "source_unscaled_artifact_sha256")
        scaled_hash = _require_sha256(value, "scaled_force_table_sha256")
        if scaled_hash != current_force_table_sha256:
            raise ValueError(
                "scaled_force_table_sha256 must match the ingested force CSV"
            )
        if value["calibration_id"] != calibration_id:
            raise ValueError(
                "scaling_provenance calibration_id must match artifact calibration_id"
            )
        _require_physical_text(value, "source")
        _require_physical_text(value, "applied_by")
    return dict(value)


def _require_short_range_provenance(
    metadata: Mapping[str, Any],
    *,
    physical_status: str,
    calibration_id: str,
    minimum_supported_distance_m: float,
) -> dict[str, object] | None:
    admission = short_range_admission(minimum_supported_distance_m)
    value = metadata.get("short_range_closure_provenance")
    if value is None and not admission["continuum_ready"]:
        return None
    if not isinstance(value, dict):
        raise ValueError(
            "short_range_closure_provenance is required for a continuum-ready "
            "table that includes r=0"
        )
    value = _metadata_with_short_range_admission(
        value, minimum_supported_distance_m
    )
    for field in ("method", "source", "calibration_id"):
        _require_text(value, field)
    if physical_status != TEST_ONLY_NOT_CALIBRATED:
        if value["calibration_id"] != calibration_id:
            raise ValueError(
                "short-range calibration_id must match artifact calibration_id"
            )
        _require_physical_text(value, "method")
        _require_physical_text(value, "source")
        if admission["continuum_ready"]:
            if (
                value.get("validation_status")
                != PairDataValidationStatus.PASSED.value
            ):
                raise ValueError(
                    "physical short-range closure requires "
                    "validation_status=PASSED"
                )
            if _require_finite_number(value, "minimum_distance_m") != 0.0:
                raise ValueError("continuum-ready short-range support must cover r=0")
            _require_sha256(value, "source_artifact_sha256")
        else:
            declared_minimum = value.get("minimum_distance_m")
            if declared_minimum is not None:
                declared = _require_finite_number(value, "minimum_distance_m")
                if declared < 0.0 or declared != minimum_supported_distance_m:
                    raise ValueError(
                        "particle-only short-range provenance minimum_distance_m "
                        "must match the force-table r_min"
                    )
            if value.get("continuum_ready") is True:
                raise ValueError(
                    "r_min>0 short-range provenance cannot claim continuum_ready"
                )
            if (
                value.get("short_range_classification")
                == CONTINUUM_READY_CLASSIFICATION
            ):
                raise ValueError(
                    "r_min>0 short-range provenance cannot claim CONTINUUM_READY"
                )
    return dict(value)


def _pair_scaling_contract_from_metadata(
    metadata: Mapping[str, Any],
    *,
    scaling: PairForceScaling,
    dataset_id: str,
    physical_status: str,
    scaling_provenance: Mapping[str, object],
) -> tuple[PairForceScaling, PairScalingConversionEvidence | None]:
    raw_native = metadata.get("native_scaling")
    raw_conversion = metadata.get("scaling_conversion")
    if raw_native is None and raw_conversion is None:
        if physical_status != TEST_ONLY_NOT_CALIBRATED:
            raise ValueError(
                "physical Kac data require native_scaling and explicit "
                "population scaling_conversion evidence"
            )
        return scaling, None
    try:
        native = PairForceScaling(_require_text(metadata, "native_scaling"))
    except ValueError as error:
        raise ValueError("metadata native_scaling is unsupported") from error
    if raw_conversion is None:
        if physical_status != TEST_ONLY_NOT_CALIBRATED:
            raise ValueError(
                "physical Kac data require explicit population "
                "scaling_conversion evidence"
            )
        if native is not scaling:
            raise ValueError(
                "native_scaling differs from scaling without conversion evidence"
            )
        return native, None
    if not isinstance(raw_conversion, dict):
        raise ValueError("scaling_conversion must be an object")
    source_dataset_id = _require_text(raw_conversion, "source_dataset_id")
    if source_dataset_id != dataset_id:
        raise ValueError(
            "scaling_conversion source_dataset_id must match artifact_id"
        )
    population = _require_finite_number(
        raw_conversion, "population_count", positive=True
    )
    if not population.is_integer():
        raise ValueError("scaling_conversion population_count must be an integer")
    try:
        source_scaling = PairForceScaling(
            _require_text(raw_conversion, "source_scaling")
        )
        target_scaling = PairForceScaling(
            _require_text(raw_conversion, "target_scaling")
        )
    except ValueError as error:
        raise ValueError("scaling_conversion contains an unsupported enum") from error
    population_provenance = _require_text(
        raw_conversion, "population_count_provenance"
    )
    if physical_status != TEST_ONLY_NOT_CALIBRATED:
        population_provenance = _require_physical_text(
            raw_conversion, "population_count_provenance"
        )
    evidence = PairScalingConversionEvidence(
        source_dataset_id=source_dataset_id,
        population_count=int(population),
        population_count_provenance=population_provenance,
        source_scaling=source_scaling,
        target_scaling=target_scaling,
    )
    if native is not evidence.source_scaling or scaling is not evidence.target_scaling:
        raise ValueError(
            "scaling_conversion enums disagree with native_scaling/scaling"
        )
    if scaling_provenance.get("source_semantics") != evidence.source_scaling.value:
        raise ValueError(
            "scaling_conversion source disagrees with scaling_provenance"
        )
    if scaling_provenance.get("target_semantics") != evidence.target_scaling.value:
        raise ValueError(
            "scaling_conversion target disagrees with scaling_provenance"
        )
    for field, expected in (
        ("population_or_concentration_value", float(evidence.population_count)),
        ("force_multiplier", float(evidence.force_multiplier)),
        ("energy_multiplier", float(evidence.force_multiplier)),
    ):
        actual = _require_finite_number(
            scaling_provenance, field, positive=True
        )
        if not np.isclose(
            actual,
            expected,
            rtol=16.0 * np.finfo(np.float64).eps,
            atol=0.0,
        ):
            raise ValueError(
                f"scaling_conversion {field} disagrees with scaling_provenance"
            )
    if (
        physical_status != TEST_ONLY_NOT_CALIBRATED
        and native is not PairForceScaling.UNSCALED_SINGLE_PAIR
    ):
        raise ValueError(
            "physical Kac conversion must originate from UNSCALED_SINGLE_PAIR"
        )
    return native, evidence


def _require_physical_provenance(
    metadata: Mapping[str, Any],
    *,
    distance: np.ndarray,
    calibration_status: str,
    calibration_id: str,
    time_scale_status: TimeScaleStatus,
    solver_status: str,
) -> dict[str, object]:
    """Validate the minimum Phase 6 physical sidecar, without guessing values."""

    value = metadata.get("physical_provenance")
    if not isinstance(value, dict):
        raise ValueError(
            "physical_provenance is required for a physical pair-force artifact"
        )

    hydrogel = value.get("hydrogel_parameters")
    if not isinstance(hydrogel, dict):
        raise ValueError("physical_provenance.hydrogel_parameters is required")
    parameter_status = _require_text(hydrogel, "calibration_status")
    if parameter_status != calibration_status:
        raise ValueError("Hydrogel and artifact calibration_status fields disagree")
    if _require_physical_text(hydrogel, "calibration_id") != calibration_id:
        raise ValueError("Hydrogel calibration_id must match artifact calibration_id")
    _require_physical_text(hydrogel, "source")
    if hydrogel.get("units") != {
        "network_density_times_solvent_volume": "1",
        "flory_huggins_chi": "1",
        "initial_polymer_volume_fraction": "1",
        "delta_chemical_potential_over_kbt": "1",
        "thermal_energy_density_pa": "Pa",
    }:
        raise ValueError(
            "Hydrogel parameters must declare exact SI/dimensionless units"
        )
    try:
        HydrogelParameters(
            network_density_times_solvent_volume=_require_finite_number(
                hydrogel, "network_density_times_solvent_volume", positive=True
            ),
            flory_huggins_chi=_require_finite_number(
                hydrogel, "flory_huggins_chi"
            ),
            initial_polymer_volume_fraction=_require_finite_number(
                hydrogel, "initial_polymer_volume_fraction", positive=True
            ),
            delta_chemical_potential_over_kbt=_require_finite_number(
                hydrogel, "delta_chemical_potential_over_kbt"
            ),
            thermal_energy_density_pa=_require_finite_number(
                hydrogel, "thermal_energy_density_pa", positive=True
            ),
            calibration_status=parameter_status,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("physical Hydrogel parameter block is invalid") from error

    geometry = value.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("physical_provenance.geometry is required")
    for field in ("model", "source"):
        _require_physical_text(geometry, field)
    if _require_physical_text(geometry, "calibration_id") != calibration_id:
        raise ValueError("geometry calibration_id must match artifact calibration_id")
    _require_finite_number(geometry, "particle_1_undeformed_radius_m", positive=True)
    _require_finite_number(geometry, "particle_2_undeformed_radius_m", positive=True)

    equilibrium = value.get("equilibrium_assumptions")
    if not isinstance(equilibrium, dict):
        raise ValueError("physical_provenance.equilibrium_assumptions is required")
    for field in ("model", "assumptions", "source"):
        _require_physical_text(equilibrium, field)

    solver = value.get("contact_solver")
    if not isinstance(solver, dict):
        raise ValueError("physical_provenance.contact_solver is required")
    for field in (
        "name",
        "version",
        "configuration_id",
        "implementation_revision",
        "method",
    ):
        _require_physical_text(solver, field)
    expected_solver_status = (
        f"ALL_SAMPLES_CONVERGED:{solver['name']}@{solver['version']}"
    )
    if solver_status != expected_solver_status:
        raise ValueError(
            "solver_status must match the converged contact-solver provenance"
        )

    mesh = value.get("mesh_and_resolution")
    if not isinstance(mesh, dict):
        raise ValueError("physical_provenance.mesh_and_resolution is required")
    for field in (
        "method",
        "selected_resolution_id",
    ):
        _require_physical_text(mesh, field)
    _require_sha256(mesh, "convergence_study_artifact_sha256")
    raw_resolution_count = _require_finite_number(
        mesh, "resolution_count", positive=True
    )
    if not raw_resolution_count.is_integer():
        raise ValueError("physical mesh resolution_count must be an integer")
    resolution_count = int(raw_resolution_count)
    if resolution_count < 2:
        raise ValueError("physical mesh provenance requires at least two resolutions")
    if mesh.get("convergence_passed") is not True:
        raise ValueError("physical mesh convergence must be explicitly passed")
    mesh_error = _require_finite_number(mesh, "maximum_relative_change")
    mesh_limit = _require_finite_number(mesh, "upper_limit", positive=True)
    if mesh_error < 0.0 or mesh_error > mesh_limit:
        raise ValueError("physical mesh convergence error exceeds its gate")

    boundary = value.get("boundary_and_contact_conditions")
    if not isinstance(boundary, dict):
        raise ValueError(
            "physical_provenance.boundary_and_contact_conditions is required"
        )
    for field in (
        "mechanical_boundary_conditions",
        "solvent_bath_boundary_conditions",
        "contact_law",
        "reaction_force_sign_convention",
        "source",
    ):
        _require_physical_text(boundary, field)

    time_scale = value.get("time_scale_assessment")
    if not isinstance(time_scale, dict):
        raise ValueError("physical_provenance.time_scale_assessment is required")
    if time_scale_status is not TimeScaleStatus.SATISFIED:
        raise ValueError("physical data require time_scale_status=SATISFIED")
    tau_gel = _require_finite_number(time_scale, "tau_gel_s", positive=True)
    tau_swarm = _require_finite_number(time_scale, "tau_swarm_s", positive=True)
    ratio = _require_finite_number(time_scale, "ratio", positive=True)
    maximum_ratio = _require_finite_number(time_scale, "maximum_ratio", positive=True)
    if not np.isclose(ratio, tau_gel / tau_swarm, rtol=1.0e-12, atol=0.0):
        raise ValueError("time-scale ratio is inconsistent with tau_gel/tau_swarm")
    if ratio > maximum_ratio or time_scale.get("passed") is not True:
        raise ValueError("physical time-scale separation did not pass its gate")
    _require_physical_text(time_scale, "source")

    sweep = value.get("distance_sweep")
    if not isinstance(sweep, dict):
        raise ValueError("physical_provenance.distance_sweep is required")
    raw_sample_count = _require_finite_number(
        sweep, "sample_count", positive=True
    )
    if not raw_sample_count.is_integer():
        raise ValueError("distance-sweep sample_count must be an integer")
    sample_count = int(raw_sample_count)
    if sample_count != int(distance.size):
        raise ValueError("distance-sweep sample_count does not match the CSV")
    minimum_distance = _require_finite_number(sweep, "minimum_distance_m")
    reference_distance = _require_finite_number(
        sweep, "reference_distance_m", positive=True
    )
    if (
        minimum_distance != float(distance[0])
        or reference_distance != float(distance[-1])
    ):
        raise ValueError("distance-sweep range does not match the CSV")
    for field in ("spacing_or_sampling_rule", "source"):
        _require_physical_text(sweep, field)

    validation = value.get("upstream_validation")
    if not isinstance(validation, dict):
        raise ValueError("physical_provenance.upstream_validation is required")
    if validation.get("overall_passed") is not True:
        raise ValueError("upstream physical validation must be explicitly passed")
    _require_sha256(validation, "report_sha256")
    errors = validation.get("validation_errors")
    limits = validation.get("acceptance_thresholds")
    gates = validation.get("gates")
    if not isinstance(errors, dict) or not errors:
        raise ValueError("upstream validation_errors must be a non-empty object")
    if not isinstance(limits, dict) or not limits:
        raise ValueError("upstream acceptance_thresholds must be a non-empty object")
    if not isinstance(gates, dict) or not gates:
        raise ValueError("upstream validation gates must be a non-empty object")
    if set(errors) != set(limits) or set(errors) != set(gates):
        raise ValueError(
            "upstream errors, thresholds and gates must use identical names"
        )
    for name in errors:
        error_value = _require_finite_number(errors, name)
        if error_value < 0.0:
            raise ValueError("upstream validation errors must be non-negative")
        threshold = _require_finite_number(limits, name, positive=True)
        recomputed_passed = error_value <= threshold
        if gates[name] is not recomputed_passed:
            raise ValueError(
                f"upstream validation gate {name!r} contradicts error/threshold"
            )
        if not recomputed_passed:
            raise ValueError(
                f"upstream validation error {name!r} exceeds its threshold"
            )
    return dict(value)


def load_pair_force_table(
    csv_path: Path, metadata_path: Path
) -> tuple[PairForceTable, dict[str, Any]]:
    metadata = read_json_object(metadata_path)
    if metadata.get("schema_name") != "mechanistic_mv.pair_force_table":
        raise ValueError(
            "metadata schema_name must be mechanistic_mv.pair_force_table"
        )
    if metadata.get("schema_version") != PHASE6_SCHEMA_VERSION:
        raise ValueError("unsupported pair-force metadata schema_version")
    if metadata.get("artifact_type") != "PAIR_FORCE_TABLE":
        raise ValueError("metadata artifact_type must be PAIR_FORCE_TABLE")
    if metadata.get("validation_status") != PairDataValidationStatus.PASSED.value:
        raise ValueError("pair-force metadata validation_status must be PASSED")
    physical_status = _require_text(metadata, "physical_status")
    calibration_status = _require_text(metadata, "calibration_status")
    workflow_status = _require_text(metadata, "workflow_status")
    if physical_status == TEST_ONLY_NOT_CALIBRATED:
        if calibration_status != TEST_ONLY_NOT_CALIBRATED:
            raise ValueError(
                "TEST_ONLY physical_status requires TEST_ONLY calibration_status"
            )
        if workflow_status != "TEST_ONLY_PASSED":
            raise ValueError(
                "TEST_ONLY pair-force data requires TEST_ONLY_PASSED workflow"
            )
        artifact_calibration_id = TEST_ONLY_NOT_CALIBRATED
    elif workflow_status != "PHYSICAL_PASSED":
        raise ValueError("physical pair-force data requires PHYSICAL_PASSED workflow")
    else:
        _require_physical_artifact_labels(
            physical_status, calibration_status
        )
        artifact_calibration_id = _require_physical_text(
            metadata, "calibration_id"
        )

    expected_hash = _require_sha256(metadata, "data_file_sha256")
    actual_hash = sha256_file(csv_path)
    if actual_hash != expected_hash:
        raise ValueError("pair-force CSV SHA256 does not match metadata")
    columns = metadata.get("columns")
    if not isinstance(columns, dict):
        raise ValueError("pair-force metadata must define columns and units")
    if columns.get("center_distance_m") != {"unit": "m"}:
        raise ValueError("center_distance_m must declare SI unit m")
    if columns.get("radial_force_newton") != {"unit": "N"}:
        raise ValueError("radial_force_newton must declare SI unit N")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["center_distance_m", "radial_force_newton"]:
            raise ValueError(
                "pair-force CSV columns must be center_distance_m,radial_force_newton"
            )
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError("pair-force CSV rows must not contain extra columns")
    try:
        distance = np.asarray(
            [float(row["center_distance_m"]) for row in rows], dtype=np.float64
        )
        force = np.asarray(
            [float(row["radial_force_newton"]) for row in rows], dtype=np.float64
        )
    except (TypeError, ValueError) as error:
        raise ValueError("pair-force CSV values must be numeric") from error
    if distance.size == 0:
        raise ValueError("pair-force CSV must contain force samples")
    metadata = _metadata_with_short_range_admission(
        metadata, float(distance[0])
    )

    try:
        scaling = PairForceScaling(_require_text(metadata, "scaling"))
        validation_status = PairDataValidationStatus(
            _require_text(metadata, "validation_status")
        )
        time_scale_status = TimeScaleStatus(
            _require_text(metadata, "time_scale_status")
        )
    except ValueError as error:
        raise ValueError("metadata contains an unsupported enum value") from error

    source = _require_text(metadata, "source")
    artifact_id = _require_text(metadata, "artifact_id")
    solver_status = _require_text(metadata, "solver_status")
    if physical_status != TEST_ONLY_NOT_CALIBRATED:
        _require_physical_text(metadata, "source")
        _require_physical_text(metadata, "artifact_id")
        if not solver_status.startswith("ALL_SAMPLES_CONVERGED:"):
            raise ValueError(
                "physical solver_status must start ALL_SAMPLES_CONVERGED:"
            )

    if scaling is PairForceScaling.UNSCALED_SINGLE_PAIR:
        raise ValueError(
            "UNSCALED_SINGLE_PAIR input is rejected: this script does not "
            "implement or infer Kac/population scaling"
        )
    scaling_provenance = _require_complete_scaling_provenance(
        metadata,
        scaling,
        physical_status=physical_status,
        calibration_id=artifact_calibration_id,
        current_force_table_sha256=expected_hash,
    )
    _require_short_range_provenance(
        metadata,
        physical_status=physical_status,
        calibration_id=artifact_calibration_id,
        minimum_supported_distance_m=float(distance[0]),
    )
    native_scaling, scaling_conversion = _pair_scaling_contract_from_metadata(
        metadata,
        scaling=scaling,
        dataset_id=artifact_id,
        physical_status=physical_status,
        scaling_provenance=scaling_provenance,
    )
    pair_metadata = PairForceMetadata(
        dataset_id=artifact_id,
        source=source,
        physical_status=physical_status,
        solver_status=solver_status,
        validation_status=validation_status,
        time_scale_status=time_scale_status,
        scaling=scaling,
        reference_distance_m=_require_finite_number(
            metadata, "reference_distance_m", positive=True
        ),
        reference_force_tolerance_newton=_require_finite_number(
            metadata, "reference_force_tolerance_newton"
        ),
        native_scaling=native_scaling,
        scaling_conversion=scaling_conversion,
    )
    table = PairForceTable(distance, force, pair_metadata)
    if physical_status != TEST_ONLY_NOT_CALIBRATED:
        _require_physical_provenance(
            metadata,
            distance=distance,
            calibration_status=calibration_status,
            calibration_id=artifact_calibration_id,
            time_scale_status=time_scale_status,
            solver_status=solver_status,
        )
    return table, metadata


def load_effective_potential_artifact(
    csv_path: Path, metadata_path: Path
) -> tuple[HydrogelEffectivePairPotential, dict[str, Any]]:
    """Load only a complete, passed effective-potential artifact.

    The stored energy column is independently recomputed from the force table;
    it is not trusted merely because the sidecar hash matches.
    """

    metadata = read_json_object(metadata_path)
    if metadata.get("schema_name") != "mechanistic_mv.effective_pair_potential":
        raise ValueError(
            "metadata schema_name must be mechanistic_mv.effective_pair_potential"
        )
    if metadata.get("schema_version") != PHASE6_SCHEMA_VERSION:
        raise ValueError("unsupported effective-potential schema_version")
    if metadata.get("artifact_type") != "EFFECTIVE_PAIR_POTENTIAL":
        raise ValueError("metadata artifact_type must be EFFECTIVE_PAIR_POTENTIAL")
    if metadata.get("overall_passed") is not True:
        raise ValueError("effective-potential artifact must have overall_passed=true")

    physical_status = _require_text(metadata, "physical_status")
    calibration_status = _require_text(metadata, "calibration_status")
    workflow_status = _require_text(metadata, "workflow_status")
    if physical_status == TEST_ONLY_NOT_CALIBRATED:
        if calibration_status != TEST_ONLY_NOT_CALIBRATED:
            raise ValueError(
                "TEST_ONLY physical_status requires TEST_ONLY calibration_status"
            )
        if workflow_status != "TEST_ONLY_PASSED":
            raise ValueError(
                "TEST_ONLY effective potential requires TEST_ONLY_PASSED workflow"
            )
        artifact_calibration_id = TEST_ONLY_NOT_CALIBRATED
    elif workflow_status != "PHYSICAL_INPUT_NUMERICAL_GATES_PASSED":
        raise ValueError(
            "physical effective potential requires "
            "PHYSICAL_INPUT_NUMERICAL_GATES_PASSED workflow"
        )
    else:
        _require_physical_artifact_labels(
            physical_status, calibration_status
        )
        artifact_calibration_id = _require_physical_text(
            metadata, "calibration_id"
        )

    expected_hash = _require_sha256(metadata, "data_file_sha256")
    if sha256_file(csv_path) != expected_hash:
        raise ValueError("effective-potential CSV SHA256 does not match metadata")
    if metadata.get("data_file") != csv_path.name:
        raise ValueError("effective-potential data_file does not match CSV filename")
    if metadata.get("units") != {
        "center_distance_m": "m",
        "radial_force_newton": "N",
        "effective_potential_joule": "J",
    }:
        raise ValueError("effective-potential columns must declare exact SI units")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != [
            "center_distance_m",
            "radial_force_newton",
            "effective_potential_joule",
        ]:
            raise ValueError(
                "effective-potential CSV must contain distance, force and energy"
            )
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(
            "effective-potential CSV rows must not contain extra columns"
        )
    try:
        distance = np.asarray(
            [float(row["center_distance_m"]) for row in rows], dtype=np.float64
        )
        force = np.asarray(
            [float(row["radial_force_newton"]) for row in rows], dtype=np.float64
        )
        stored_potential = np.asarray(
            [float(row["effective_potential_joule"]) for row in rows],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("effective-potential CSV values must be numeric") from error
    if distance.size == 0:
        raise ValueError("effective-potential CSV must contain samples")
    if not np.all(np.isfinite(stored_potential)):
        raise ValueError("effective-potential energy values must be finite")
    metadata = _metadata_with_short_range_admission(
        metadata, float(distance[0])
    )

    raw_pair_metadata = metadata.get("pair_force_metadata")
    if not isinstance(raw_pair_metadata, dict):
        raise ValueError("complete pair_force_metadata is required")
    raw_pair_metadata = _metadata_with_short_range_admission(
        raw_pair_metadata, float(distance[0])
    )
    metadata["pair_force_metadata"] = raw_pair_metadata
    try:
        scaling = PairForceScaling(_require_text(raw_pair_metadata, "scaling"))
        validation_status = PairDataValidationStatus(
            _require_text(raw_pair_metadata, "validation_status")
        )
        time_scale_status = TimeScaleStatus(
            _require_text(raw_pair_metadata, "time_scale_status")
        )
    except ValueError as error:
        raise ValueError("pair_force_metadata contains an unsupported enum") from error
    if scaling is not PairForceScaling.KAC_NORMALIZED_PROBABILITY:
        raise ValueError("effective potential must have explicit Kac scaling")
    if raw_pair_metadata.get("physical_status") != physical_status:
        raise ValueError("pair-force and effective-potential physical_status differ")
    if metadata.get("scaling") != scaling.value:
        raise ValueError("effective-potential scaling fields disagree")
    source = _require_text(raw_pair_metadata, "source")
    dataset_id = _require_text(raw_pair_metadata, "dataset_id")
    solver_status = _require_text(raw_pair_metadata, "solver_status")
    if physical_status != TEST_ONLY_NOT_CALIBRATED:
        _require_physical_text(raw_pair_metadata, "source")
        _require_physical_text(raw_pair_metadata, "dataset_id")
        if not solver_status.startswith("ALL_SAMPLES_CONVERGED:"):
            raise ValueError(
                "physical solver_status must start ALL_SAMPLES_CONVERGED:"
            )
    source_artifacts = metadata.get("source_artifacts")
    if not isinstance(source_artifacts, dict):
        raise ValueError("effective potential requires source_artifacts metadata")
    source_force_sha256 = _require_sha256(
        source_artifacts, "source_force_sha256"
    )
    if physical_status != TEST_ONLY_NOT_CALIBRATED:
        _require_text(source_artifacts, "force_csv")
        _require_text(source_artifacts, "force_metadata")
        _require_sha256(source_artifacts, "force_metadata_sha256")
    scaling_provenance = _require_complete_scaling_provenance(
        metadata,
        scaling,
        physical_status=physical_status,
        calibration_id=artifact_calibration_id,
        current_force_table_sha256=source_force_sha256,
    )
    _require_short_range_provenance(
        metadata,
        physical_status=physical_status,
        calibration_id=artifact_calibration_id,
        minimum_supported_distance_m=float(distance[0]),
    )
    native_scaling, scaling_conversion = _pair_scaling_contract_from_metadata(
        raw_pair_metadata,
        scaling=scaling,
        dataset_id=dataset_id,
        physical_status=physical_status,
        scaling_provenance=scaling_provenance,
    )
    pair_metadata = PairForceMetadata(
        dataset_id=dataset_id,
        source=source,
        physical_status=physical_status,
        solver_status=solver_status,
        validation_status=validation_status,
        time_scale_status=time_scale_status,
        scaling=scaling,
        reference_distance_m=_require_finite_number(
            raw_pair_metadata, "reference_distance_m", positive=True
        ),
        reference_force_tolerance_newton=_require_finite_number(
            raw_pair_metadata, "reference_force_tolerance_newton"
        ),
        native_scaling=native_scaling,
        scaling_conversion=scaling_conversion,
    )
    table = PairForceTable(distance, force, pair_metadata)
    if physical_status != TEST_ONLY_NOT_CALIBRATED:
        _require_physical_provenance(
            metadata,
            distance=distance,
            calibration_status=calibration_status,
            calibration_id=artifact_calibration_id,
            time_scale_status=time_scale_status,
            solver_status=solver_status,
        )
    potential = effective_potential_from_table(table)
    recomputed = potential.radial_potential_joule(distance)
    if not np.allclose(
        stored_potential,
        recomputed,
        rtol=32.0 * np.finfo(np.float64).eps,
        atol=1.0e-36,
    ):
        raise ValueError(
            "stored effective potential is inconsistent with the force antiderivative"
        )
    return potential, metadata


def effective_potential_from_table(
    table: PairForceTable,
) -> HydrogelEffectivePairPotential:
    return HydrogelEffectivePairPotential(
        table,
        derivative_absolute_tolerance_newton=(
            FORCE_DERIVATIVE_ABSOLUTE_TOLERANCE_N
        ),
        derivative_relative_tolerance=FORCE_DERIVATIVE_RELATIVE_TOLERANCE,
        finite_difference_step_fraction=FORCE_DERIVATIVE_STEP_FRACTION,
    )
