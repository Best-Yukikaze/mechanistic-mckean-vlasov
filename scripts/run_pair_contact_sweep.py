"""Collect and preflight real contact inputs without running a FEM solve."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from mechanistic_mv.mechanics.density_scaling import DensityConvention
from mechanistic_mv.mechanics.hydrogel import (
    HydrogelParameters,
    assess_time_scale_separation,
)
from mechanistic_mv.mechanics.pair_interaction import (
    ContactInputPurpose,
    ContactLawAndTolerances,
    ContactProblemInput,
    ContactSolverSpecification,
    DistanceScanPlan,
    InputProvenance,
    InputVerificationStatus,
    MeanFieldScalingInput,
    MeanFieldScalingMode,
    MechanicalBoundaryConditions,
    MeshConvergencePlan,
    NormalContactLaw,
    PairAxisDefinition,
    PairForceScaling,
    ReactionForceConvention,
    ReactionForceSignConvention,
    ShortRangeClosureInput,
    ShortRangeClosureKind,
    SolventBathBoundaryConditions,
    SolverInputStatus,
    TotalFreeEnergyReference,
    TwoSphereGeometry,
)

try:
    from ._phase6_common import (
        PAIR_VALIDATION_DIRECTORY,
        PHASE6_SCHEMA_VERSION,
        provenance,
        sha256_file,
        write_json,
    )
except ImportError:  # direct ``python scripts/...`` execution
    from _phase6_common import (  # type: ignore[no-redef]
        PAIR_VALIDATION_DIRECTORY,
        PHASE6_SCHEMA_VERSION,
        provenance,
        sha256_file,
        write_json,
    )


MANIFEST_SCHEMA_NAME = "mechanistic_mv.contact_problem_input_manifest"
MANIFEST_SCHEMA_VERSION = 1
INPUT_TEMPLATE_NOT_PHYSICAL = "INPUT_TEMPLATE_NOT_PHYSICAL"
CONTACT_FEM_BACKEND_UNAVAILABLE = "CONTACT_FEM_BACKEND_UNAVAILABLE"
CONTACT_RESULTS_NOT_GENERATED = "CONTACT_RESULTS_NOT_GENERATED"

_PROHIBITED_RESULT_FILES = (
    "pair_force.csv",
    "force_curve.png",
    "potential_curve.png",
    "force_potential_consistency.png",
)
_TEST_ONLY_MARKERS = (
    "test_only",
    "test-only",
    "test only",
    "not_calibrated",
    "not-calibrated",
    "not calibrated",
)


class _ManifestSnapshotError(ValueError):
    """Strict JSON error that retains the hash of the bytes that failed."""

    def __init__(self, message: str, sha256: str | None) -> None:
        super().__init__(message)
        self.sha256 = sha256

_NUMERIC_UNITS: dict[str, tuple[str, str]] = {
    "hydrogel_parameters.network_density_times_solvent_volume": ("number", "1"),
    "hydrogel_parameters.flory_huggins_chi": ("number", "1"),
    "hydrogel_parameters.initial_polymer_volume_fraction": ("number", "1"),
    "hydrogel_parameters.delta_chemical_potential_over_kbt": ("number", "1"),
    "hydrogel_parameters.thermal_energy_density_pa": ("number", "Pa"),
    "geometry.first_radius_m": ("number", "m"),
    "geometry.second_radius_m": ("number", "m"),
    "solvent_bath_boundary_conditions.bath_chemical_potential_over_kbt": (
        "number",
        "1",
    ),
    "contact_law.normal_gap_tolerance_m": ("number", "m"),
    "contact_law.force_balance_tolerance_newton": ("number", "N"),
    "mesh_convergence.characteristic_lengths_m": ("array[number]", "m"),
    "mesh_convergence.nonlinear_residual_tolerance_dimensionless": (
        "number",
        "1",
    ),
    "mesh_convergence.relative_force_convergence_tolerance_dimensionless": (
        "number",
        "1",
    ),
    "mesh_convergence.relative_energy_convergence_tolerance_dimensionless": (
        "number",
        "1",
    ),
    "mesh_convergence.relative_stress_convergence_tolerance_dimensionless": (
        "number",
        "1",
    ),
    "mesh_convergence.maximum_nonlinear_iterations": ("integer", "1 (count)"),
    "distance_scan.center_distances_m": ("array[number]", "m"),
    "distance_scan.reference_distance_m": ("number", "m"),
    "distance_scan.reference_force_tolerance_newton": ("number", "N"),
    "reaction_force.integration_tolerance_newton": ("number", "N"),
    "time_scale.tau_gel_s": ("number|null", "s"),
    "time_scale.tau_swarm_s": ("number|null", "s"),
    "time_scale.required_max_ratio": ("number|null", "1"),
    "mean_field_scaling.population_count": ("integer", "1 (particle count)"),
    "mean_field_scaling.source_population_count": (
        "integer",
        "1 (particle count)",
    ),
    "mean_field_scaling.areal_number_density_per_m2": ("number|null", "1/m^2"),
    "mean_field_scaling.representative_area_m2": ("number|null", "m^2"),
    "short_range.minimum_supported_distance_m": ("number", "m"),
    "short_range.force_match_tolerance_newton": ("number|null", "N"),
    "short_range.energy_match_tolerance_joule": ("number|null", "J"),
}

_BOOLEAN_PATHS = {"contact_law.frictionless", "contact_law.adhesive"}
_NULLABLE_TEXT_PATHS = {
    path
    for paths in ContactProblemInput.required_input_field_paths().values()
    for path in paths
    if path.endswith("verification_record_id")
} | {
    "short_range.closure_description",
    "short_range.validation_method",
}
_ENUM_VALUES: dict[str, list[str]] = {
    "contact_law.normal_contact_model": [item.value for item in NormalContactLaw],
    "reaction_force.pair_axis_definition": [item.value for item in PairAxisDefinition],
    "reaction_force.positive_sign_convention": [
        item.value for item in ReactionForceSignConvention
    ],
    "reaction_force.total_free_energy_reference": [
        item.value for item in TotalFreeEnergyReference
    ],
    "mean_field_scaling.density_convention": [item.value for item in DensityConvention],
    "mean_field_scaling.scaling_mode": [item.value for item in MeanFieldScalingMode],
    "short_range.closure_kind": [item.value for item in ShortRangeClosureKind],
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PAIR_VALIDATION_DIRECTORY,
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--write-input-template", type=Path)
    actions.add_argument("--input-manifest", type=Path)
    return parser


def _field_specification(path: str) -> dict[str, object]:
    if path in _NUMERIC_UNITS:
        json_type, unit = _NUMERIC_UNITS[path]
        return {
            "field_path": path,
            "expected_json_type": json_type,
            "unit_or_dimension": unit,
        }
    if path in _BOOLEAN_PATHS:
        return {
            "field_path": path,
            "expected_json_type": "boolean",
            "unit_or_dimension": "not applicable (boolean)",
        }
    allowed_values = _ENUM_VALUES.get(path)
    if path.endswith("verification_status"):
        allowed_values = [item.value for item in InputVerificationStatus]
    if allowed_values is not None:
        return {
            "field_path": path,
            "expected_json_type": "string (enum)",
            "unit_or_dimension": "not applicable (enum)",
            "allowed_values": allowed_values,
        }
    return {
        "field_path": path,
        "expected_json_type": (
            "string|null" if path in _NULLABLE_TEXT_PATHS else "string"
        ),
        "unit_or_dimension": "not applicable (text/provenance)",
    }


def _input_contract_schema() -> list[dict[str, object]]:
    return [
        {
            "requirement_id": requirement_id,
            "fields": [_field_specification(path) for path in paths],
        }
        for requirement_id, paths in (
            ContactProblemInput.required_input_field_paths().items()
        )
    ]


def _set_nested_null(root: dict[str, object], path: str) -> None:
    parts = path.split(".")
    cursor = root
    for part in parts[:-1]:
        child = cursor.setdefault(part, {})
        if not isinstance(child, dict):
            raise RuntimeError("contact input field paths overlap unexpectedly")
        cursor = child
    cursor[parts[-1]] = None


def _template_payload() -> dict[str, object]:
    contact_problem: dict[str, object] = {}
    for paths in ContactProblemInput.required_input_field_paths().values():
        for path in paths:
            _set_nested_null(contact_problem, path)
    return {
        "schema_name": MANIFEST_SCHEMA_NAME,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_type": "CONTACT_PROBLEM_INPUT_TEMPLATE",
        "template_status": INPUT_TEMPLATE_NOT_PHYSICAL,
        "input_purpose": INPUT_TEMPLATE_NOT_PHYSICAL,
        "physical_values_supplied": False,
        "contact_problem": contact_problem,
        "required_input_contract": _input_contract_schema(),
        "instructions": [
            "replace nulls only with user-supplied, source-traceable values",
            "set input_purpose to PHYSICAL_CONTACT_PROBLEM only for physical data",
            "the template is not contact data and cannot authorize a FEM solve",
        ],
    }


_MISSING = object()


def _lookup_path(root: Mapping[str, object], path: str) -> object:
    cursor: object = root
    for part in path.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return _MISSING
        cursor = cursor[part]
    return cursor


def _field_reports(
    contact_problem: Mapping[str, object],
    group_statuses: Mapping[str, str] | None = None,
) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    statuses = {} if group_statuses is None else group_statuses
    for requirement_id, paths in (
        ContactProblemInput.required_input_field_paths().items()
    ):
        for path in paths:
            value = _lookup_path(contact_problem, path)
            specification = _field_specification(path)
            specification.update(
                {
                    "requirement_id": requirement_id,
                    "presence_status": (
                        "MISSING_KEY"
                        if value is _MISSING
                        else "EXPLICIT_NULL"
                        if value is None
                        else "PRESENT"
                    ),
                    "contract_group_status": statuses.get(
                        requirement_id, "NOT_VALIDATED"
                    ),
                }
            )
            reports.append(specification)
    return reports


def _allowed_tree() -> dict[str, object]:
    root: dict[str, object] = {}
    for paths in ContactProblemInput.required_input_field_paths().values():
        for path in paths:
            cursor = root
            for part in path.split("."):
                child = cursor.setdefault(part, {})
                if not isinstance(child, dict):
                    raise RuntimeError("contact input field paths overlap unexpectedly")
                cursor = child
    return root


def _reject_unknown_fields(
    supplied: Mapping[str, object],
    allowed: Mapping[str, object],
    prefix: str = "contact_problem",
) -> None:
    unknown = sorted(set(supplied) - set(allowed))
    if unknown:
        raise ValueError(f"{prefix} contains unknown fields: {', '.join(unknown)}")
    for name, value in supplied.items():
        child = allowed[name]
        if isinstance(child, Mapping) and child:
            if not isinstance(value, Mapping):
                raise TypeError(f"{prefix}.{name} must be a JSON object")
            _reject_unknown_fields(value, child, f"{prefix}.{name}")


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a JSON object")
    return value


def _array(value: object, path: str) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{path} must be a JSON array")
    return tuple(value)


def _enum(enum_type: type[Any], value: object, path: str) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} contains an unsupported enum value") from error


def _provenance(value: object, path: str) -> InputProvenance:
    payload = _mapping(value, path)
    return InputProvenance(
        source_id=payload.get("source_id"),
        source_description=payload.get("source_description"),
        verification_status=_enum(
            InputVerificationStatus,
            payload.get("verification_status"),
            f"{path}.verification_status",
        ),
        verification_record_id=payload.get("verification_record_id"),
    )


def _group_is_structurally_complete(
    contact_problem: Mapping[str, object], requirement_id: str
) -> bool:
    paths = ContactProblemInput.required_input_field_paths()[requirement_id]
    return all(_lookup_path(contact_problem, path) is not _MISSING for path in paths)


def _decode_contact_problem(
    manifest: Mapping[str, object],
) -> tuple[ContactProblemInput, Mapping[str, object]]:
    allowed_root = {
        "schema_name",
        "schema_version",
        "input_purpose",
        "contact_problem",
    }
    unknown_root = sorted(set(manifest) - allowed_root)
    if unknown_root:
        raise ValueError(
            "manifest contains unknown root fields: " + ", ".join(unknown_root)
        )
    if manifest.get("schema_name") != MANIFEST_SCHEMA_NAME:
        raise ValueError(f"schema_name must be {MANIFEST_SCHEMA_NAME}")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported contact input manifest schema_version")
    purpose = _enum(
        ContactInputPurpose,
        manifest.get("input_purpose"),
        "input_purpose",
    )
    contact = _mapping(manifest.get("contact_problem"), "contact_problem")
    _reject_unknown_fields(contact, _allowed_tree())
    requirements = ContactProblemInput.REQUIRED_INPUT_NAMES
    values: dict[str, object] = {}

    if _group_is_structurally_complete(contact, requirements[0]):
        hydrogel = _mapping(contact["hydrogel_parameters"], "hydrogel_parameters")
        values["hydrogel_parameters"] = HydrogelParameters(
            network_density_times_solvent_volume=hydrogel.get(
                "network_density_times_solvent_volume"
            ),
            flory_huggins_chi=hydrogel.get("flory_huggins_chi"),
            initial_polymer_volume_fraction=hydrogel.get(
                "initial_polymer_volume_fraction"
            ),
            delta_chemical_potential_over_kbt=hydrogel.get(
                "delta_chemical_potential_over_kbt"
            ),
            thermal_energy_density_pa=hydrogel.get("thermal_energy_density_pa"),
            calibration_status=hydrogel.get("calibration_status"),
        )
        values["hydrogel_parameters_provenance"] = _provenance(
            contact["hydrogel_parameters_provenance"],
            "hydrogel_parameters_provenance",
        )

    if _group_is_structurally_complete(contact, requirements[1]):
        geometry = _mapping(contact["geometry"], "geometry")
        values["geometry"] = TwoSphereGeometry(
            first_radius_m=geometry.get("first_radius_m"),
            second_radius_m=geometry.get("second_radius_m"),
            calibration_status=geometry.get("calibration_status"),
        )
        values["geometry_provenance"] = _provenance(
            contact["geometry_provenance"], "geometry_provenance"
        )

    if _group_is_structurally_complete(contact, requirements[2]):
        item = _mapping(
            contact["mechanical_boundary_conditions"],
            "mechanical_boundary_conditions",
        )
        values["mechanical_boundary_conditions"] = MechanicalBoundaryConditions(
            center_separation_control=item.get("center_separation_control"),
            rigid_body_constraint=item.get("rigid_body_constraint"),
            noncontact_surface_traction=item.get("noncontact_surface_traction"),
            loading_protocol=item.get("loading_protocol"),
            provenance=_provenance(
                item.get("provenance"),
                "mechanical_boundary_conditions.provenance",
            ),
        )

    if _group_is_structurally_complete(contact, requirements[3]):
        item = _mapping(
            contact["solvent_bath_boundary_conditions"],
            "solvent_bath_boundary_conditions",
        )
        values["solvent_bath_boundary_conditions"] = SolventBathBoundaryConditions(
            bath_chemical_potential_over_kbt=item.get(
                "bath_chemical_potential_over_kbt"
            ),
            exposed_surface_exchange_condition=item.get(
                "exposed_surface_exchange_condition"
            ),
            contact_surface_transport_condition=item.get(
                "contact_surface_transport_condition"
            ),
            initial_solvent_state=item.get("initial_solvent_state"),
            provenance=_provenance(
                item.get("provenance"),
                "solvent_bath_boundary_conditions.provenance",
            ),
        )

    if _group_is_structurally_complete(contact, requirements[4]):
        item = _mapping(contact["contact_law"], "contact_law")
        values["contact_law"] = ContactLawAndTolerances(
            normal_contact_model=_enum(
                NormalContactLaw,
                item.get("normal_contact_model"),
                "contact_law.normal_contact_model",
            ),
            contact_enforcement_method=item.get("contact_enforcement_method"),
            frictionless=item.get("frictionless"),
            adhesive=item.get("adhesive"),
            normal_gap_tolerance_m=item.get("normal_gap_tolerance_m"),
            force_balance_tolerance_newton=item.get(
                "force_balance_tolerance_newton"
            ),
            provenance=_provenance(item.get("provenance"), "contact_law.provenance"),
        )

    if _group_is_structurally_complete(contact, requirements[5]):
        item = _mapping(contact["solver"], "solver")
        values["solver"] = ContactSolverSpecification(
            solver_name=item.get("solver_name"),
            solver_version=item.get("solver_version"),
            implementation_id=item.get("implementation_id"),
            configuration_id=item.get("configuration_id"),
            nonlinear_algorithm=item.get("nonlinear_algorithm"),
            linear_solver=item.get("linear_solver"),
            provenance=_provenance(item.get("provenance"), "solver.provenance"),
        )

    if _group_is_structurally_complete(contact, requirements[6]):
        item = _mapping(contact["mesh_convergence"], "mesh_convergence")
        values["mesh_convergence"] = MeshConvergencePlan(
            discretization_method=item.get("discretization_method"),
            element_family=item.get("element_family"),
            characteristic_lengths_m=_array(
                item.get("characteristic_lengths_m"),
                "mesh_convergence.characteristic_lengths_m",
            ),
            nonlinear_residual_tolerance_dimensionless=item.get(
                "nonlinear_residual_tolerance_dimensionless"
            ),
            relative_force_convergence_tolerance_dimensionless=item.get(
                "relative_force_convergence_tolerance_dimensionless"
            ),
            relative_energy_convergence_tolerance_dimensionless=item.get(
                "relative_energy_convergence_tolerance_dimensionless"
            ),
            relative_stress_convergence_tolerance_dimensionless=item.get(
                "relative_stress_convergence_tolerance_dimensionless"
            ),
            maximum_nonlinear_iterations=item.get("maximum_nonlinear_iterations"),
            provenance=_provenance(
                item.get("provenance"), "mesh_convergence.provenance"
            ),
        )

    if _group_is_structurally_complete(contact, requirements[7]):
        item = _mapping(contact["distance_scan"], "distance_scan")
        values["distance_scan"] = DistanceScanPlan(
            center_distances_m=_array(
                item.get("center_distances_m"), "distance_scan.center_distances_m"
            ),
            reference_distance_m=item.get("reference_distance_m"),
            reference_force_tolerance_newton=item.get(
                "reference_force_tolerance_newton"
            ),
            provenance=_provenance(
                item.get("provenance"), "distance_scan.provenance"
            ),
        )

    if _group_is_structurally_complete(contact, requirements[8]):
        item = _mapping(contact["reaction_force"], "reaction_force")
        values["reaction_force"] = ReactionForceConvention(
            integration_boundary=item.get("integration_boundary"),
            traction_quantity=item.get("traction_quantity"),
            pair_axis_definition=_enum(
                PairAxisDefinition,
                item.get("pair_axis_definition"),
                "reaction_force.pair_axis_definition",
            ),
            positive_sign_convention=_enum(
                ReactionForceSignConvention,
                item.get("positive_sign_convention"),
                "reaction_force.positive_sign_convention",
            ),
            total_free_energy_reference=_enum(
                TotalFreeEnergyReference,
                item.get("total_free_energy_reference"),
                "reaction_force.total_free_energy_reference",
            ),
            integration_tolerance_newton=item.get("integration_tolerance_newton"),
            provenance=_provenance(
                item.get("provenance"), "reaction_force.provenance"
            ),
        )

    if _group_is_structurally_complete(contact, requirements[9]):
        item = _mapping(contact["time_scale"], "time_scale")
        values["time_scale"] = assess_time_scale_separation(
            tau_gel_s=item.get("tau_gel_s"),
            tau_swarm_s=item.get("tau_swarm_s"),
            required_max_ratio=item.get("required_max_ratio"),
        )
        values["time_scale_provenance"] = _provenance(
            contact["time_scale_provenance"], "time_scale_provenance"
        )

    if _group_is_structurally_complete(contact, requirements[10]):
        item = _mapping(contact["mean_field_scaling"], "mean_field_scaling")
        values["mean_field_scaling"] = MeanFieldScalingInput(
            density_convention=_enum(
                DensityConvention,
                item.get("density_convention"),
                "mean_field_scaling.density_convention",
            ),
            scaling_mode=_enum(
                MeanFieldScalingMode,
                item.get("scaling_mode"),
                "mean_field_scaling.scaling_mode",
            ),
            population_count=item.get("population_count"),
            source_population_count=item.get("source_population_count"),
            areal_number_density_per_m2=item.get("areal_number_density_per_m2"),
            representative_area_m2=item.get("representative_area_m2"),
            scaling_definition=item.get("scaling_definition"),
            provenance=_provenance(
                item.get("provenance"), "mean_field_scaling.provenance"
            ),
        )

    if _group_is_structurally_complete(contact, requirements[11]):
        item = _mapping(contact["short_range"], "short_range")
        values["short_range"] = ShortRangeClosureInput(
            minimum_supported_distance_m=item.get("minimum_supported_distance_m"),
            closure_kind=_enum(
                ShortRangeClosureKind,
                item.get("closure_kind"),
                "short_range.closure_kind",
            ),
            closure_description=item.get("closure_description"),
            validation_method=item.get("validation_method"),
            force_match_tolerance_newton=item.get("force_match_tolerance_newton"),
            energy_match_tolerance_joule=item.get("energy_match_tolerance_joule"),
            provenance=_provenance(item.get("provenance"), "short_range.provenance"),
        )

    return ContactProblemInput(purpose=purpose, **values), contact


def _contains_test_only_marker(value: object) -> bool:
    if isinstance(value, str):
        folded = value.casefold()
        return any(marker in folded for marker in _TEST_ONLY_MARKERS)
    if isinstance(value, Mapping):
        return any(_contains_test_only_marker(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_test_only_marker(item) for item in value)
    return False


def _read_manifest_snapshot(path: Path) -> tuple[dict[str, object], str]:
    """Parse and hash the exact same bytes with strict JSON semantics."""

    try:
        raw = path.read_bytes()
    except OSError as error:
        raise _ManifestSnapshotError(
            f"input manifest cannot be read: {path}", None
        ) from error
    digest = hashlib.sha256(raw).hexdigest()

    def reject_nonfinite_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value!r} is forbidden")

    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} is forbidden")
            result[key] = value
        return result

    try:
        decoded = raw.decode("utf-8")
        value = json.loads(
            decoded,
            parse_constant=reject_nonfinite_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise _ManifestSnapshotError(
            f"input manifest is not strict UTF-8 JSON: {path}", digest
        ) from error
    if not isinstance(value, dict):
        raise _ManifestSnapshotError(
            "input manifest root must be a JSON object", digest
        )
    return value, digest


def _blank_readiness() -> tuple[dict[str, object], list[dict[str, object]]]:
    problem = ContactProblemInput(purpose=ContactInputPurpose.PHYSICAL)
    readiness = problem.evaluate_readiness()
    statuses = {
        item.requirement_id: item.status.value for item in readiness.required_inputs
    }
    return readiness.to_dict(), _field_reports({}, statuses)


def _preexisting_result_files(output_directory: Path) -> list[str]:
    return [
        name
        for name in _PROHIBITED_RESULT_FILES
        if (output_directory / name).exists()
    ]


def _write_status_reports(
    output_directory: Path,
    *,
    payload: dict[str, object],
) -> Path:
    status_path = output_directory / "pair_contact_sweep_status.json"
    metadata_path = output_directory / "metadata.json"
    common = {
        "schema_version": PHASE6_SCHEMA_VERSION,
        "workflow_status": "BLOCKED",
        "contact_solver_status": CONTACT_FEM_BACKEND_UNAVAILABLE,
        "contact_fem_backend_status": CONTACT_FEM_BACKEND_UNAVAILABLE,
        "contact_results_status": CONTACT_RESULTS_NOT_GENERATED,
        "fem_backend_available": False,
        "physical_results_present": False,
        "physical_results_generated": False,
        "pair_force_table_generated": False,
        "data_semantics": PairForceScaling.UNSCALED_SINGLE_PAIR.value,
        "generated_data_files": [],
        "preexisting_untrusted_artifacts": _preexisting_result_files(
            output_directory
        ),
        "prohibited_fabricated_outputs": list(_PROHIBITED_RESULT_FILES),
        **payload,
    }
    write_json(
        metadata_path,
        {
            "schema_name": "mechanistic_mv.pair_contact_sweep_metadata",
            "artifact_type": "PAIR_CONTACT_SWEEP_METADATA",
            **common,
        },
    )
    write_json(
        status_path,
        {
            "schema_name": "mechanistic_mv.pair_contact_sweep_status",
            "artifact_type": "PAIR_CONTACT_SWEEP_STATUS",
            "metadata_file": metadata_path.name,
            **common,
        },
    )
    return status_path


def _base_payload(
    *,
    physical_status: str,
    validation_status: str,
    manifest_status: str,
    manifest_file: str | None,
    manifest_hash: str | None,
    input_purpose: str,
    readiness: dict[str, object],
    fields: list[dict[str, object]],
    reason: str,
    run_provenance: dict[str, object],
) -> dict[str, object]:
    return {
        "physical_status": physical_status,
        "validation_status": validation_status,
        "input_manifest_status": manifest_status,
        "input_manifest_file": manifest_file,
        "input_manifest_sha256": manifest_hash,
        "input_purpose": input_purpose,
        "input_readiness": SolverInputStatus.BLOCKED.value,
        "physics_readiness": readiness,
        "generated_parameter_collection_files": [],
        "required_input_contract": _input_contract_schema(),
        "required_input_groups": readiness["required_inputs"],
        "required_inputs": fields,
        "reason": reason,
        **run_provenance,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_directory: Path = args.output_directory
    command = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    run_provenance = provenance(command)
    blank_readiness, blank_fields = _blank_readiness()

    if args.write_input_template is not None:
        template_path: Path = args.write_input_template
        if template_path.suffix.casefold() != ".json":
            payload = _base_payload(
                physical_status=INPUT_TEMPLATE_NOT_PHYSICAL,
                validation_status="UNVERIFIED",
                manifest_status="NOT_PROVIDED",
                manifest_file=None,
                manifest_hash=None,
                input_purpose=INPUT_TEMPLATE_NOT_PHYSICAL,
                readiness=blank_readiness,
                fields=blank_fields,
                reason="--write-input-template path must end in .json",
                run_provenance=run_provenance,
            )
            status_path = _write_status_reports(output_directory, payload=payload)
            print(status_path)
            return 2
        write_json(template_path, _template_payload())
        payload = _base_payload(
            physical_status=INPUT_TEMPLATE_NOT_PHYSICAL,
            validation_status="INPUT_TEMPLATE_ONLY",
            manifest_status="NOT_PROVIDED",
            manifest_file=None,
            manifest_hash=None,
            input_purpose=INPUT_TEMPLATE_NOT_PHYSICAL,
            readiness=blank_readiness,
            fields=blank_fields,
            reason=(
                "an all-null parameter collection template was written; it is "
                "not physical input and no contact result was generated"
            ),
            run_provenance=run_provenance,
        )
        payload.update(
            {
                "generated_parameter_collection_files": [str(template_path)],
                "input_template_file": str(template_path),
                "input_template_sha256": sha256_file(template_path),
            }
        )
        _write_status_reports(output_directory, payload=payload)
        print(template_path)
        return 0

    if args.input_manifest is None:
        payload = _base_payload(
            physical_status="PHYSICAL_CONTACT_DATA_NOT_AVAILABLE",
            validation_status="UNVERIFIED",
            manifest_status="MISSING",
            manifest_file=None,
            manifest_hash=None,
            input_purpose="NOT_PROVIDED",
            readiness=blank_readiness,
            fields=blank_fields,
            reason=(
                "no input manifest was supplied; all physical contact inputs "
                "remain missing and no force samples were run"
            ),
            run_provenance=run_provenance,
        )
        status_path = _write_status_reports(output_directory, payload=payload)
        print(status_path)
        return 2

    manifest_path: Path = args.input_manifest
    manifest_hash: str | None = None
    contact_problem: Mapping[str, object] = {}
    input_purpose = "NOT_PARSED"
    try:
        manifest, manifest_hash = _read_manifest_snapshot(manifest_path)
        raw_contact = manifest.get("contact_problem")
        if isinstance(raw_contact, Mapping):
            contact_problem = raw_contact
        raw_purpose = manifest.get("input_purpose")
        if isinstance(raw_purpose, str):
            input_purpose = raw_purpose
        if _contains_test_only_marker(manifest):
            raise ValueError(
                "TEST_ONLY or NOT_CALIBRATED manifest content cannot claim "
                "physical contact readiness"
            )
        problem, contact_problem = _decode_contact_problem(manifest)
        readiness = problem.evaluate_readiness()
        statuses = {
            item.requirement_id: item.status.value
            for item in readiness.required_inputs
        }
        field_reports = _field_reports(contact_problem, statuses)
        physical_ready = bool(
            problem.purpose is ContactInputPurpose.PHYSICAL
            and readiness.solver_input_status
            is SolverInputStatus.READY_FOR_CONTACT_SOLVER
            and readiness.physical_inputs_verified
        )
        payload = _base_payload(
            physical_status=(
                "PHYSICAL_INPUTS_VERIFIED_CONTACT_RESULTS_ABSENT"
                if physical_ready
                else "PHYSICAL_CONTACT_DATA_NOT_AVAILABLE"
            ),
            validation_status=(
                "INPUT_CONTRACT_VERIFIED" if physical_ready else "UNVERIFIED"
            ),
            manifest_status="PARSED_AND_CONTRACT_CHECKED",
            manifest_file=str(manifest_path),
            manifest_hash=manifest_hash,
            input_purpose=input_purpose,
            readiness=readiness.to_dict(),
            fields=field_reports,
            reason=(
                "physical input contract is verified and ready for a future "
                "contact solver, but the FEM backend is unavailable and no "
                "result was generated"
                if physical_ready
                else (
                    "input contract is incomplete, unverified, or nonphysical; "
                    "contact solve remains blocked"
                )
            ),
            run_provenance=run_provenance,
        )
        if physical_ready:
            payload["input_readiness"] = (
                SolverInputStatus.READY_FOR_CONTACT_SOLVER.value
            )
        status_path = _write_status_reports(output_directory, payload=payload)
        print(status_path)
        return 2
    except (OSError, TypeError, ValueError) as error:
        if isinstance(error, _ManifestSnapshotError):
            manifest_hash = error.sha256
        fields = (
            _field_reports(contact_problem)
            if contact_problem
            else blank_fields
        )
        payload = _base_payload(
            physical_status="PHYSICAL_CONTACT_DATA_NOT_AVAILABLE",
            validation_status="REJECTED",
            manifest_status="REJECTED",
            manifest_file=str(manifest_path),
            manifest_hash=manifest_hash,
            input_purpose=input_purpose,
            readiness=blank_readiness,
            fields=fields,
            reason=f"input manifest rejected: {error}",
            run_provenance=run_provenance,
        )
        status_path = _write_status_reports(output_directory, payload=payload)
        print(status_path)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
