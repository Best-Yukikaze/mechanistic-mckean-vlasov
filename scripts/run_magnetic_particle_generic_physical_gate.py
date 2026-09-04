"""Source-first, fail-closed admission audit for magnetic MV validation.

This program consumes the frozen external-source register, performs only
calculations that are admissible from that register, calls the Physics-owned
2-D closure guard, and writes phase-specific evidence.  It deliberately does
not fill missing magnetic, transport, observation, or thin-slab inputs and it
does not run a continuum simulation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from mechanistic_mv.mechanics.magnetic_particle.continuum_admission import (
    CURRENT_2D_MV_PHYSICAL_REDUCTION_INVALID,
    assess_magnetic_2d_closure,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROVENANCE = (
    ROOT
    / "data"
    / "external"
    / "magnetic_particle"
    / "generic__source_provenance__r1.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "validation" / "magnetic_particle"

PHASE_ARTIFACT_STEMS = {
    "phase_a": "generic__transport_closure__r1",
    "phase_b": "generic__magnetic_drift__r1",
    "phase_c": "generic__dipolar_interaction__r1",
    "phase_d": "generic__joint_mv__r1",
}
REPORT_ARTIFACT_STEM = "generic__physical_gate__r1"

SCHEMA = "mechanistic_mv.mv_physical_validation"
SCHEMA_VERSION = 1
FINAL_OUTCOMES = (
    "PHYSICAL_CONTROLLED_MV_VALIDATED",
    "PHYSICAL_FOKKER_PLANCK_VALIDATED_MV_INTERACTION_NOT_VALIDATED",
    "MAGNETIC_POTENTIAL_VALIDATED_CONSTANT_M_REJECTED",
    "CURRENT_2D_MV_PHYSICAL_REDUCTION_INVALID",
    "INSUFFICIENT_EXPERIMENTAL_DATA_FOR_PARAMETER_FREE_JOINT_VALIDATION",
)

REQUIRED_SCALAR_FIELDS = {
    "name",
    "value",
    "units",
    "paper",
    "table_figure_page_section",
    "particle_batch",
    "gel_condition",
    "value_class",
    "uncertainty",
    "transfer_justification",
    "admissibility",
}

S1_VELOCITY_NAMES = {
    "8nm": (
        "s1_8nm_LG_velocity",
        "s1_8nm_IG_velocity",
        "s1_8nm_HG_velocity",
    ),
    "11nm": (
        "s1_11nm_LG_velocity",
        "s1_11nm_IG_velocity",
        "s1_11nm_HG_velocity",
    ),
}


class ValidationInputError(ValueError):
    """A malformed or scientifically unsafe source register."""


def _reject_constant(value: str) -> None:
    raise ValidationInputError(f"non-finite JSON constant {value!r} is forbidden")


def _reject_duplicate(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationInputError(f"duplicate JSON key {key!r} is forbidden")
        result[key] = value
    return result


def load_provenance(path: Path) -> dict[str, object]:
    """Load and structurally validate the Experiment-owned source register."""

    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            payload = json.load(
                stream,
                parse_constant=_reject_constant,
                object_pairs_hook=_reject_duplicate,
            )
    except FileNotFoundError as error:
        raise ValidationInputError(f"provenance file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValidationInputError(f"provenance file is invalid JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValidationInputError("provenance root must be an object")
    if payload.get("schema") != "mechanistic_mv.mv_physical_validation.source_provenance":
        raise ValidationInputError("unexpected provenance schema")
    if payload.get("schema_version") != 1:
        raise ValidationInputError("unsupported provenance schema_version")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValidationInputError("sources must be a non-empty list")
    source_ids: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValidationInputError(f"sources[{index}] must be an object")
        for field in (
            "source_id",
            "paper",
            "primary_source_url",
            "access_status",
            "source_locator",
            "admissible_role",
            "limitation",
        ):
            _require_text(source.get(field), f"sources[{index}].{field}")
        source_id = str(source["source_id"])
        if source_id in source_ids:
            raise ValidationInputError(f"duplicate source_id {source_id!r}")
        source_ids.add(source_id)
    required_ids = {
        "S1_LYONS_2026",
        "S1B_LYONS_THESIS_2020",
        "S2_FATIN_ROUGE_2004",
        "S2B_MONCURE_2022",
        "S3_MYROVALI_2016",
        "S4_BASAK_2009",
    }
    if source_ids != required_ids:
        raise ValidationInputError(
            "source set mismatch: " + ", ".join(sorted(source_ids.symmetric_difference(required_ids)))
        )
    records = payload.get("scalar_records")
    if not isinstance(records, list) or not records:
        raise ValidationInputError("scalar_records must be a non-empty list")
    names: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValidationInputError(f"scalar_records[{index}] must be an object")
        missing = REQUIRED_SCALAR_FIELDS.difference(record)
        if missing:
            raise ValidationInputError(
                f"scalar_records[{index}] missing fields: {', '.join(sorted(missing))}"
            )
        unexpected = set(record).difference(REQUIRED_SCALAR_FIELDS)
        if unexpected:
            raise ValidationInputError(
                f"scalar_records[{index}] unexpected fields: {', '.join(sorted(unexpected))}"
            )
        name = _require_text(record["name"], f"scalar_records[{index}].name")
        if name in names:
            raise ValidationInputError(f"duplicate scalar name {name!r}")
        names.add(name)
        _validate_finite_value(record["value"], f"scalar_records[{index}].value")
        for field in (
            "units",
            "paper",
            "table_figure_page_section",
            "particle_batch",
            "gel_condition",
            "value_class",
            "transfer_justification",
            "admissibility",
        ):
            _require_text(record[field], f"scalar_records[{index}].{field}")
        uncertainty = record["uncertainty"]
        if uncertainty is not None:
            if not isinstance(uncertainty, dict):
                raise ValidationInputError(f"scalar_records[{index}].uncertainty must be null or object")
            if set(uncertainty) != {"value", "units", "kind"}:
                raise ValidationInputError(f"scalar_records[{index}].uncertainty schema mismatch")
            _validate_finite_value(uncertainty["value"], f"scalar_records[{index}].uncertainty.value")
            _require_text(uncertainty["units"], f"scalar_records[{index}].uncertainty.units")
            _require_text(uncertainty["kind"], f"scalar_records[{index}].uncertainty.kind")
    required_velocity_names = {item for values in S1_VELOCITY_NAMES.values() for item in values}
    if not required_velocity_names.issubset(names):
        raise ValidationInputError("S1 velocity records are incomplete")
    _validate_no_promoted_s1b_transfer(payload)
    _validate_s4_digitization_audit(payload)
    return payload


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationInputError(f"{label} must be non-empty text")
    return value.strip()


def _validate_finite_value(value: object, label: str) -> None:
    if isinstance(value, bool):
        raise ValidationInputError(f"{label} must be finite numeric data")
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValidationInputError(f"{label} must be finite")
        return
    if isinstance(value, list) and value:
        for index, item in enumerate(value):
            _validate_finite_value(item, f"{label}[{index}]")
        return
    raise ValidationInputError(f"{label} must be a finite number or non-empty numeric list")


def _validate_no_promoted_s1b_transfer(payload: Mapping[str, object]) -> None:
    checks = payload.get("transfer_checks")
    if not isinstance(checks, list):
        raise ValidationInputError("transfer_checks must be a list")
    by_id = {
        item.get("transfer_id"): item
        for item in checks
        if isinstance(item, dict) and isinstance(item.get("transfer_id"), str)
    }
    item = by_id.get("S1B_TO_S1_2026_FORCE")
    if not isinstance(item, dict) or item.get("status") != "REJECTED_MISSING_BATCH_AND_FIELD_COMPATIBILITY":
        raise ValidationInputError("S1b-to-S1 magnetic parameter transfer must remain explicitly rejected")


def _validate_s4_digitization_audit(payload: Mapping[str, object]) -> None:
    audit = payload.get("s4_digitization_audit")
    if not isinstance(audit, dict):
        raise ValidationInputError("s4_digitization_audit must be an object")
    if audit.get("admissible_as_particle_number_density") is not False:
        raise ValidationInputError("S4 normalized optical intensity cannot be promoted to rho")
    if audit.get("fabricated_numeric_records") != 0:
        raise ValidationInputError("S4 audit must not contain fabricated numeric records")
    if audit.get("raw_numeric_table_found") is not False:
        raise ValidationInputError("S4 raw numeric table status disagrees with the source audit")


def _scalar_map(payload: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    records = payload["scalar_records"]
    assert isinstance(records, list)
    return {str(record["name"]): record for record in records if isinstance(record, dict)}


def mm_per_hour_to_m_per_second(value: float) -> float:
    """Convert a source velocity in mm/h to SI m/s."""

    return float(value) * 1.0e-3 / 3600.0


def relative_population_cv(values: Sequence[float]) -> float:
    """Return population standard deviation divided by positive mean."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("CV requires at least two finite values")
    mean = float(np.mean(array))
    if mean <= 0.0:
        raise ValueError("CV requires a positive mean")
    return float(np.std(array, ddof=0) / mean)


def thesis_candidate_force_newton(payload: Mapping[str, object]) -> float:
    """Compute the S1b force only within its own thesis parameter context."""

    records = _scalar_map(payload)
    chi = float(records["s1b_chi_v"]["value"])
    volume = float(records["s1b_magnetic_material_volume"]["value"])
    field = float(records["s1b_flux_density"]["value"])
    gradient = float(records["s1b_flux_density_gradient"]["value"])
    mu0 = 4.0e-7 * math.pi
    return chi * volume * field * gradient / mu0


def build_phase_a(payload: Mapping[str, object]) -> dict[str, object]:
    records = _scalar_map(payload)
    observations: list[dict[str, object]] = []
    velocity_cv: dict[str, dict[str, object]] = {}
    for particle_class, names in S1_VELOCITY_NAMES.items():
        source_values: list[float] = []
        for condition, name in zip(("LG", "IG", "HG"), names, strict=True):
            record = records[name]
            velocity = float(record["value"])
            uncertainty = record["uncertainty"]
            uncertainty_value = None
            if isinstance(uncertainty, dict):
                uncertainty_value = float(uncertainty["value"])
            source_values.append(velocity)
            observations.append(
                {
                    "particle_class": particle_class,
                    "condition": condition,
                    "velocity_mm_per_h": velocity,
                    "velocity_uncertainty_mm_per_h": uncertainty_value,
                    "velocity_m_per_s": mm_per_hour_to_m_per_second(velocity),
                    "velocity_uncertainty_m_per_s": (
                        None
                        if uncertainty_value is None
                        else mm_per_hour_to_m_per_second(uncertainty_value)
                    ),
                    "force_newton": None,
                    "mobility_m_per_N_s": None,
                    "einstein_diffusivity_m2_per_s": None,
                    "status": "BLOCKED_SAME_CONDITION_MAGNETIC_FORCE_UNAVAILABLE",
                }
            )
        velocity_cv[particle_class] = {
            "value": relative_population_cv(source_values),
            "meaning": "DESCRIPTIVE_CV_OF_RAW_VELOCITY_NOT_CV_OF_MOBILITY",
            "constant_mobility_gate_applied": False,
        }
    candidate_force = thesis_candidate_force_newton(payload)
    return {
        "schema": f"{SCHEMA}.phase_a_transport_closure",
        "schema_version": SCHEMA_VERSION,
        "phase": "A",
        "status": "BLOCKED_SAME_BATCH_MAGNETIC_FORCE_UNAVAILABLE",
        "formulae": {
            "mobility": "M_eff = v_exp/F_mag",
            "einstein": "D_Einstein = M_eff*k_B*T",
            "free_solution": "D0 = k_B*T/(3*pi*eta*d_hyd)",
        },
        "acceptance_gate": {
            "CV_M_eff_le_0.25": "PASS_LINEAR_MOBILITY",
            "0.25_lt_CV_M_eff_le_0.40": "MARGINAL",
            "CV_M_eff_gt_0.40": "FAIL_LINEAR_MOBILITY",
            "result": "NOT_EVALUABLE",
        },
        "observations": observations,
        "descriptive_velocity_cv": velocity_cv,
        "s1b_candidate_force": {
            "value_newton": candidate_force,
            "formula": "chi_v*V_m*B*grad(B)/mu0",
            "scope": "S1B_THESIS_CONTEXT_ONLY",
            "transfer_to_s1_2026": "REJECTED_MISSING_BATCH_AND_FIELD_COMPATIBILITY",
        },
        "same_system_diffusivity": None,
        "stokes_einstein_check": "NOT_EVALUABLE_WITHOUT_ADMISSIBLE_D_AND_MATCHED_TEMPERATURE_VISCOSITY",
        "hindrance_evidence": {
            "S2": "INDEPENDENT_REGIME_EVIDENCE_ONLY",
            "S2b": "INDEPENDENT_REGIME_EVIDENCE_ONLY_POINTWISE_DIFFUSION_TABLE_NOT_ACQUIRED",
        },
        "simulation_run": False,
        "parameter_fit_performed": False,
        "blockers": [
            "S1 LG/IG/HG observations do not have condition-specific, batch-matched magnetic force tuples.",
            "S1b parameters cannot be transferred to S1 without batch, coating and field-geometry compatibility evidence.",
            "No same-system D is available for a Stokes-Einstein hindrance check.",
        ],
    }


def build_phase_b(phase_a: Mapping[str, object]) -> dict[str, object]:
    assert phase_a["status"] != "PASS_LINEAR_MOBILITY"
    return {
        "schema": f"{SCHEMA}.phase_b_magnetic_drift",
        "schema_version": SCHEMA_VERSION,
        "phase": "B",
        "status": "BLOCKED_PHASE_A_TRANSPORT_NOT_LOCKED",
        "magnetic_law": "V_mag=-chi_v*V_m*B^2/(2*mu0); F_mag=chi_v*V_m*B*grad(B)/mu0",
        "calibration_anchor": "8 nm PEG2000 LG",
        "anchor_mobility": None,
        "heldout_predictions": [],
        "heldout_MAE": None,
        "heldout_MAPE": None,
        "heldout_R2": None,
        "actual_pde_run": False,
        "mass_boundary_numerics": "NOT_RUN_DEPENDENCY_BLOCKED",
        "parameter_fit_performed": False,
        "blockers": [
            "Phase A did not produce an admissible locked D/M pair.",
            "Condition-specific S1 magnetic field/force data are unavailable.",
        ],
    }


def build_phase_c(payload: Mapping[str, object], closure_report: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema": f"{SCHEMA}.phase_c_dipolar_w",
        "schema_version": SCHEMA_VERSION,
        "phase": "C",
        "status": "BLOCKED_SOURCE_BACKED_MOMENT_CONTACT_AND_2D_CLOSURE_UNAVAILABLE",
        "interaction_law": (
            "W_dd(r,theta)=mu0*m^2*(1-3*cos(theta)^2)/(4*pi*r^3)"
        ),
        "source_regime_evidence": {
            "source": "S3_MYROVALI_2016",
            "field_tesla": 0.040,
            "particle_diameters_m": [1.0e-08, 4.0e-08],
            "mnp_concentrations_mg_per_mL": [1.0, 2.0, 4.0],
            "verified_ordering": [
                "40 nm particles show substantially stronger chain formation than 10 nm particles.",
                "For 40 nm simulations, chain length/density grows with concentration.",
                "Agarose content changes particle mobility and chain formation.",
            ],
            "role": "W_FORM_AND_REGIME_EVIDENCE_NOT_FIXED_GEL_QUANTITATIVE_VALIDATION",
        },
        "lambda_dd_10nm": None,
        "lambda_dd_40nm": None,
        "lambda_order_gate": "NOT_EVALUABLE",
        "continuum_metrics": None,
        "physics_2d_closure": dict(closure_report),
        "continuum_run": False,
        "free_interaction_multiplier_used": False,
        "blockers": [
            "No source-backed particle moment/magnetization law and physical contact distance are jointly available for S3 10 nm and 40 nm cases.",
            "Mass concentration cannot be converted to number density without admitted particle density/volume and aggregation state.",
            "The 3-D dipolar law has no source-backed depth-integrated 2-D closure in the current source register.",
        ],
    }


def build_phase_d(payload: Mapping[str, object], phase_a: Mapping[str, object], phase_b: Mapping[str, object], phase_c: Mapping[str, object]) -> dict[str, object]:
    audit = payload["s4_digitization_audit"]
    assert isinstance(audit, dict)
    return {
        "schema": f"{SCHEMA}.phase_d_joint_mv",
        "schema_version": SCHEMA_VERSION,
        "phase": "D",
        "status": "BLOCKED_NO_LOCKED_UPSTREAM_PARAMETERS_AND_NO_CALIBRATED_S4_OBSERVATION_MODEL",
        "source": "S4_BASAK_2009",
        "conditions": {
            "field_gradients_T_per_m": [0.0, 0.55, 8.0],
            "times_h": [0.0, 3.0, 6.0, 16.0, 32.0, 48.0],
            "particle_sizes_m": [1.30e-07, 2.50e-07],
            "agarose_fraction": 0.006,
        },
        "observation": {
            "published_quantity": "average normalized microscope pixel value",
            "particle_number_density_mapping": None,
            "digitization_audit": dict(audit),
            "digitized_records": [],
            "fabricated_records": 0,
        },
        "locked_parameter_transfer": {
            "D_M": phase_a["status"],
            "V_mag": phase_b["status"],
            "W_dd": phase_c["status"],
            "all_locked": False,
        },
        "ablations": {
            "D_only": "NOT_RUN",
            "D_plus_V_mag": "NOT_RUN",
            "D_plus_W_dd": "NOT_RUN",
            "D_plus_V_mag_plus_W_dd": "NOT_RUN",
        },
        "joint_predictions": [],
        "field_ordering_gate": "NOT_EVALUABLE",
        "quantitative_curve_gate": "NOT_EVALUABLE",
        "parameter_refit_performed": False,
        "simulation_run": False,
        "blockers": [
            "No independently locked D/M, V_mag and W_dd parameter set exists.",
            "S4 magnetic moment/susceptibility and exact cluster number density are unresolved.",
            "Normalized optical intensity is not calibrated to particle-number density and lacks pointwise uncertainty.",
        ],
    }


def build_report(payload: Mapping[str, object], provenance_path: Path) -> dict[str, object]:
    phase_a = build_phase_a(payload)
    closure = assess_magnetic_2d_closure(provenance_path).as_jsonable()
    phase_b = build_phase_b(phase_a)
    phase_c = build_phase_c(payload, closure)
    phase_d = build_phase_d(payload, phase_a, phase_b, phase_c)

    closure_status = closure["status"]
    if closure_status == CURRENT_2D_MV_PHYSICAL_REDUCTION_INVALID:
        final_decision = CURRENT_2D_MV_PHYSICAL_REDUCTION_INVALID
    else:  # pragma: no cover - current source record intentionally blocks closure
        final_decision = "INSUFFICIENT_EXPERIMENTAL_DATA_FOR_PARAMETER_FREE_JOINT_VALIDATION"
    if final_decision not in FINAL_OUTCOMES:  # pragma: no cover - defensive invariant
        raise RuntimeError("invalid final outcome")

    return {
        "schema": f"{SCHEMA}.report",
        "schema_version": SCHEMA_VERSION,
        "evidence_status": "COMPLETED_FAIL_CLOSED_SOURCE_AND_ADMISSION_AUDIT",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "repo_head": _git_head(),
        },
        "provenance": {
            "path": str(provenance_path.resolve()),
            "sha256": _sha256(provenance_path),
            "sha256_role": "OPTIONAL_ARCHIVAL_PROVENANCE_NOT_AN_ADMISSION_GATE",
            "source_count": len(payload["sources"]),
            "scalar_record_count": len(payload["scalar_records"]),
        },
        "physical_system": {
            "rho": "internal mobile magnetic-particle number density in one fixed hydrogel",
            "domain": "one fixed macroscopic gel with closed-gel no-flux baseline",
            "flux": "J=-D*grad(rho)-M*rho*grad(V_base+V_mag+W_dd*rho)",
            "physical_execution": "BLOCKED",
        },
        "phase_a": phase_a,
        "phase_b": phase_b,
        "phase_c": phase_c,
        "phase_d": phase_d,
        "numerics": {
            "magnetic_pde_run": False,
            "mass_conservation": "NOT_RUN_DEPENDENCY_BLOCKED",
            "nonnegativity": "NOT_RUN_DEPENDENCY_BLOCKED",
            "no_flux": "NOT_RUN_DEPENDENCY_BLOCKED",
            "grid_convergence": "NOT_RUN_DEPENDENCY_BLOCKED",
            "timestep_convergence": "NOT_RUN_DEPENDENCY_BLOCKED",
        },
        "joint_evidence_sufficiency": "BLOCKED_SAME_BATCH_AND_OBSERVATION_DATA_GAPS",
        "final_decision": final_decision,
        "final_decision_is_exclusive": True,
        "failure_cause": [
            "The Physics-owned 2-D closure guard rejects the actual frozen source record because slab depth, vertical distribution, physical contact/cell closure, transport D/M and magnetic particle law are unresolved.",
            "S1 supplies drift observations but not same-batch condition-specific magnetic forces, so constant mobility cannot be tested or locked.",
            "S4 supplies a qualitative optical proxy without a calibrated rho observation model, so no parameter-free joint comparison is currently admissible.",
        ],
        "next_milestone": (
            "Obtain one auditable same-batch S1 data package containing condition-specific B and grad(B), magnetic moment/susceptibility and material volume with uncertainty; do not proceed to PDE validation before that package passes provenance review."
        ),
        "scope": {
            "parameter_fit_performed": False,
            "magnetic_simulation_run": False,
            "gym_rl_dqn_run": False,
            "training_run": False,
            "old_D8_evidence_modified": False,
        },
    }


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def write_outputs(report: Mapping[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    phases = {
        PHASE_ARTIFACT_STEMS["phase_a"]: report["phase_a"],
        PHASE_ARTIFACT_STEMS["phase_b"]: report["phase_b"],
        PHASE_ARTIFACT_STEMS["phase_c"]: report["phase_c"],
        PHASE_ARTIFACT_STEMS["phase_d"]: report["phase_d"],
    }
    for stem, phase in phases.items():
        assert isinstance(phase, Mapping)
        _write_json(output_dir / f"{stem}.json", phase)
        _write_markdown(output_dir / f"{stem}.md", _phase_markdown(phase))
        _write_phase_csv(output_dir / f"{stem}.csv", phase)
    _write_json(output_dir / f"{REPORT_ARTIFACT_STEM}.json", report)
    _write_markdown(output_dir / f"{REPORT_ARTIFACT_STEM}.md", _report_markdown(report))


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_markdown(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_phase_csv(path: Path, phase: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        if phase.get("phase") == "A":
            fields = (
                "particle_class",
                "condition",
                "velocity_mm_per_h",
                "velocity_uncertainty_mm_per_h",
                "velocity_m_per_s",
                "velocity_uncertainty_m_per_s",
                "force_newton",
                "mobility_m_per_N_s",
                "einstein_diffusivity_m2_per_s",
                "status",
            )
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            observations = phase.get("observations", [])
            assert isinstance(observations, list)
            writer.writerows(observations)
        else:
            writer = csv.DictWriter(
                stream,
                fieldnames=("phase", "status", "numeric_records", "simulation_run"),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "phase": phase.get("phase"),
                    "status": phase.get("status"),
                    "numeric_records": 0,
                    "simulation_run": False,
                }
            )


def _phase_markdown(phase: Mapping[str, object]) -> str:
    lines = [
        f"# Magnetic-particle generic physical gate — Phase {phase['phase']}",
        "",
        f"Status: **{phase['status']}**",
        "",
        "This is a fail-closed physical-admission record. Missing values remain null; no",
        "parameter fitting or magnetic continuum simulation was performed.",
    ]
    blockers = phase.get("blockers")
    if isinstance(blockers, list):
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {item}" for item in blockers)
    return "\n".join(lines)


def _report_markdown(report: Mapping[str, object]) -> str:
    phase_a = report["phase_a"]
    phase_c = report["phase_c"]
    assert isinstance(phase_a, Mapping)
    assert isinstance(phase_c, Mapping)
    velocity_cv = phase_a["descriptive_velocity_cv"]
    assert isinstance(velocity_cv, Mapping)
    return f"""# Magnetic-particle generic physical gate — r1

Final decision: **{report['final_decision']}**

## Physical system

`rho` is the internal mobile magnetic-particle number density in one fixed
macroscopic hydrogel. The target flux is
`J=-D grad(rho)-M rho grad(V_base+V_mag+W_dd*rho)` with a closed-gel no-flux
baseline.

## D/M

Phase A: **{phase_a['status']}**. Six S1 velocities were converted to SI, but
their condition-specific magnetic forces are absent. The descriptive raw
velocity CV values are {velocity_cv['8nm']['value']:.6f} (8 nm) and
{velocity_cv['11nm']['value']:.6f} (11 nm); these are not mobility CVs and the
constant-M gate was not applied.

## V_control

The source-supported candidate law is
`V_mag=-chi_v V_m B^2/(2 mu0)`. Phase B is blocked because no admissible D/M
pair or LG anchor mobility can be locked. No held-out prediction or PDE was
run.

## W

The target is the anisotropic dipolar energy. S3 supports the qualitative
10 nm versus 40 nm and concentration ordering, but `lambda_dd` and the 2-D
continuum response are not computable without source-backed moments, contact
distance, number density and slab closure. Physics closure status:
**{phase_c['physics_2d_closure']['status']}**.

## Joint MV

S4 verifies the zero/0.55/8 T/m and 0–48 h experimental design. Its Figure 7
observable is normalized optical intensity, not calibrated `rho`. No upstream
parameter set is locked and no no-refit joint run or ablation was performed.

## Numerics

Mass, non-negativity, no-flux, grid and timestep checks are not applicable yet
because the source/closure guard stopped execution before any magnetic PDE.

## Failure cause and next milestone

The current 2-D reduction is not physically admitted, while the same-batch S1
force data and S4 observation model are also insufficient. The one next
milestone is to obtain a traceable same-batch S1 magnetic data package with
condition-specific `B`, `grad(B)`, moment/susceptibility, material volume and
uncertainty. No Gym, RL, DQN or training was run.
"""


def _blocked_error_report(error: Exception, provenance_path: Path) -> dict[str, object]:
    return {
        "schema": f"{SCHEMA}.report",
        "schema_version": SCHEMA_VERSION,
        "evidence_status": "BLOCKED_INVALID_SOURCE_PROVENANCE",
        "provenance_path": str(provenance_path),
        "error": str(error),
        "final_decision": None,
        "final_decision_status": "NOT_ISSUED_INVALID_INPUT",
        "physical_execution": "NOT_RUN",
        "scope": {
            "parameter_fit_performed": False,
            "magnetic_simulation_run": False,
            "gym_rl_dqn_run": False,
            "training_run": False,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = load_provenance(args.provenance)
        report = build_report(payload, args.provenance)
    except (OSError, TypeError, ValueError, ValidationInputError) as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        blocked = _blocked_error_report(error, args.provenance)
        _write_json(args.output_dir / f"{REPORT_ARTIFACT_STEM}.json", blocked)
        _write_markdown(
            args.output_dir / f"{REPORT_ARTIFACT_STEM}.md",
            "# Magnetic-particle generic physical gate — r1\n\n"
            f"Status: **BLOCKED_INVALID_SOURCE_PROVENANCE**\n\n{error}",
        )
        print(json.dumps(blocked, indent=2, ensure_ascii=False))
        return 2
    write_outputs(report, args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
