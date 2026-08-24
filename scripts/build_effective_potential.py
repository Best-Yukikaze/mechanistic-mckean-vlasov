"""Build a validation-gated effective potential from an approved force table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from mechanistic_mv.mechanics.hydrogel import TEST_ONLY_NOT_CALIBRATED
from mechanistic_mv.mechanics.pair_interaction import PchipIntegratedForceLaw

try:
    from ._phase6_common import (
        FORCE_DERIVATIVE_ABSOLUTE_TOLERANCE_N,
        FORCE_DERIVATIVE_RELATIVE_TOLERANCE,
        FORCE_DERIVATIVE_STEP_FRACTION,
        PAIR_VALIDATION_DIRECTORY,
        PHASE6_SCHEMA_VERSION,
        effective_potential_from_table,
        load_pair_force_table,
        provenance,
        sha256_file,
        test_only_pair_force_table,
        test_only_scaling_provenance,
        test_only_short_range_provenance,
        write_json,
    )
except ImportError:  # direct ``python scripts/...`` execution
    from _phase6_common import (  # type: ignore[no-redef]
        FORCE_DERIVATIVE_ABSOLUTE_TOLERANCE_N,
        FORCE_DERIVATIVE_RELATIVE_TOLERANCE,
        FORCE_DERIVATIVE_STEP_FRACTION,
        PAIR_VALIDATION_DIRECTORY,
        PHASE6_SCHEMA_VERSION,
        effective_potential_from_table,
        load_pair_force_table,
        provenance,
        sha256_file,
        test_only_pair_force_table,
        test_only_scaling_provenance,
        test_only_short_range_provenance,
        write_json,
    )


VECTOR_SCALAR_RELATIVE_TOLERANCE = 1.0e-14


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-force-csv", type=Path)
    parser.add_argument("--input-metadata", type=Path)
    parser.add_argument("--test-only-fixture", action="store_true")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PAIR_VALIDATION_DIRECTORY,
    )
    return parser


def _median_runtime(callable_object, repeats: int = 9) -> float:
    durations = []
    for _ in range(repeats):
        start = perf_counter()
        callable_object()
        durations.append(perf_counter() - start)
    return float(np.median(durations))


def _array_digest(distance: np.ndarray, force: np.ndarray) -> str:
    values = np.column_stack((distance, force)).astype("<f8", copy=False)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _write_effective_csv(
    path: Path,
    distance: np.ndarray,
    force: np.ndarray,
    potential: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "center_distance_m",
                "radial_force_newton",
                "effective_potential_joule",
            )
        )
        writer.writerows(zip(distance, force, potential, strict=True))
    temporary.replace(path)


def _blocked(
    output_directory: Path,
    *,
    reason: str,
    missing_inputs: list[str],
    run_provenance: dict[str, object],
) -> Path:
    status = output_directory / "effective_potential_status.json"
    preexisting = [
        name
        for name in (
            "effective_potential.csv",
            "effective_potential_metadata.json",
            "TEST_ONLY_effective_potential.csv",
            "TEST_ONLY_effective_potential_metadata.json",
        )
        if (output_directory / name).exists()
    ]
    write_json(
        status,
        {
            "schema_name": "mechanistic_mv.effective_potential_status",
            "schema_version": PHASE6_SCHEMA_VERSION,
            "artifact_type": "EFFECTIVE_PAIR_POTENTIAL_STATUS",
            "workflow_status": "BLOCKED",
            "physical_status": "NOT_EVALUATED",
            "reason": reason,
            "missing_inputs": missing_inputs,
            "generated_data_files": [],
            "preexisting_untrusted_artifacts": preexisting,
            "kac_scaling_implemented_by_this_script": False,
            **run_provenance,
        },
    )
    return status


def _implementation_comparison(table, potential) -> dict[str, object]:
    law = PchipIntegratedForceLaw(table)
    nodes = table.center_distance_m
    pchip_seconds = _median_runtime(lambda: law.potential_joule(nodes))
    trapezoid_seconds = _median_runtime(
        law.potential_at_nodes_by_trapezoid_joule
    )
    pchip_values = law.potential_joule(nodes)
    trapezoid_values = law.potential_at_nodes_by_trapezoid_joule()
    pchip_trapezoid_difference = float(
        np.max(np.abs(pchip_values - trapezoid_values))
    )

    radius = np.linspace(0.0, table.metadata.reference_distance_m, 1024)
    angle = np.linspace(0.0, 2.0 * np.pi, radius.size, endpoint=False)
    displacement = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
    potential.potential_joule(displacement)
    potential.force_newton(displacement)
    vector_seconds = _median_runtime(
        lambda: (
            potential.potential_joule(displacement),
            potential.force_newton(displacement),
        )
    )
    scalar_seconds = _median_runtime(
        lambda: (
            np.asarray([potential.potential_joule(value) for value in displacement]),
            np.stack([potential.force_newton(value) for value in displacement]),
        ),
        repeats=5,
    )
    vector_energy = potential.potential_joule(displacement)
    vector_force = potential.force_newton(displacement)
    scalar_energy = np.asarray(
        [potential.potential_joule(value) for value in displacement]
    )
    scalar_force = np.stack(
        [potential.force_newton(value) for value in displacement]
    )
    energy_difference = float(np.max(np.abs(vector_energy - scalar_energy)))
    force_difference = float(np.max(np.abs(vector_force - scalar_force)))
    vector_scalar_equivalent = bool(
        np.allclose(
            vector_energy,
            scalar_energy,
            rtol=VECTOR_SCALAR_RELATIVE_TOLERANCE,
            atol=0.0,
        )
        and np.allclose(
            vector_force,
            scalar_force,
            rtol=VECTOR_SCALAR_RELATIVE_TOLERANCE,
            atol=0.0,
        )
    )
    return {
        "pchip_vs_trapezoid": {
            "timing_statistic": "median wall-clock seconds",
            "pchip_repeats": 9,
            "trapezoid_repeats": 9,
            "pchip_exact_antiderivative_seconds": pchip_seconds,
            "node_trapezoid_seconds": trapezoid_seconds,
            "maximum_difference_joule": pchip_trapezoid_difference,
            "implementation_equivalence_verdict": (
                "DIAGNOSTIC_ONLY_NO_INDEPENDENT_PHYSICAL_ERROR_BUDGET"
            ),
            "physical_equivalence_verdict": (
                "NOT_ESTABLISHED_NO_INDEPENDENT_PHYSICAL_ERROR_BUDGET"
            ),
            "selected_method": "shape-preserving PCHIP exact antiderivative",
        },
        "vectorized_vs_scalar": {
            "timing_statistic": "median wall-clock seconds after warm-up",
            "vectorized_repeats": 9,
            "scalar_repeats": 5,
            "evaluated_displacements": int(displacement.shape[0]),
            "vectorized_seconds": vector_seconds,
            "scalar_seconds": scalar_seconds,
            "scalar_over_vectorized_speed_ratio": (
                scalar_seconds / vector_seconds if vector_seconds > 0.0 else None
            ),
            "maximum_energy_difference_joule": energy_difference,
            "maximum_force_component_difference_newton": force_difference,
            "relative_tolerance": VECTOR_SCALAR_RELATIVE_TOLERANCE,
            "tolerance_source": "existing vector/scalar regression gate",
            "implementation_equivalence_verdict": (
                "EQUIVALENT_WITHIN_EXISTING_NUMERICAL_GATE"
                if vector_scalar_equivalent
                else "NOT_EQUIVALENT"
            ),
            "physical_equivalence_verdict": (
                "NOT_ESTABLISHED_IMPLEMENTATION_COMPARISON_ONLY"
            ),
            "passed": vector_scalar_equivalent,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_directory: Path = args.output_directory
    command = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    run_provenance = provenance(command)
    supplied_inputs = (
        args.input_force_csv is not None or args.input_metadata is not None
    )
    if args.test_only_fixture and supplied_inputs:
        status = _blocked(
            output_directory,
            reason="--test-only-fixture cannot be mixed with input artifacts",
            missing_inputs=[],
            run_provenance=run_provenance,
        )
        print(status)
        return 2
    if not args.test_only_fixture:
        missing = []
        if args.input_force_csv is None:
            missing.append("--input-force-csv")
        if args.input_metadata is None:
            missing.append("--input-metadata")
        if missing:
            status = _blocked(
                output_directory,
                reason="a validated force table and complete sidecar are required",
                missing_inputs=missing,
                run_provenance=run_provenance,
            )
            print(status)
            return 2

    try:
        if args.test_only_fixture:
            table = test_only_pair_force_table()
            input_metadata = {
                "artifact_id": table.metadata.dataset_id,
                "workflow_status": "TEST_ONLY_PASSED",
                "physical_status": TEST_ONLY_NOT_CALIBRATED,
                "calibration_status": TEST_ONLY_NOT_CALIBRATED,
                "calibration_id": TEST_ONLY_NOT_CALIBRATED,
                "source": table.metadata.source,
                "source_array_sha256": _array_digest(
                    table.center_distance_m, table.radial_force_newton
                ),
                "scaling_provenance": test_only_scaling_provenance(),
                "short_range_closure_provenance": (
                    test_only_short_range_provenance()
                ),
            }
            prefix = "TEST_ONLY_"
            workflow_status = "TEST_ONLY_PASSED"
            input_metadata_sha256 = None
        else:
            input_metadata_sha256 = sha256_file(args.input_metadata)
            table, input_metadata = load_pair_force_table(
                args.input_force_csv, args.input_metadata
            )
            if sha256_file(args.input_metadata) != input_metadata_sha256:
                raise ValueError(
                    "input metadata changed while it was being validated"
                )
            if table.metadata.physical_status == TEST_ONLY_NOT_CALIBRATED:
                raise ValueError(
                    "TEST_ONLY force tables require the explicit "
                    "--test-only-fixture path"
                )
            prefix = ""
            workflow_status = "PHYSICAL_INPUT_NUMERICAL_GATES_PASSED"
        potential = effective_potential_from_table(table)
    except (OSError, TypeError, ValueError) as error:
        status = _blocked(
            output_directory,
            reason=str(error),
            missing_inputs=[],
            run_provenance=run_provenance,
        )
        print(status)
        return 2

    comparison = _implementation_comparison(table, potential)
    overall_passed = bool(
        potential.validation_report.passed
        and comparison["vectorized_vs_scalar"]["passed"]
    )
    if not overall_passed:
        workflow_status = "FAILED"
    csv_path = output_directory / f"{prefix}effective_potential.csv"
    metadata_path = output_directory / f"{prefix}effective_potential_metadata.json"
    status_path = output_directory / "effective_potential_status.json"
    radial_potential = potential.radial_potential_joule(table.center_distance_m)
    _write_effective_csv(
        csv_path,
        table.center_distance_m,
        table.radial_force_newton,
        radial_potential,
    )
    report = {
        "schema_name": "mechanistic_mv.effective_pair_potential",
        "schema_version": PHASE6_SCHEMA_VERSION,
        "artifact_type": "EFFECTIVE_PAIR_POTENTIAL",
        "artifact_id": table.metadata.dataset_id,
        "workflow_status": workflow_status,
        "physical_status": table.metadata.physical_status,
        "calibration_status": input_metadata["calibration_status"],
        "calibration_id": input_metadata.get(
            "calibration_id", input_metadata["calibration_status"]
        ),
        "test_only_fixture": bool(args.test_only_fixture),
        "validation_scope": (
            "PAIR_FORCE_ARTIFACT_INTEGRITY_AND_IMPLEMENTATION_CONSISTENCY_ONLY"
        ),
        "overall_passed": overall_passed,
        "data_file": csv_path.name,
        "data_file_sha256": None,
        "source_artifacts": {
            "force_csv": (
                str(args.input_force_csv) if args.input_force_csv else None
            ),
            "force_metadata": (
                str(args.input_metadata) if args.input_metadata else None
            ),
            "force_metadata_sha256": input_metadata_sha256,
            "source_force_sha256": input_metadata.get(
                "data_file_sha256", input_metadata.get("source_array_sha256")
            ),
        },
        "density_normalization": "integral(rho dx)=1",
        "particle_pair_prefactor": "1/N",
        "kac_scaling_implemented_by_this_script": False,
        "input_must_already_be_kac_scaled": True,
        "scaling": table.metadata.scaling.value,
        "scaling_provenance": input_metadata["scaling_provenance"],
        "short_range_closure_provenance": input_metadata[
            "short_range_closure_provenance"
        ],
        "physical_provenance": input_metadata.get("physical_provenance"),
        "units": {
            "center_distance_m": "m",
            "radial_force_newton": "N",
            "effective_potential_joule": "J",
        },
        "interpolation_method": "PCHIP_SHAPE_PRESERVING",
        "integration_method": "PCHIP_EXACT_ANTIDERIVATIVE",
        "reference_condition": {
            "reference_distance_m": table.metadata.reference_distance_m,
            "potential_at_reference_joule": 0.0,
            "reference_force_tolerance_newton": (
                table.metadata.reference_force_tolerance_newton
            ),
        },
        "pair_force_metadata": {
            "dataset_id": table.metadata.dataset_id,
            "source": table.metadata.source,
            "physical_status": table.metadata.physical_status,
            "solver_status": table.metadata.solver_status,
            "validation_status": table.metadata.validation_status.value,
            "time_scale_status": table.metadata.time_scale_status.value,
            "scaling": table.metadata.scaling.value,
            "reference_distance_m": table.metadata.reference_distance_m,
            "reference_force_tolerance_newton": (
                table.metadata.reference_force_tolerance_newton
            ),
        },
        "force_potential_consistency": {
            "passed": potential.validation_report.passed,
            "evaluated_point_count": (
                potential.validation_report.evaluated_point_count
            ),
            "maximum_absolute_error_newton": (
                potential.validation_report.maximum_absolute_error_newton
            ),
            "maximum_relative_error": (
                potential.validation_report.maximum_relative_error
            ),
            "absolute_tolerance_newton": (
                FORCE_DERIVATIVE_ABSOLUTE_TOLERANCE_N
            ),
            "relative_tolerance": FORCE_DERIVATIVE_RELATIVE_TOLERANCE,
            "finite_difference_step_fraction": FORCE_DERIVATIVE_STEP_FRACTION,
            "evaluated_location": (
                "interpolation interval midpoints; endpoints excluded"
            ),
            "tolerance_source": (
                "existing PCHIP/float64 centered-difference regression gate; "
                "not FEM or experimental uncertainty"
            ),
        },
        "implementation_comparison": comparison,
        "scientific_limitations": [
            (
                "F=-dW/dr checks interpolation/integration consistency, "
                "not contact physics"
            ),
            "PCHIP-vs-trapezoid has no independent physical error budget",
            (
                "PHYSICAL input status means the supplied provenance passed "
                "its upstream gates; this script does not recalibrate material "
                "or contact physics"
            ),
        ],
        **run_provenance,
    }
    report["data_file_sha256"] = sha256_file(csv_path)
    write_json(metadata_path, report)
    write_json(
        status_path,
        {
            "schema_name": "mechanistic_mv.effective_potential_status",
            "schema_version": PHASE6_SCHEMA_VERSION,
            "artifact_type": "EFFECTIVE_PAIR_POTENTIAL_STATUS",
            "workflow_status": workflow_status,
            "physical_status": table.metadata.physical_status,
            "test_only_fixture": bool(args.test_only_fixture),
            "overall_passed": overall_passed,
            "kac_scaling_implemented_by_this_script": False,
            "metadata_file": metadata_path.name,
            "data_file": csv_path.name,
            **run_provenance,
        },
    )
    print(metadata_path)
    return 0 if overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
