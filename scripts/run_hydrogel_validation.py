"""Validate implemented Hydrogel identities without calibration claims."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from mechanistic_mv.mechanics.hydrogel import (
    HydrogelParameters,
    TEST_ONLY_NOT_CALIBRATED,
    dimensionless_gibbs_free_energy_density,
    first_piola_stress_pa,
    gibbs_conjugate_solvent_content_dimensionless,
    gibbs_free_energy_density_pa,
)

try:
    from ._phase6_common import (
        PHASE6_SCHEMA_VERSION,
        VALIDATION_DIRECTORY,
        provenance,
        test_only_hydrogel_parameters,
        write_json,
    )
except ImportError:  # direct ``python scripts/...`` execution
    from _phase6_common import (  # type: ignore[no-redef]
        PHASE6_SCHEMA_VERSION,
        VALIDATION_DIRECTORY,
        provenance,
        test_only_hydrogel_parameters,
        write_json,
    )


STRESS_FINITE_DIFFERENCE_STEP = 1.0e-6
STRESS_RELATIVE_TOLERANCE = 2.0e-8
STRESS_ABSOLUTE_TOLERANCE_PA = 2.0e-5
CONJUGACY_FINITE_DIFFERENCE_STEP = 1.0e-6
CONJUGACY_RELATIVE_TOLERANCE = 2.0e-10
CONJUGACY_ABSOLUTE_TOLERANCE = 1.0e-11


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-only-fixture", action="store_true")
    parser.add_argument("--network-density-times-solvent-volume", type=float)
    parser.add_argument("--flory-huggins-chi", type=float)
    parser.add_argument("--initial-polymer-volume-fraction", type=float)
    parser.add_argument("--delta-chemical-potential-over-kbt", type=float)
    parser.add_argument("--thermal-energy-density-pa", type=float)
    parser.add_argument("--calibration-status")
    parser.add_argument("--parameter-source")
    parser.add_argument("--output", type=Path)
    return parser


def _explicit_parameter_values(args: argparse.Namespace) -> dict[str, object]:
    return {
        "network_density_times_solvent_volume": (
            args.network_density_times_solvent_volume
        ),
        "flory_huggins_chi": args.flory_huggins_chi,
        "initial_polymer_volume_fraction": args.initial_polymer_volume_fraction,
        "delta_chemical_potential_over_kbt": (
            args.delta_chemical_potential_over_kbt
        ),
        "thermal_energy_density_pa": args.thermal_energy_density_pa,
        "calibration_status": args.calibration_status,
        "parameter_source": args.parameter_source,
    }


def _default_output(test_only: bool) -> Path:
    filename = (
        "TEST_ONLY_hydrogel_validation.json"
        if test_only
        else "hydrogel_validation.json"
    )
    return VALIDATION_DIRECTORY / "hydrogel" / filename


def _blocked_report(
    output: Path,
    *,
    missing: list[str],
    reason: str,
    run_provenance: dict[str, object],
) -> None:
    write_json(
        output,
        {
            "schema_name": "mechanistic_mv.hydrogel_constitutive_validation",
            "schema_version": PHASE6_SCHEMA_VERSION,
            "artifact_type": "HYDROGEL_CONSTITUTIVE_VALIDATION",
            "workflow_status": "BLOCKED",
            "physical_status": "NOT_EVALUATED",
            "validation_scope": "CONSTITUTIVE_IDENTITY_ONLY",
            "reason": reason,
            "missing_inputs": missing,
            "generated_data_files": [],
            **run_provenance,
        },
    )


def _finite_difference_stress(
    deformation: np.ndarray, parameters: HydrogelParameters
) -> np.ndarray:
    result = np.empty((3, 3), dtype=np.float64)
    for row in range(3):
        for column in range(3):
            offset = np.zeros((3, 3), dtype=np.float64)
            offset[row, column] = STRESS_FINITE_DIFFERENCE_STEP
            result[row, column] = (
                gibbs_free_energy_density_pa(deformation + offset, parameters)
                - gibbs_free_energy_density_pa(deformation - offset, parameters)
            ) / (2.0 * STRESS_FINITE_DIFFERENCE_STEP)
    return result


def _chemical_conjugacy_finite_difference(
    deformation: np.ndarray, parameters: HydrogelParameters
) -> float:
    plus = HydrogelParameters(
        parameters.network_density_times_solvent_volume,
        parameters.flory_huggins_chi,
        parameters.initial_polymer_volume_fraction,
        parameters.delta_chemical_potential_over_kbt
        + CONJUGACY_FINITE_DIFFERENCE_STEP,
        parameters.thermal_energy_density_pa,
        parameters.calibration_status,
    )
    minus = HydrogelParameters(
        parameters.network_density_times_solvent_volume,
        parameters.flory_huggins_chi,
        parameters.initial_polymer_volume_fraction,
        parameters.delta_chemical_potential_over_kbt
        - CONJUGACY_FINITE_DIFFERENCE_STEP,
        parameters.thermal_energy_density_pa,
        parameters.calibration_status,
    )
    return float(
        -(
            dimensionless_gibbs_free_energy_density(deformation, plus)
            - dimensionless_gibbs_free_energy_density(deformation, minus)
        )
        / (2.0 * CONJUGACY_FINITE_DIFFERENCE_STEP)
    )


def _error_summary(
    actual: np.ndarray,
    reference: np.ndarray,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, object]:
    error = np.abs(actual - reference)
    scale = np.maximum(np.abs(reference), np.finfo(np.float64).tiny)
    allowed = absolute_tolerance + relative_tolerance * np.abs(reference)
    return {
        "passed": bool(np.all(error <= allowed)),
        "maximum_absolute_error": float(np.max(error)),
        "maximum_relative_error": float(np.max(error / scale)),
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "tolerance_source": (
            "existing float64 centered-finite-difference regression budget; "
            "not material calibration uncertainty"
        ),
        "evaluated_value_count": int(error.size),
    }


def _median_runtime(callable_object, repeats: int) -> float:
    durations = []
    for _ in range(repeats):
        start = perf_counter()
        callable_object()
        durations.append(perf_counter() - start)
    return float(np.median(durations))


def validate(parameters: HydrogelParameters) -> dict[str, object]:
    deformations = np.asarray(
        [
            np.eye(3),
            [[1.08, 0.04, 0.00], [0.01, 1.03, 0.02], [0.00, 0.01, 1.02]],
            [[1.05, 0.06, 0.00], [0.00, 1.02, 0.03], [0.01, 0.00, 1.04]],
        ],
        dtype=np.float64,
    )
    analytic_stress = np.stack(
        [first_piola_stress_pa(value, parameters) for value in deformations]
    )
    finite_difference_stress = np.stack(
        [_finite_difference_stress(value, parameters) for value in deformations]
    )
    stress_check = _error_summary(
        analytic_stress,
        finite_difference_stress,
        absolute_tolerance=STRESS_ABSOLUTE_TOLERANCE_PA,
        relative_tolerance=STRESS_RELATIVE_TOLERANCE,
    )
    stress_check.update(
        {
            "identity": "P_pa = partial G_pa / partial F",
            "unit": "Pa",
            "finite_difference_scheme": "second-order centered",
            "finite_difference_step_in_dimensionless_F": (
                STRESS_FINITE_DIFFERENCE_STEP
            ),
        }
    )

    analytic_conjugacy = np.asarray(
        [
            gibbs_conjugate_solvent_content_dimensionless(value, parameters)
            for value in deformations
        ],
        dtype=np.float64,
    )
    finite_difference_conjugacy = np.asarray(
        [
            _chemical_conjugacy_finite_difference(value, parameters)
            for value in deformations
        ],
        dtype=np.float64,
    )
    conjugacy_check = _error_summary(
        analytic_conjugacy,
        finite_difference_conjugacy,
        absolute_tolerance=CONJUGACY_ABSOLUTE_TOLERANCE,
        relative_tolerance=CONJUGACY_RELATIVE_TOLERANCE,
    )
    conjugacy_check.update(
        {
            "identity": "-partial DeltaG/partial(delta_mu_over_kBT) = det(F)-phi0",
            "unit": "1",
            "finite_difference_scheme": "second-order centered",
            "finite_difference_step": CONJUGACY_FINITE_DIFFERENCE_STEP,
            "reference_volume_note": (
                "initial-swollen Gibbs conjugate; not independent dPsi/dC"
            ),
        }
    )

    analytic_seconds = _median_runtime(
        lambda: first_piola_stress_pa(deformations, parameters), 21
    )
    finite_difference_seconds = _median_runtime(
        lambda: np.stack(
            [_finite_difference_stress(value, parameters) for value in deformations]
        ),
        7,
    )
    return {
        "deformation_cases": deformations.tolist(),
        "checks": {
            "first_piola_is_gibbs_gradient": stress_check,
            "gibbs_chemical_conjugacy": conjugacy_check,
        },
        "implementation_comparison": {
            "comparison": "analytic stress vs centered finite difference",
            "timing_statistic": "median wall-clock seconds",
            "analytic_repeats": 21,
            "finite_difference_repeats": 7,
            "analytic_vectorized_seconds": analytic_seconds,
            "finite_difference_seconds": finite_difference_seconds,
            "finite_difference_over_analytic_speed_ratio": (
                finite_difference_seconds / analytic_seconds
                if analytic_seconds > 0.0
                else None
            ),
            "maximum_absolute_difference_pa": stress_check[
                "maximum_absolute_error"
            ],
            "maximum_relative_difference": stress_check[
                "maximum_relative_error"
            ],
            "implementation_equivalence_verdict": (
                "EQUIVALENT_WITHIN_EXISTING_NUMERICAL_GATE"
                if stress_check["passed"]
                else "NOT_EQUIVALENT"
            ),
            "physical_equivalence_verdict": (
                "NOT_ESTABLISHED_CONSTITUTIVE_IDENTITY_CHECK_ONLY"
            ),
        },
        "unsupported_claims": {
            "independent_helmholtz_Psi_of_F_and_C": "NOT_IMPLEMENTED_BY_SOURCE",
            "mu_equals_partial_Psi_partial_C": "NOT_EVALUATED",
        },
        "overall_passed": bool(stress_check["passed"] and conjugacy_check["passed"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    explicit = _explicit_parameter_values(args)
    supplied = [name for name, value in explicit.items() if value is not None]
    output = args.output or _default_output(args.test_only_fixture)
    command = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    run_provenance = provenance(command)

    if args.test_only_fixture and supplied:
        _blocked_report(
            output,
            missing=[],
            reason="--test-only-fixture cannot be mixed with explicit parameters",
            run_provenance=run_provenance,
        )
        print(output)
        return 2
    if not args.test_only_fixture:
        missing = [name for name, value in explicit.items() if value is None]
        if missing:
            _blocked_report(
                output,
                missing=missing,
                reason="real/test parameters have no defaults and must be explicit",
                run_provenance=run_provenance,
            )
            print(output)
            return 2
        calibration_status = str(args.calibration_status).strip()
        parameter_source = str(args.parameter_source).strip()
        normalized_calibration = (
            calibration_status.casefold().replace("-", "_").replace(" ", "_")
        )
        nonphysical_tokens = (
            "test_only",
            "placeholder",
            "not_calibrated",
            "unverified",
            "pending",
            "unknown",
            "tbd",
            "todo",
            "unspecified",
        )
        if (
            not normalized_calibration.startswith("calibrated_")
            or len(normalized_calibration) <= len("calibrated_")
            or any(token in normalized_calibration for token in nonphysical_tokens)
        ):
            _blocked_report(
                output,
                missing=["--test-only-fixture"],
                reason=(
                    "nonphysical/test-only calibration labels require the "
                    "explicit --test-only-fixture path"
                ),
                run_provenance=run_provenance,
            )
            print(output)
            return 2
        normalized_source = (
            parameter_source.casefold().replace("-", "_").replace(" ", "_")
        )
        if not normalized_source or any(
            token in normalized_source for token in nonphysical_tokens
        ):
            _blocked_report(
                output,
                missing=["--calibration-status", "--parameter-source"],
                reason="calibration status and parameter source must be specific",
                run_provenance=run_provenance,
            )
            print(output)
            return 2
        explicit_fixture_values = (
            args.network_density_times_solvent_volume == 0.025
            and args.flory_huggins_chi == 0.31
            and args.initial_polymer_volume_fraction == 0.18
            and args.delta_chemical_potential_over_kbt == 0.12
            and args.thermal_energy_density_pa == 2.4e5
        )
        if explicit_fixture_values:
            _blocked_report(
                output,
                missing=["--test-only-fixture"],
                reason=(
                    "the reserved numerical fixture values require the explicit "
                    "--test-only-fixture path"
                ),
                run_provenance=run_provenance,
            )
            print(output)
            return 2

    if args.test_only_fixture:
        parameters = test_only_hydrogel_parameters()
        parameter_source = "explicit --test-only-fixture values from unit tests"
        workflow_status = "TEST_ONLY_PASSED"
    else:
        try:
            parameters = HydrogelParameters(
                network_density_times_solvent_volume=(
                    args.network_density_times_solvent_volume
                ),
                flory_huggins_chi=args.flory_huggins_chi,
                initial_polymer_volume_fraction=(
                    args.initial_polymer_volume_fraction
                ),
                delta_chemical_potential_over_kbt=(
                    args.delta_chemical_potential_over_kbt
                ),
                thermal_energy_density_pa=args.thermal_energy_density_pa,
                calibration_status=calibration_status,
            )
        except (TypeError, ValueError) as error:
            _blocked_report(
                output,
                missing=[],
                reason=f"invalid explicit Hydrogel parameters: {error}",
                run_provenance=run_provenance,
            )
            print(output)
            return 2
        workflow_status = "CONSTITUTIVE_IDENTITY_CHECK_PASSED"

    try:
        results = validate(parameters)
    except (FloatingPointError, TypeError, ValueError) as error:
        _blocked_report(
            output,
            missing=[],
            reason=f"Hydrogel identity evaluation failed: {error}",
            run_provenance=run_provenance,
        )
        print(output)
        return 2
    if not results["overall_passed"]:
        workflow_status = "FAILED"
    report = {
        "schema_name": "mechanistic_mv.hydrogel_constitutive_validation",
        "schema_version": PHASE6_SCHEMA_VERSION,
        "artifact_type": "HYDROGEL_CONSTITUTIVE_VALIDATION",
        "workflow_status": workflow_status,
        "physical_status": parameters.calibration_status,
        "calibration_status": parameters.calibration_status,
        "test_only_fixture": bool(args.test_only_fixture),
        "validation_scope": (
            "CONSTITUTIVE_IDENTITY_ONLY_NOT_MATERIAL_CALIBRATION"
        ),
        "model": "HongGuoModelII constrained Gibbs formulation",
        "parameter_provenance": {
            "source": parameter_source,
            "network_density_times_solvent_volume": {
                "value": parameters.network_density_times_solvent_volume,
                "unit": "1",
            },
            "flory_huggins_chi": {
                "value": parameters.flory_huggins_chi,
                "unit": "1",
            },
            "initial_polymer_volume_fraction": {
                "value": parameters.initial_polymer_volume_fraction,
                "unit": "1",
            },
            "delta_chemical_potential_over_kbt": {
                "value": parameters.delta_chemical_potential_over_kbt,
                "unit": "1",
            },
            "thermal_energy_density_pa": {
                "value": parameters.thermal_energy_density_pa,
                "unit": "Pa",
            },
            "calibration_status": parameters.calibration_status,
        },
        **results,
        **run_provenance,
    }
    write_json(output, report)
    print(output)
    return 0 if results["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
