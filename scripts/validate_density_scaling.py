"""Validate probability/number-density scaling with TEST_ONLY fixtures.

This experiment measures the public Physics Engine contract.  It does not
modify an equation, calibrate Hydrogel mechanics, or promote the Gaussian
regression potential to a physical model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from mechanistic_mv.continuum.convolution import (
    FFTPairConvolver,
    direct_pair_convolution_joule,
)
from mechanistic_mv.continuum.initial_conditions import gaussian_density
from mechanistic_mv.continuum.mckean_vlasov import McKeanVlasovSolver
from mechanistic_mv.mechanics.density_scaling import DensityConvention
from mechanistic_mv.mechanics.energies import discrete_particle_energy_joule
from mechanistic_mv.mechanics.geometry import CartesianGrid, RectangularDomain
from mechanistic_mv.mechanics.hydrogel import (
    TEST_ONLY_NOT_CALIBRATED,
    TimeScaleStatus,
)
from mechanistic_mv.mechanics.pair_interaction import (
    HydrogelEffectivePairPotential,
    PairDataValidationStatus,
    PairForceMetadata,
    PairForceScaling,
    PairForceTable,
    convert_single_pair_table_to_kac,
)
from mechanistic_mv.mechanics.pair_potential import (
    TestOnlyGaussianRepulsion,
    mean_field_pair_force_newton,
)
from mechanistic_mv.mechanics.parameters import PhysicalParameters
from mechanistic_mv.mechanics.particle_model import ParticleMechanicalState
from mechanistic_mv.particle_sim.empirical_density import empirical_density
from mechanistic_mv.particle_sim.langevin_interacting import (
    overdamped_langevin_step,
)

try:
    from ._phase6_common import (
        PHASE6_SCHEMA_VERSION,
        VALIDATION_DIRECTORY,
        effective_potential_from_table,
        provenance,
        short_range_admission,
        write_json,
    )
except ImportError:  # direct ``python scripts/...`` execution
    from _phase6_common import (  # type: ignore[no-redef]
        PHASE6_SCHEMA_VERSION,
        VALIDATION_DIRECTORY,
        effective_potential_from_table,
        provenance,
        short_range_admission,
        write_json,
    )


TEST_ONLY_NOT_FINAL_PHYSICS = "TEST_ONLY_NOT_FINAL_PHYSICS"
DEFAULT_OUTPUT = (
    VALIDATION_DIRECTORY
    / "density_scaling"
    / "TEST_ONLY_density_scaling_validation.json"
)
DEFAULT_SEED = 20260831
POPULATION_COUNT = 6

TOLERANCE_SOURCE = (
    "pre-registered independent Experiment Lab float64 error budget; "
    "relative limits are small multiples of machine precision except the "
    "Direct/FFT transform gate, absolute floors protect near-zero SI values, "
    "and mass uses 64*eps*N; no gate is looser than the existing Physics "
    "regression limits; not material, FEM, or experimental uncertainty"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-only-fixture", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _maximum_relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = np.maximum(
        np.maximum(np.abs(actual), np.abs(expected)),
        np.finfo(np.float64).tiny,
    )
    return float(np.max(np.abs(actual - expected) / denominator))


def _equivalence_check(
    actual: np.ndarray | float,
    expected: np.ndarray | float,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
    unit: str,
    formula: str,
) -> dict[str, object]:
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    if actual_array.shape != expected_array.shape:
        raise ValueError("equivalence arrays must have matching shapes")
    if not np.all(np.isfinite(actual_array)) or not np.all(
        np.isfinite(expected_array)
    ):
        raise ValueError("equivalence arrays must be finite")
    absolute_error = np.abs(actual_array - expected_array)
    allowed = absolute_tolerance + relative_tolerance * np.abs(expected_array)
    report: dict[str, object] = {
        "passed": bool(np.all(absolute_error <= allowed)),
        "formula": formula,
        "maximum_absolute_error": float(np.max(absolute_error)),
        "maximum_relative_error": _maximum_relative_error(
            actual_array, expected_array
        ),
        "maximum_allowed_absolute_error": float(np.max(allowed)),
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "unit": unit,
        "evaluated_value_count": int(actual_array.size),
        "tolerance_source": TOLERANCE_SOURCE,
    }
    if actual_array.size == 1:
        report["observed_value"] = float(actual_array.reshape(-1)[0])
        report["reference_value"] = float(expected_array.reshape(-1)[0])
    return report


def _boolean_check(
    passed: bool,
    *,
    observed: object,
    required: object,
    formula: str,
) -> dict[str, object]:
    return {
        "passed": bool(passed),
        "formula": formula,
        "observed": observed,
        "required": required,
    }


def _expected_rejection(
    callable_object: Callable[[], object],
    *,
    expected_message: str,
    requirement: str,
) -> dict[str, object]:
    try:
        callable_object()
    except (TypeError, ValueError) as error:
        message = str(error)
        return {
            "passed": expected_message in message,
            "expected_outcome": "REJECTED",
            "observed_outcome": "REJECTED",
            "exception_type": type(error).__name__,
            "reason": message,
            "expected_message_fragment": expected_message,
            "requirement": requirement,
        }
    return {
        "passed": False,
        "expected_outcome": "REJECTED",
        "observed_outcome": "ACCEPTED",
        "exception_type": None,
        "reason": "the invalid contract was accepted",
        "expected_message_fragment": expected_message,
        "requirement": requirement,
    }


def _test_only_single_pair_table(
    *, minimum_distance_m: float = 0.0
) -> PairForceTable:
    reference_distance_m = 4.3e-6
    distance = np.linspace(minimum_distance_m, reference_distance_m, 19)
    normalized = distance / reference_distance_m
    force = 1.1e-12 * normalized * (1.0 - normalized) ** 3
    metadata = PairForceMetadata(
        dataset_id="TEST_ONLY_NOT_FINAL_PHYSICS_single_pair_scaling_fixture",
        source=(
            "analytic TEST_ONLY single-pair density-scaling fixture; "
            "not Hydrogel contact data"
        ),
        physical_status=TEST_ONLY_NOT_CALIBRATED,
        solver_status="TEST_ONLY_NOT_A_CONTACT_SOLVER",
        validation_status=PairDataValidationStatus.PASSED,
        time_scale_status=TimeScaleStatus.UNVERIFIED,
        scaling=PairForceScaling.UNSCALED_SINGLE_PAIR,
        reference_distance_m=reference_distance_m,
        reference_force_tolerance_newton=0.0,
    )
    return PairForceTable(distance, force, metadata)


def _hydrogel_test_potential(
    table: PairForceTable,
    density_convention: DensityConvention,
) -> HydrogelEffectivePairPotential:
    return HydrogelEffectivePairPotential(
        table,
        derivative_absolute_tolerance_newton=1.0e-20,
        derivative_relative_tolerance=2.0e-7,
        finite_difference_step_fraction=1.0e-4,
        density_convention=density_convention,
    )


def _stable_signature(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_validation(seed: int = DEFAULT_SEED) -> dict[str, object]:
    """Return deterministic scaling evidence without run-time provenance."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    population = POPULATION_COUNT
    parameters = PhysicalParameters(
        particle_mass_kg=1.3e-15,
        drag_coefficient_kg_per_s=1.2e-8,
        temperature_kelvin=301.2,
    )
    domain = RectangularDomain((0.0, 7.3e-6), (0.0, 4.7e-6))
    grid = CartesianGrid(domain, 7, 6)
    pair_energy = 0.37 * parameters.thermal_energy_joule
    pair_length = 0.93e-6
    pair_potential = TestOnlyGaussianRepulsion(
        pair_energy,
        pair_length,
        density_convention=DensityConvention.NUMBER,
    )
    kac_potential = TestOnlyGaussianRepulsion(
        population * pair_energy,
        pair_length,
        density_convention=DensityConvention.PROBABILITY,
    )
    positions = np.asarray(
        [
            [0.9e-6, 0.8e-6],
            [2.05e-6, 1.25e-6],
            [3.4e-6, 0.95e-6],
            [1.55e-6, 2.85e-6],
            [4.65e-6, 3.3e-6],
            [6.25e-6, 3.95e-6],
        ],
        dtype=np.float64,
    )

    checks: dict[str, dict[str, object]] = {}
    empirical_probability = empirical_density(positions, grid)
    empirical_number = empirical_density(
        positions,
        grid,
        density_convention=DensityConvention.NUMBER,
    )
    initial_probability = gaussian_density(
        grid,
        (3.4e-6, 2.18e-6),
        0.73e-6,
    )
    initial_number = gaussian_density(
        grid,
        (3.4e-6, 2.18e-6),
        0.73e-6,
        density_convention=DensityConvention.NUMBER,
        population_count=population,
    )
    area = grid.cell_area_m2
    mass_tolerance = 64.0 * np.finfo(np.float64).eps * population
    for name, value, expected in (
        (
            "empirical_probability_mass",
            float(np.sum(empirical_probability) * area),
            1.0,
        ),
        (
            "empirical_number_mass",
            float(np.sum(empirical_number) * area),
            float(population),
        ),
        (
            "initial_probability_mass",
            float(np.sum(initial_probability) * area),
            1.0,
        ),
        (
            "initial_number_mass",
            float(np.sum(initial_number) * area),
            float(population),
        ),
    ):
        checks[name] = _equivalence_check(
            value,
            expected,
            relative_tolerance=0.0,
            absolute_tolerance=mass_tolerance,
            unit="1",
            formula="integral(rho dx)=1; integral(n dx)=N",
        )
    checks["empirical_number_equals_N_probability"] = _equivalence_check(
        empirical_number,
        population * empirical_probability,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
        unit="1/m^2",
        formula="n_empirical=N*rho_empirical",
    )
    checks["initial_number_equals_N_probability"] = _equivalence_check(
        initial_number,
        population * initial_probability,
        relative_tolerance=5.0e-16,
        absolute_tolerance=0.0,
        unit="1/m^2",
        formula="n_initial=N*rho_initial",
    )
    checks["kac_potential_scale"] = _equivalence_check(
        kac_potential.energy_scale_joule,
        population * pair_potential.energy_scale_joule,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
        unit="J",
        formula="W_Kac=N*W_pair",
    )

    probability_force = mean_field_pair_force_newton(
        positions, kac_potential
    )
    number_force = mean_field_pair_force_newton(
        positions,
        pair_potential,
        density_convention=DensityConvention.NUMBER,
    )
    checks["particle_force_equivalence"] = _equivalence_check(
        probability_force,
        number_force,
        relative_tolerance=2.0e-16,
        absolute_tolerance=1.0e-30,
        unit="N",
        formula="(1/N)*sum(F_Kac)=sum(F_pair)",
    )

    dt_particle_s = 7.5e-4
    particle_pair_chunk_size = 3
    probability_step, probability_diagnostics = overdamped_langevin_step(
        positions,
        parameters,
        kac_potential,
        domain,
        dt_s=dt_particle_s,
        rng=np.random.default_rng(seed),
        pair_chunk_size=particle_pair_chunk_size,
    )
    number_step, number_diagnostics = overdamped_langevin_step(
        positions,
        parameters,
        pair_potential,
        domain,
        dt_s=dt_particle_s,
        rng=np.random.default_rng(seed),
        pair_chunk_size=particle_pair_chunk_size,
        density_convention=DensityConvention.NUMBER,
    )
    checks["one_step_langevin_equivalence"] = _equivalence_check(
        probability_step,
        number_step,
        relative_tolerance=0.0,
        absolute_tolerance=1.0e-21,
        unit="m",
        formula="same drift and same seeded Brownian increment",
    )
    checks["one_step_langevin_no_collision"] = _boolean_check(
        probability_diagnostics.collision_count == 0
        and number_diagnostics.collision_count == 0,
        observed=[
            probability_diagnostics.collision_count,
            number_diagnostics.collision_count,
        ],
        required=[0, 0],
        formula="fixture remains away from the reflecting boundary",
    )

    state = ParticleMechanicalState(positions, np.zeros_like(positions))
    probability_particle_energy = discrete_particle_energy_joule(
        state,
        parameters,
        kac_potential,
    )
    number_particle_energy = discrete_particle_energy_joule(
        state,
        parameters,
        pair_potential,
        density_convention=DensityConvention.NUMBER,
    )
    checks["particle_total_energy_equivalence"] = _equivalence_check(
        probability_particle_energy,
        number_particle_energy,
        relative_tolerance=0.0,
        absolute_tolerance=1.0e-32,
        unit="J",
        formula="sum(W_Kac)/(2N)=sum(W_pair)/2",
    )

    direct_probability = direct_pair_convolution_joule(
        initial_probability,
        grid,
        kac_potential,
        population_count=population,
    )
    direct_number = direct_pair_convolution_joule(
        initial_number,
        grid,
        pair_potential,
        density_convention=DensityConvention.NUMBER,
        population_count=population,
    )
    fft_probability = FFTPairConvolver(
        grid,
        kac_potential,
        population_count=population,
    ).convolve_joule(initial_probability)
    fft_number = FFTPairConvolver(
        grid,
        pair_potential,
        density_convention=DensityConvention.NUMBER,
        population_count=population,
    ).convolve_joule(initial_number)
    for name, actual, expected, rtol, atol, formula in (
        (
            "direct_convolution_convention_equivalence",
            direct_number,
            direct_probability,
            3.0e-15,
            1.0e-35,
            "W_pair*n=W_Kac*rho (Direct)",
        ),
        (
            "fft_convolution_convention_equivalence",
            fft_number,
            fft_probability,
            3.0e-15,
            1.0e-35,
            "W_pair*n=W_Kac*rho (FFT)",
        ),
        (
            "probability_direct_fft_equivalence",
            fft_probability,
            direct_probability,
            2.0e-13,
            1.0e-34,
            "FFT(W_Kac*rho)=Direct(W_Kac*rho)",
        ),
        (
            "number_direct_fft_equivalence",
            fft_number,
            direct_number,
            2.0e-13,
            1.0e-34,
            "FFT(W_pair*n)=Direct(W_pair*n)",
        ),
    ):
        checks[name] = _equivalence_check(
            actual,
            expected,
            relative_tolerance=rtol,
            absolute_tolerance=atol,
            unit="J",
            formula=formula,
        )

    cfl_safety = 0.83
    probability_solver = McKeanVlasovSolver(
        grid,
        parameters,
        kac_potential,
        cfl_safety=cfl_safety,
        population_count=population,
    )
    number_solver = McKeanVlasovSolver(
        grid,
        parameters,
        pair_potential,
        cfl_safety=cfl_safety,
        density_convention=DensityConvention.NUMBER,
        population_count=population,
    )
    probability_flux, probability_interaction = probability_solver.face_fluxes(
        initial_probability
    )
    number_flux, number_interaction = number_solver.face_fluxes(initial_number)
    checks["solver_interaction_equivalence"] = _equivalence_check(
        number_interaction,
        probability_interaction,
        relative_tolerance=3.0e-15,
        absolute_tolerance=1.0e-35,
        unit="J",
        formula="W_pair*n=W_Kac*rho inside both solvers",
    )
    checks["x_face_number_flux_equals_N_probability_flux"] = _equivalence_check(
        number_flux.x_per_m_s,
        population * probability_flux.x_per_m_s,
        relative_tolerance=3.0e-15,
        absolute_tolerance=1.0e-22,
        unit="1/(m*s)",
        formula="J_n,x=N*J_rho,x",
    )
    checks["y_face_number_flux_equals_N_probability_flux"] = _equivalence_check(
        number_flux.y_per_m_s,
        population * probability_flux.y_per_m_s,
        relative_tolerance=3.0e-15,
        absolute_tolerance=1.0e-22,
        unit="1/(m*s)",
        formula="J_n,y=N*J_rho,y",
    )
    dt_fvm_s = min(
        8.0e-4,
        0.5 * probability_solver.stable_dt_s(probability_flux),
        0.5 * number_solver.stable_dt_s(number_flux),
    )
    probability_next, probability_step_diagnostics = probability_solver.step(
        initial_probability, dt_fvm_s
    )
    number_next, number_step_diagnostics = number_solver.step(
        initial_number, dt_fvm_s
    )
    checks["one_fvm_step_number_equals_N_probability"] = _equivalence_check(
        number_next,
        population * probability_next,
        relative_tolerance=3.0e-15,
        absolute_tolerance=1.0e-7,
        unit="1/m^2",
        formula="n^(k+1)=N*rho^(k+1)",
    )
    checks["one_fvm_substep_each"] = _boolean_check(
        probability_step_diagnostics.substeps == 1
        and number_step_diagnostics.substeps == 1,
        observed=[
            probability_step_diagnostics.substeps,
            number_step_diagnostics.substeps,
        ],
        required=[1, 1],
        formula="requested dt is below half of both initial stable dt values",
    )

    probability_energy = probability_solver.free_energy(initial_probability)
    number_energy = number_solver.free_energy(initial_number)
    for name, number_value, probability_value in (
        (
            "continuum_entropy_energy_scaling",
            number_energy.entropy_joule,
            probability_energy.entropy_joule,
        ),
        (
            "continuum_external_energy_scaling",
            number_energy.external_joule,
            probability_energy.external_joule,
        ),
        (
            "continuum_interaction_energy_scaling",
            number_energy.interaction_joule,
            probability_energy.interaction_joule,
        ),
        (
            "continuum_total_free_energy_scaling",
            number_energy.total_joule,
            probability_energy.total_joule,
        ),
    ):
        checks[name] = _equivalence_check(
            number_value,
            population * probability_value,
            relative_tolerance=3.0e-15,
            absolute_tolerance=1.0e-31,
            unit="J",
            formula="F_number[n;W_pair]=N*F_probability[rho;W_Kac]",
        )

    converted = convert_single_pair_table_to_kac(
        _test_only_single_pair_table(),
        population_count=population,
        population_count_provenance=(
            "TEST_ONLY fixture population equals the six particle positions"
        ),
    )
    population_bound_potential = effective_potential_from_table(converted)
    particle_only_potential = _hydrogel_test_potential(
        _test_only_single_pair_table(minimum_distance_m=0.5e-6),
        DensityConvention.NUMBER,
    )
    particle_only_admission = short_range_admission(
        particle_only_potential.minimum_supported_distance_m
    )
    rejection_gates = {
        "scaling_mismatch": _expected_rejection(
            lambda: mean_field_pair_force_newton(positions, pair_potential),
            expected_message="scaling mismatch",
            requirement=(
                "probability particles require a Kac-normalized potential"
            ),
        ),
        "population_mismatch": _expected_rejection(
            lambda: mean_field_pair_force_newton(
                positions[:-1], population_bound_potential
            ),
            expected_message="runtime population",
            requirement=(
                "declared Kac population must equal the runtime particle count"
            ),
        ),
        "probability_mass_mismatch": _expected_rejection(
            lambda: probability_solver.step(initial_number, dt_fvm_s),
            expected_message="density mass",
            requirement="probability density must integrate to one",
        ),
        "number_mass_mismatch": _expected_rejection(
            lambda: number_solver.step(initial_probability, dt_fvm_s),
            expected_message="density mass",
            requirement="number density must integrate to population N",
        ),
        "particle_only_short_range": _expected_rejection(
            lambda: FFTPairConvolver(
                grid,
                particle_only_potential,
                density_convention=DensityConvention.NUMBER,
                population_count=population,
            ),
            expected_message="zero displacement",
            requirement=(
                "r_min>0 is PARTICLE_ONLY_SHORT_RANGE_UNRESOLVED and cannot "
                "form a continuum kernel"
            ),
        ),
    }
    rejection_gates["particle_only_short_range"].update(
        particle_only_admission
    )

    formulas = {
        "density": "n(x)=N*rho(x)",
        "pair_potential": "W_Kac(r)=N*W_pair(r)",
        "particle_force": "(1/N)*sum_j F_Kac=sum_j F_pair",
        "particle_energy": "sum(W_Kac)/(2N)=sum(W_pair)/2",
        "convolution": "W_Kac*rho=W_pair*n",
        "flux": "J_n=N*J_rho",
        "finite_volume_step": "n^(k+1)=N*rho^(k+1)",
        "free_energy": "F_number=N*F_probability when n_ref=N*rho_ref",
    }
    parameters_payload = {
        "fixture_name": "TEST_ONLY_GAUSSIAN_REPULSION_NOT_HYDROGEL",
        "physical_status": TEST_ONLY_NOT_FINAL_PHYSICS,
        "population_count": population,
        "seed": seed,
        "rng_bit_generator": (
            np.random.default_rng(seed).bit_generator.__class__.__name__
        ),
        "physical_parameters": parameters.as_dict(),
        "domain_x_m": list(domain.x_limits_m),
        "domain_y_m": list(domain.y_limits_m),
        "grid_shape": [grid.ny, grid.nx],
        "initial_centre_m": [3.4e-6, 2.18e-6],
        "initial_standard_deviation_m": 0.73e-6,
        "single_pair_energy_scale_joule": pair_energy,
        "kac_energy_scale_joule": population * pair_energy,
        "pair_length_scale_m": pair_length,
        "particle_dt_s": dt_particle_s,
        "particle_pair_chunk_size": particle_pair_chunk_size,
        "continuum_cfl_safety": cfl_safety,
        "finite_volume_dt_s": dt_fvm_s,
        "particle_positions_m": positions.tolist(),
        "parameter_source": (
            "fixed TEST_ONLY density-contract regression fixture; not calibrated"
        ),
    }
    thresholds = {
        name: {
            key: value
            for key, value in check.items()
            if key
            in {
                "absolute_tolerance",
                "relative_tolerance",
                "maximum_allowed_absolute_error",
                "unit",
                "tolerance_source",
            }
        }
        for name, check in checks.items()
        if "absolute_tolerance" in check
    }
    overall_passed = all(check["passed"] for check in checks.values()) and all(
        gate["passed"] for gate in rejection_gates.values()
    )
    stable_evidence: dict[str, object] = {
        "formulas": formulas,
        "parameters": parameters_payload,
        "thresholds": thresholds,
        "checks": checks,
        "rejection_gates": rejection_gates,
        "overall_passed": overall_passed,
    }
    return {
        "schema_name": "mechanistic_mv.density_scaling_validation",
        "schema_version": PHASE6_SCHEMA_VERSION,
        "artifact_type": "DENSITY_SCALING_VALIDATION",
        "workflow_status": "TEST_ONLY_PASSED" if overall_passed else "FAILED",
        "physical_status": TEST_ONLY_NOT_FINAL_PHYSICS,
        "calibration_status": TEST_ONLY_NOT_CALIBRATED,
        "test_only_fixture": True,
        "validation_scope": (
            "TEST_ONLY_DENSITY_CONTRACT_REGRESSION_"
            "NOT_HYDROGEL_PHYSICAL_VALIDATION"
        ),
        **stable_evidence,
        "deterministic_evidence_sha256": _stable_signature(stable_evidence),
        "overall_conclusion": (
            "TEST_ONLY_PROBABILITY_AND_NUMBER_CONVENTIONS_EQUIVALENT"
            if overall_passed
            else "DENSITY_SCALING_REGRESSION_FAILED"
        ),
        "scientific_limitations": [
            "the Gaussian pair law is TEST_ONLY_NOT_FINAL_PHYSICS",
            "this validates implementation scaling, not Hydrogel contact mechanics",
            "tolerances are deterministic float64 regression budgets",
            "the one-step checks are not long-time convergence evidence",
        ],
    }


def _write_blocked(
    output: Path,
    *,
    reason: str,
    run_provenance: dict[str, object],
) -> Path:
    status = output.parent / "density_scaling_validation_status.json"
    write_json(
        status,
        {
            "schema_name": "mechanistic_mv.density_scaling_validation_status",
            "schema_version": PHASE6_SCHEMA_VERSION,
            "artifact_type": "DENSITY_SCALING_VALIDATION_STATUS",
            "workflow_status": "BLOCKED",
            "physical_status": "NOT_EVALUATED",
            "reason": reason,
            "generated_data_files": [],
            "numerical_results_generated": False,
            **run_provenance,
        },
    )
    return status


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    run_provenance = provenance(command)
    if not args.test_only_fixture:
        status = _write_blocked(
            args.output,
            reason=(
                "this deterministic validation requires explicit "
                "--test-only-fixture acknowledgement"
            ),
            run_provenance=run_provenance,
        )
        print(status)
        return 2
    try:
        report = run_validation(args.seed)
    except (TypeError, ValueError, RuntimeError, FloatingPointError) as error:
        status = _write_blocked(
            args.output,
            reason=str(error),
            run_provenance=run_provenance,
        )
        print(status)
        return 2
    report.update(run_provenance)
    write_json(args.output, report)
    print(args.output)
    return 0 if report["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
