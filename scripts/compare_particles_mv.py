"""Compare particles and MV using one validation-gated Hydrogel pair potential."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from mechanistic_mv.continuum.diagnostics import (
    density_moments,
    jensen_shannon_divergence,
    relative_l2_error,
)
from mechanistic_mv.continuum.initial_conditions import gaussian_density
from mechanistic_mv.continuum.mckean_vlasov import McKeanVlasovSolver
from mechanistic_mv.mechanics.geometry import (
    CartesianGrid,
    RectangularDomain,
    reflect_outer_walls,
)
from mechanistic_mv.mechanics.hydrogel import TEST_ONLY_NOT_CALIBRATED
from mechanistic_mv.mechanics.parameters import PhysicalParameters
from mechanistic_mv.particle_sim.empirical_density import empirical_density
from mechanistic_mv.particle_sim.langevin_interacting import (
    overdamped_langevin_step,
)

try:
    from ._phase6_common import (
        PAIR_VALIDATION_DIRECTORY,
        PHASE6_SCHEMA_VERSION,
        effective_potential_from_table,
        load_effective_potential_artifact,
        provenance,
        sha256_file,
        test_only_pair_force_table,
        write_json,
    )
except ImportError:  # direct ``python scripts/...`` execution
    from _phase6_common import (  # type: ignore[no-redef]
        PAIR_VALIDATION_DIRECTORY,
        PHASE6_SCHEMA_VERSION,
        effective_potential_from_table,
        load_effective_potential_artifact,
        provenance,
        sha256_file,
        test_only_pair_force_table,
        write_json,
    )


MASS_ERROR_LIMIT = 1.0e-12
POSITIVE_ENERGY_INCREMENT_LIMIT_JOULE = 1.0e-30
RELATIVE_L2_LIMIT = 0.25
JS_DIVERGENCE_LIMIT_NATS = 0.05
CENTROID_TRAJECTORY_RMSE_LIMIT_M = 2.0e-7
COVARIANCE_RELATIVE_ERROR_LIMIT = 0.35

_FIXTURE_CONFIGURATION: dict[str, object] = {
    "domain_size_m": 12.0e-6,
    "initial_centre_x_m": 6.0e-6,
    "initial_centre_y_m": 6.0e-6,
    "initial_standard_deviation_m": 1.0e-6,
    "particle_count": 400,
    "grid_size": 14,
    "steps": 12,
    "dt_s": 1.0e-3,
    "seed": 20260824,
    "pair_chunk_size": 128,
    "particle_mass_kg": 1.0e-15,
    "drag_coefficient_kg_per_s": 1.0e-8,
    "temperature_kelvin": 298.15,
    "parameter_source": (
        "Phase 6 TEST_ONLY numerical-regression fixture; not calibrated"
    ),
}

_CONFIGURATION_ARGUMENTS = {
    "domain_size_m": "--domain-size-m",
    "initial_centre_x_m": "--initial-centre-x-m",
    "initial_centre_y_m": "--initial-centre-y-m",
    "initial_standard_deviation_m": "--initial-standard-deviation-m",
    "particle_count": "--particle-count",
    "grid_size": "--grid-size",
    "steps": "--steps",
    "dt_s": "--dt-s",
    "seed": "--seed",
    "pair_chunk_size": "--pair-chunk-size",
    "particle_mass_kg": "--particle-mass-kg",
    "drag_coefficient_kg_per_s": "--drag-coefficient-kg-per-s",
    "temperature_kelvin": "--temperature-kelvin",
    "parameter_source": "--parameter-source",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effective-potential-csv", type=Path)
    parser.add_argument("--effective-potential-metadata", type=Path)
    parser.add_argument("--test-only-fixture", action="store_true")
    parser.add_argument(
        "--output-directory", type=Path, default=PAIR_VALIDATION_DIRECTORY
    )
    parser.add_argument("--domain-size-m", type=float)
    parser.add_argument("--initial-centre-x-m", type=float)
    parser.add_argument("--initial-centre-y-m", type=float)
    parser.add_argument("--initial-standard-deviation-m", type=float)
    parser.add_argument("--particle-count", type=int)
    parser.add_argument("--grid-size", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--dt-s", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--pair-chunk-size", type=int)
    parser.add_argument("--particle-mass-kg", type=float)
    parser.add_argument("--drag-coefficient-kg-per-s", type=float)
    parser.add_argument("--temperature-kelvin", type=float)
    parser.add_argument("--parameter-source")
    return parser


def _blocked(
    output_directory: Path,
    *,
    reason: str,
    missing_inputs: list[str],
    run_provenance: dict[str, object],
) -> Path:
    status_path = output_directory / "particles_mv_comparison_status.json"
    preexisting = [
        name
        for name in (
            "particles_mv_comparison.json",
            "TEST_ONLY_particles_mv_comparison.json",
        )
        if (output_directory / name).exists()
    ]
    write_json(
        status_path,
        {
            "schema_name": "mechanistic_mv.particles_mv_comparison_status",
            "schema_version": PHASE6_SCHEMA_VERSION,
            "artifact_type": "PARTICLES_MV_COMPARISON_STATUS",
            "workflow_status": "BLOCKED",
            "physical_status": "NOT_EVALUATED",
            "reason": reason,
            "missing_inputs": missing_inputs,
            "generated_data_files": [],
            "preexisting_untrusted_artifacts": preexisting,
            "numerical_results_generated": False,
            **run_provenance,
        },
    )
    return status_path


def _configuration(
    args: argparse.Namespace,
) -> tuple[dict[str, object] | None, list[str]]:
    supplied = {
        name: getattr(args, name) for name in _CONFIGURATION_ARGUMENTS
    }
    if args.test_only_fixture:
        values = dict(_FIXTURE_CONFIGURATION)
        values.update(
            {name: value for name, value in supplied.items() if value is not None}
        )
        return values, []
    missing = [
        flag
        for name, flag in _CONFIGURATION_ARGUMENTS.items()
        if supplied[name] is None
    ]
    return (None, missing) if missing else (supplied, [])


def _validated_configuration(values: dict[str, object]) -> dict[str, object]:
    positive_float_names = (
        "domain_size_m",
        "initial_standard_deviation_m",
        "dt_s",
        "particle_mass_kg",
        "drag_coefficient_kg_per_s",
        "temperature_kelvin",
    )
    for name in positive_float_names:
        value = float(values[name])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{_CONFIGURATION_ARGUMENTS[name]} must be positive")
        values[name] = value
    for name, minimum in (
        ("particle_count", 2),
        ("grid_size", 4),
        ("steps", 1),
        ("pair_chunk_size", 1),
    ):
        value = int(values[name])
        if value < minimum:
            raise ValueError(
                f"{_CONFIGURATION_ARGUMENTS[name]} must be at least {minimum}"
            )
        values[name] = value
    seed = int(values["seed"])
    if seed < 0:
        raise ValueError("--seed must be non-negative")
    values["seed"] = seed
    for name in ("initial_centre_x_m", "initial_centre_y_m"):
        value = float(values[name])
        if not np.isfinite(value) or not 0.0 < value < values["domain_size_m"]:
            raise ValueError(
                f"{_CONFIGURATION_ARGUMENTS[name]} must lie inside the domain"
            )
        values[name] = value
    source = values["parameter_source"]
    if not isinstance(source, str) or not source.strip():
        raise ValueError("--parameter-source must be a non-empty provenance string")
    if source.strip().casefold() in {"unknown", "tbd", "unspecified"}:
        raise ValueError("--parameter-source must be specific")
    values["parameter_source"] = source.strip()
    return values


def _fixture_source_digest() -> str:
    table = test_only_pair_force_table()
    values = np.column_stack(
        (table.center_distance_m, table.radial_force_newton)
    ).astype("<f8", copy=False)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _run_comparison(
    potential,
    configuration: dict[str, object],
) -> dict[str, object]:
    total_start = perf_counter()
    domain_size = float(configuration["domain_size_m"])
    domain = RectangularDomain((0.0, domain_size), (0.0, domain_size))
    grid_size = int(configuration["grid_size"])
    grid = CartesianGrid(domain, grid_size, grid_size)
    parameters = PhysicalParameters(
        particle_mass_kg=float(configuration["particle_mass_kg"]),
        drag_coefficient_kg_per_s=float(
            configuration["drag_coefficient_kg_per_s"]
        ),
        temperature_kelvin=float(configuration["temperature_kelvin"]),
    )
    centre = (
        float(configuration["initial_centre_x_m"]),
        float(configuration["initial_centre_y_m"]),
    )
    standard_deviation = float(configuration["initial_standard_deviation_m"])
    density = gaussian_density(grid, centre, standard_deviation)
    solver = McKeanVlasovSolver(grid, parameters, potential)
    rng = np.random.default_rng(int(configuration["seed"]))
    particles = rng.normal(
        np.asarray(centre),
        standard_deviation,
        size=(int(configuration["particle_count"]), 2),
    )
    particles = reflect_outer_walls(particles, domain)

    mass = [solver.mass(density)]
    energy_components = [solver.free_energy(density)]
    mv_centroids = [density_moments(density, grid).mean_m]
    particle_centroids = [np.mean(particles, axis=0)]
    clipped_negative_mass = 0.0
    collision_count = 0
    total_substeps = 0

    continuum_seconds = 0.0
    particle_seconds = 0.0
    steps = int(configuration["steps"])
    dt_s = float(configuration["dt_s"])
    for _ in range(steps):
        start = perf_counter()
        density, diagnostics = solver.step(density, dt_s)
        continuum_seconds += perf_counter() - start
        start = perf_counter()
        particles, particle_diagnostics = overdamped_langevin_step(
            particles,
            parameters,
            potential,
            domain,
            dt_s=dt_s,
            rng=rng,
            pair_chunk_size=int(configuration["pair_chunk_size"]),
        )
        particle_seconds += perf_counter() - start
        mass.append(diagnostics.final_mass)
        energy_components.append(solver.free_energy(density))
        mv_centroids.append(density_moments(density, grid).mean_m)
        particle_centroids.append(np.mean(particles, axis=0))
        clipped_negative_mass += diagnostics.clipped_negative_mass
        collision_count += particle_diagnostics.collision_count
        total_substeps += diagnostics.substeps

    metrics_start = perf_counter()
    particle_density = empirical_density(particles, grid)
    mv_moments = density_moments(density, grid)
    particle_moments = density_moments(particle_density, grid)
    centroid_difference = np.asarray(particle_centroids) - np.asarray(mv_centroids)
    covariance_denominator = np.maximum(
        np.abs(np.diag(mv_moments.covariance_m2)),
        np.finfo(np.float64).tiny,
    )
    covariance_relative_error = np.abs(
        np.diag(particle_moments.covariance_m2)
        - np.diag(mv_moments.covariance_m2)
    ) / covariance_denominator
    energy = np.asarray(
        [component.total_joule for component in energy_components],
        dtype=np.float64,
    )
    energy_increment = np.diff(energy)
    raw_maximum_energy_increment = float(np.max(energy_increment))
    maximum_positive_energy_increment = max(0.0, raw_maximum_energy_increment)
    relative_l2 = relative_l2_error(particle_density, density, grid)
    js_divergence = jensen_shannon_divergence(particle_density, density, grid)
    centroid_rmse = float(
        np.sqrt(np.mean(np.sum(centroid_difference**2, axis=1)))
    )
    maximum_covariance_error = float(np.max(covariance_relative_error))
    maximum_mass_error = float(np.max(np.abs(np.asarray(mass) - mass[0])))
    metrics_seconds = perf_counter() - metrics_start

    checks = {
        "same_effective_pair_potential_instance": {
            "passed": solver.pair_potential is potential,
            "value": solver.pair_potential is potential,
            "required": True,
        },
        "kac_scaling_for_unit_mass_density": {
            "passed": potential.scaling_semantics
            == "KAC_EFFECTIVE_FOR_UNIT_MASS_RHO_AND_ONE_OVER_N_PARTICLE_FORCE",
            "value": potential.scaling_semantics,
            "required": (
                "KAC_EFFECTIVE_FOR_UNIT_MASS_RHO_AND_ONE_OVER_N_PARTICLE_FORCE"
            ),
        },
        "continuum_mass_error": {
            "passed": maximum_mass_error <= MASS_ERROR_LIMIT,
            "value": maximum_mass_error,
            "upper_limit": MASS_ERROR_LIMIT,
        },
        "no_material_negative_mass_clipping": {
            "passed": clipped_negative_mass == 0.0,
            "value": clipped_negative_mass,
            "upper_limit": 0.0,
        },
        "passive_free_energy_nonincrease": {
            "passed": maximum_positive_energy_increment
            <= POSITIVE_ENERGY_INCREMENT_LIMIT_JOULE,
            "value_joule": maximum_positive_energy_increment,
            "raw_maximum_increment_joule": raw_maximum_energy_increment,
            "upper_limit_joule": POSITIVE_ENERGY_INCREMENT_LIMIT_JOULE,
            "sampling_scope": (
                "requested-step endpoints only; not every adaptive substep or "
                "a continuous-time dissipation proof"
            ),
        },
        "particle_MV_JS_divergence": {
            "passed": js_divergence <= JS_DIVERGENCE_LIMIT_NATS,
            "value_nats": js_divergence,
            "upper_limit_nats": JS_DIVERGENCE_LIMIT_NATS,
        },
        "particle_MV_relative_L2": {
            "passed": relative_l2 <= RELATIVE_L2_LIMIT,
            "value": relative_l2,
            "upper_limit": RELATIVE_L2_LIMIT,
        },
        "particle_MV_centroid_trajectory": {
            "passed": centroid_rmse <= CENTROID_TRAJECTORY_RMSE_LIMIT_M,
            "value_m": centroid_rmse,
            "upper_limit_m": CENTROID_TRAJECTORY_RMSE_LIMIT_M,
        },
        "particle_MV_covariance": {
            "passed": maximum_covariance_error
            <= COVARIANCE_RELATIVE_ERROR_LIMIT,
            "value": maximum_covariance_error,
            "upper_limit": COVARIANCE_RELATIVE_ERROR_LIMIT,
        },
    }
    overall_passed = all(bool(value["passed"]) for value in checks.values())
    final_particle_energy = solver.free_energy(particle_density)
    return {
        "overall_passed": overall_passed,
        "checks": checks,
        "metrics": {
            "relative_L2_density_error": relative_l2,
            "JS_divergence_nats": js_divergence,
            "centroid_trajectory_RMSE_m": centroid_rmse,
            "mean_error_m": (
                particle_moments.mean_m - mv_moments.mean_m
            ).tolist(),
            "covariance_relative_error_diagonal": (
                covariance_relative_error.tolist()
            ),
            "maximum_covariance_relative_error": maximum_covariance_error,
            "maximum_absolute_continuum_mass_error": maximum_mass_error,
            "total_clipped_negative_mass": clipped_negative_mass,
            "continuum_initial_free_energy_joule": float(energy[0]),
            "continuum_final_free_energy_joule": float(energy[-1]),
            "continuum_energy_change_joule": float(energy[-1] - energy[0]),
            "continuum_raw_maximum_energy_increment_joule": (
                raw_maximum_energy_increment
            ),
            "continuum_maximum_positive_energy_increment_joule": (
                maximum_positive_energy_increment
            ),
            "continuum_final_interaction_energy_joule": (
                energy_components[-1].interaction_joule
            ),
            "continuum_functional_on_particle_histogram_final_free_energy_joule": (
                final_particle_energy.total_joule
            ),
            "continuum_functional_on_particle_histogram_interaction_energy_joule": (
                final_particle_energy.interaction_joule
            ),
            "particle_collision_count": collision_count,
            "continuum_total_substeps": total_substeps,
        },
        "trajectories": {
            "continuum_mass": mass,
            "continuum_free_energy_joule": energy.tolist(),
            "continuum_centroid_m": np.asarray(mv_centroids).tolist(),
            "particle_centroid_m": np.asarray(particle_centroids).tolist(),
        },
        "timing": {
            "continuum_steps_seconds": continuum_seconds,
            "particle_steps_seconds": particle_seconds,
            "final_metrics_seconds": metrics_seconds,
            "total_seconds": perf_counter() - total_start,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_directory: Path = args.output_directory
    command = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    run_provenance = provenance(command)
    has_artifact_csv = args.effective_potential_csv is not None
    has_artifact_metadata = args.effective_potential_metadata is not None
    supplied_artifact = has_artifact_csv and has_artifact_metadata
    if has_artifact_csv != has_artifact_metadata:
        missing = (
            ["--effective-potential-metadata"]
            if has_artifact_csv
            else ["--effective-potential-csv"]
        )
        status = _blocked(
            output_directory,
            reason="effective-potential CSV and metadata must be supplied together",
            missing_inputs=missing,
            run_provenance=run_provenance,
        )
        print(status)
        return 2
    if not args.test_only_fixture:
        missing = []
        if args.effective_potential_csv is None:
            missing.append("--effective-potential-csv")
        if args.effective_potential_metadata is None:
            missing.append("--effective-potential-metadata")
        if missing:
            status = _blocked(
                output_directory,
                reason="a passed physical effective-potential artifact is required",
                missing_inputs=missing,
                run_provenance=run_provenance,
            )
            print(status)
            return 2

    configuration, missing_configuration = _configuration(args)
    if missing_configuration:
        status = _blocked(
            output_directory,
            reason="physical particle/MV conditions have no implicit defaults",
            missing_inputs=missing_configuration,
            run_provenance=run_provenance,
        )
        print(status)
        return 2
    try:
        assert configuration is not None
        configuration = _validated_configuration(configuration)
        if args.test_only_fixture and supplied_artifact:
            potential_metadata_sha256 = sha256_file(
                args.effective_potential_metadata
            )
            potential, potential_metadata = load_effective_potential_artifact(
                args.effective_potential_csv,
                args.effective_potential_metadata,
            )
            if potential.physical_status != TEST_ONLY_NOT_CALIBRATED:
                raise ValueError(
                    "--test-only-fixture accepts only explicitly TEST_ONLY artifacts"
                )
            if (
                sha256_file(args.effective_potential_metadata)
                != potential_metadata_sha256
            ):
                raise ValueError(
                    "effective-potential metadata changed during validation"
                )
            potential_source_sha256 = potential_metadata["data_file_sha256"]
            potential_input_mode = "VALIDATED_TEST_ONLY_EFFECTIVE_ARTIFACT"
            physical_status = TEST_ONLY_NOT_CALIBRATED
            calibration_status = TEST_ONLY_NOT_CALIBRATED
            calibration_id = potential_metadata["calibration_id"]
            workflow_status = "TEST_ONLY_PASSED"
            prefix = "TEST_ONLY_"
        elif args.test_only_fixture:
            potential = effective_potential_from_table(
                test_only_pair_force_table()
            )
            potential_source_sha256 = _fixture_source_digest()
            physical_status = TEST_ONLY_NOT_CALIBRATED
            calibration_status = TEST_ONLY_NOT_CALIBRATED
            calibration_id = TEST_ONLY_NOT_CALIBRATED
            workflow_status = "TEST_ONLY_PASSED"
            prefix = "TEST_ONLY_"
            potential_metadata_sha256 = None
            potential_input_mode = "INTERNAL_ANALYTIC_TEST_ONLY_FIXTURE"
        else:
            potential_metadata_sha256 = sha256_file(
                args.effective_potential_metadata
            )
            potential, potential_metadata = load_effective_potential_artifact(
                args.effective_potential_csv,
                args.effective_potential_metadata,
            )
            if potential.physical_status == TEST_ONLY_NOT_CALIBRATED:
                raise ValueError(
                    "TEST_ONLY artifacts require the explicit --test-only-fixture path"
                )
            if (
                sha256_file(args.effective_potential_metadata)
                != potential_metadata_sha256
            ):
                raise ValueError(
                    "effective-potential metadata changed during validation"
                )
            potential_source_sha256 = potential_metadata["data_file_sha256"]
            physical_status = potential.physical_status
            calibration_status = potential_metadata["calibration_status"]
            calibration_id = potential_metadata["calibration_id"]
            workflow_status = (
                "NUMERICAL_REGRESSION_PASSED_WITH_PHYSICAL_INPUT"
            )
            prefix = ""
            potential_input_mode = "VALIDATED_PHYSICAL_EFFECTIVE_ARTIFACT"
        results = _run_comparison(potential, configuration)
    except (
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
        FloatingPointError,
    ) as error:
        status = _blocked(
            output_directory,
            reason=str(error),
            missing_inputs=[],
            run_provenance=run_provenance,
        )
        print(status)
        return 2

    if not results["overall_passed"]:
        workflow_status = "FAILED"
    report_path = output_directory / f"{prefix}particles_mv_comparison.json"
    status_path = output_directory / "particles_mv_comparison_status.json"
    report = {
        "schema_name": "mechanistic_mv.particles_mv_comparison",
        "schema_version": PHASE6_SCHEMA_VERSION,
        "artifact_type": "PARTICLES_MV_COMPARISON",
        "workflow_status": workflow_status,
        "physical_status": physical_status,
        "calibration_status": calibration_status,
        "calibration_id": calibration_id,
        "test_only_fixture": bool(args.test_only_fixture),
        "validation_scope": (
            "SHORT_NUMERICAL_REGRESSION_NOT_HYDROGEL_PHYSICAL_VALIDATION"
            if args.test_only_fixture
            else (
                "PHYSICAL_INPUT_SHORT_SINGLE_SEED_NUMERICAL_COMPARISON_"
                "NOT_EXPERIMENTAL_VALIDATION"
            )
        ),
        "potential": {
            "name": potential.name,
            "dataset_id": potential.force_data.metadata.dataset_id,
            "source_data_sha256": potential_source_sha256,
            "source_metadata_sha256": potential_metadata_sha256,
            "input_mode": potential_input_mode,
            "scaling": potential.scaling_semantics,
            "same_python_object_used_by_particles_and_continuum": True,
            "force_potential_validation_passed": (
                potential.validation_report.passed
            ),
            "shared_object_interpretation": (
                "implementation reuse only; not evidence of physical calibration"
            ),
        },
        "configuration": configuration,
        "physical_parameters": PhysicalParameters(
            particle_mass_kg=float(configuration["particle_mass_kg"]),
            drag_coefficient_kg_per_s=float(
                configuration["drag_coefficient_kg_per_s"]
            ),
            temperature_kelvin=float(configuration["temperature_kelvin"]),
        ).as_dict(),
        "gate_provenance": {
            "mass_energy_and_single_seed_particle_MV": (
                "existing schema-1 validation gates"
            ),
            "covariance_relative_error": (
                "single-seed screening against the existing schema-3 "
                "multi-seed upper limit; this run is not a schema-3 study"
            ),
            "these_are_numerical_regression_limits": True,
            "these_are_not_material_calibration_uncertainties": True,
        },
        "scientific_limitations": [
            "one fixed seed is a reproducible regression, not a multi-seed study",
            "finite particle count and histogram discretization contribute error",
            "particle and continuum initial states differ by finite sampling error",
            (
                "thresholds were inherited from earlier TEST_ONLY numerical "
                "validation and are not Hydrogel material/contact acceptance limits"
            ),
            (
                "the empirical-histogram energy is the continuum functional "
                "evaluated on a histogram, not discrete particle-system energy"
            ),
        ],
        **results,
        **run_provenance,
    }
    write_json(report_path, report)
    write_json(
        status_path,
        {
            "schema_name": "mechanistic_mv.particles_mv_comparison_status",
            "schema_version": PHASE6_SCHEMA_VERSION,
            "artifact_type": "PARTICLES_MV_COMPARISON_STATUS",
            "workflow_status": workflow_status,
            "physical_status": physical_status,
            "test_only_fixture": bool(args.test_only_fixture),
            "overall_passed": results["overall_passed"],
            "report_file": report_path.name,
            "generated_data_files": [report_path.name],
            **run_provenance,
        },
    )
    print(report_path)
    return 0 if results["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
