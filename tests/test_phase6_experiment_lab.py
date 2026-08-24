from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mechanistic_mv.mechanics.hydrogel import TEST_ONLY_NOT_CALIBRATED
from mechanistic_mv.mechanics.pair_interaction import PairForceScaling

from scripts import (
    build_effective_potential,
    compare_particles_mv,
    run_hydrogel_validation,
    run_pair_contact_sweep,
)
from scripts._phase6_common import (
    load_effective_potential_artifact,
    load_pair_force_table,
    pair_force_metadata_payload,
    read_json_object,
    sha256_file,
    test_only_pair_force_table,
    test_only_scaling_provenance,
    test_only_short_range_provenance,
    write_pair_force_csv,
)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"expected a JSON object in {path}")
    return value


class HydrogelValidationScriptTests(unittest.TestCase):
    def test_missing_real_parameters_write_blocked_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "hydrogel.json"
            with patch("builtins.print"):
                return_code = run_hydrogel_validation.main(
                    ["--output", str(output)]
                )

            self.assertEqual(return_code, 2)
            report = _read_json(output)
            self.assertEqual(report["workflow_status"], "BLOCKED")
            self.assertEqual(report["physical_status"], "NOT_EVALUATED")
            self.assertEqual(report["generated_data_files"], [])
            self.assertEqual(
                set(report["missing_inputs"]),
                {
                    "network_density_times_solvent_volume",
                    "flory_huggins_chi",
                    "initial_polymer_volume_fraction",
                    "delta_chemical_potential_over_kbt",
                    "thermal_energy_density_pa",
                    "calibration_status",
                    "parameter_source",
                },
            )

    def test_fixture_is_marked_test_only_and_identity_gates_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "TEST_ONLY_hydrogel.json"
            with patch("builtins.print"):
                return_code = run_hydrogel_validation.main(
                    ["--test-only-fixture", "--output", str(output)]
                )

            self.assertEqual(return_code, 0)
            report = _read_json(output)
            self.assertTrue(report["test_only_fixture"])
            self.assertTrue(report["overall_passed"])
            self.assertEqual(report["workflow_status"], "TEST_ONLY_PASSED")
            self.assertEqual(
                report["physical_status"], TEST_ONLY_NOT_CALIBRATED
            )
            self.assertEqual(
                report["calibration_status"], TEST_ONLY_NOT_CALIBRATED
            )
            self.assertIn(
                "NOT_MATERIAL_CALIBRATION", report["validation_scope"]
            )

            checks = report["checks"]
            stress = checks["first_piola_is_gibbs_gradient"]
            conjugacy = checks["gibbs_chemical_conjugacy"]
            self.assertTrue(stress["passed"])
            self.assertEqual(stress["absolute_tolerance"], 2.0e-5)
            self.assertEqual(stress["relative_tolerance"], 2.0e-8)
            self.assertEqual(
                stress["finite_difference_step_in_dimensionless_F"], 1.0e-6
            )
            self.assertTrue(conjugacy["passed"])
            self.assertEqual(conjugacy["absolute_tolerance"], 1.0e-11)
            self.assertEqual(conjugacy["relative_tolerance"], 2.0e-10)
            self.assertEqual(conjugacy["finite_difference_step"], 1.0e-6)
            self.assertEqual(
                report["unsupported_claims"]["mu_equals_partial_Psi_partial_C"],
                "NOT_EVALUATED",
            )

    def test_test_only_status_and_invalid_real_values_fail_structurally(self) -> None:
        base = [
            "--network-density-times-solvent-volume",
            "0.025",
            "--flory-huggins-chi",
            "0.31",
            "--initial-polymer-volume-fraction",
            "0.18",
            "--delta-chemical-potential-over-kbt",
            "0.12",
            "--thermal-energy-density-pa",
            "240000",
            "--calibration-status",
            TEST_ONLY_NOT_CALIBRATED,
            "--parameter-source",
            "explicit test invocation",
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            test_status_output = root / "test_status.json"
            with patch("builtins.print"):
                return_code = run_hydrogel_validation.main(
                    [*base, "--output", str(test_status_output)]
                )
            self.assertEqual(return_code, 2)
            report = _read_json(test_status_output)
            self.assertEqual(report["workflow_status"], "BLOCKED")
            self.assertIn("--test-only-fixture", report["reason"])

            embedded_marker_output = root / "embedded_marker.json"
            embedded_marker = list(base)
            embedded_marker[
                embedded_marker.index(TEST_ONLY_NOT_CALIBRATED)
            ] = "CALIBRATED_TEST_ONLY"
            embedded_marker[embedded_marker.index("240000")] = "240001"
            with patch("builtins.print"):
                return_code = run_hydrogel_validation.main(
                    [
                        *embedded_marker,
                        "--output",
                        str(embedded_marker_output),
                    ]
                )
            self.assertEqual(return_code, 2)
            report = _read_json(embedded_marker_output)
            self.assertEqual(report["workflow_status"], "BLOCKED")

            invalid_output = root / "invalid.json"
            invalid = list(base)
            invalid[invalid.index("0.18")] = "1.5"
            invalid[invalid.index(TEST_ONLY_NOT_CALIBRATED)] = "CALIBRATED_LAB_SET"
            with patch("builtins.print"):
                return_code = run_hydrogel_validation.main(
                    [*invalid, "--output", str(invalid_output)]
                )
            self.assertEqual(return_code, 2)
            report = _read_json(invalid_output)
            self.assertEqual(report["workflow_status"], "BLOCKED")
            self.assertIn("invalid explicit Hydrogel parameters", report["reason"])


class PairContactSweepScriptTests(unittest.TestCase):
    def test_missing_fem_inputs_fail_closed_without_force_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            with patch("builtins.print"):
                return_code = run_pair_contact_sweep.main(
                    ["--output-directory", str(output_directory)]
                )

            self.assertEqual(return_code, 2)
            status = _read_json(
                output_directory / "pair_contact_sweep_status.json"
            )
            metadata = _read_json(output_directory / "metadata.json")
            for report in (status, metadata):
                self.assertEqual(report["workflow_status"], "BLOCKED")
                self.assertEqual(report["validation_status"], "UNVERIFIED")
                self.assertEqual(
                    report["contact_solver_status"],
                    "CONTACT_FEM_BACKEND_UNAVAILABLE",
                )
                self.assertEqual(
                    report["contact_results_status"],
                    "CONTACT_RESULTS_NOT_GENERATED",
                )
                self.assertEqual(
                    report["data_semantics"],
                    PairForceScaling.UNSCALED_SINGLE_PAIR.value,
                )
                self.assertEqual(report["generated_data_files"], [])
                required_groups = {
                    item["requirement_id"]: item["status"]
                    for item in report["required_input_groups"]
                }
                self.assertEqual(
                    required_groups["solver.identity_and_configuration"],
                    "MISSING",
                )
                self.assertEqual(
                    required_groups["geometry.two_sphere"], "MISSING"
                )
                self.assertEqual(
                    required_groups["boundary.solvent_bath"], "MISSING"
                )
                self.assertEqual(
                    required_groups["timescale.separation_and_source"],
                    "MISSING",
                )
            self.assertEqual(list(output_directory.rglob("*.csv")), [])
            self.assertEqual(list(output_directory.rglob("*.png")), [])


class EffectivePotentialBuilderScriptTests(unittest.TestCase):
    def test_missing_input_artifacts_write_blocked_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            with patch("builtins.print"):
                return_code = build_effective_potential.main(
                    ["--output-directory", str(output_directory)]
                )

            self.assertEqual(return_code, 2)
            status = _read_json(
                output_directory / "effective_potential_status.json"
            )
            self.assertEqual(status["workflow_status"], "BLOCKED")
            self.assertEqual(status["generated_data_files"], [])
            self.assertFalse(status["kac_scaling_implemented_by_this_script"])
            self.assertEqual(
                set(status["missing_inputs"]),
                {"--input-force-csv", "--input-metadata"},
            )
            self.assertEqual(list(output_directory.rglob("*.csv")), [])

    def test_unscaled_single_pair_table_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_directory = root / "input"
            output_directory = root / "output"
            force_csv = input_directory / "pair_force.csv"
            metadata_path = input_directory / "pair_force_metadata.json"
            table = test_only_pair_force_table()
            write_pair_force_csv(force_csv, table)
            scaling_provenance = test_only_scaling_provenance()
            scaling_provenance.update(
                {
                    "source_semantics": (
                        PairForceScaling.UNSCALED_SINGLE_PAIR.value
                    ),
                    "target_semantics": (
                        PairForceScaling.UNSCALED_SINGLE_PAIR.value
                    ),
                }
            )
            metadata = pair_force_metadata_payload(
                table,
                data_file_sha256=sha256_file(force_csv),
                scaling_provenance=scaling_provenance,
                short_range_closure_provenance=(
                    test_only_short_range_provenance()
                ),
                workflow_status="TEST_ONLY_PASSED",
                calibration_status=TEST_ONLY_NOT_CALIBRATED,
            )
            metadata["scaling"] = PairForceScaling.UNSCALED_SINGLE_PAIR.value
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            with patch("builtins.print"):
                return_code = build_effective_potential.main(
                    [
                        "--input-force-csv",
                        str(force_csv),
                        "--input-metadata",
                        str(metadata_path),
                        "--output-directory",
                        str(output_directory),
                    ]
                )

            self.assertEqual(return_code, 2)
            status = _read_json(
                output_directory / "effective_potential_status.json"
            )
            self.assertEqual(status["workflow_status"], "BLOCKED")
            self.assertIn("UNSCALED_SINGLE_PAIR", status["reason"])
            self.assertFalse(status["kac_scaling_implemented_by_this_script"])
            self.assertEqual(list(output_directory.rglob("*.csv")), [])

    def test_fixture_passes_force_potential_and_vector_scalar_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            with patch("builtins.print"):
                return_code = build_effective_potential.main(
                    [
                        "--test-only-fixture",
                        "--output-directory",
                        str(output_directory),
                    ]
                )

            self.assertEqual(return_code, 0)
            metadata_path = (
                output_directory
                / "TEST_ONLY_effective_potential_metadata.json"
            )
            data_path = (
                output_directory / "TEST_ONLY_effective_potential.csv"
            )
            self.assertTrue(metadata_path.is_file())
            self.assertTrue(data_path.is_file())
            report = _read_json(metadata_path)
            status = _read_json(
                output_directory / "effective_potential_status.json"
            )
            self.assertTrue(report["test_only_fixture"])
            self.assertTrue(report["overall_passed"])
            self.assertEqual(report["workflow_status"], "TEST_ONLY_PASSED")
            self.assertEqual(
                report["physical_status"], TEST_ONLY_NOT_CALIBRATED
            )
            self.assertEqual(
                report["calibration_status"], TEST_ONLY_NOT_CALIBRATED
            )
            self.assertEqual(
                report["scaling"],
                PairForceScaling.KAC_NORMALIZED_PROBABILITY.value,
            )
            self.assertTrue(report["force_potential_consistency"]["passed"])
            self.assertEqual(
                report["force_potential_consistency"][
                    "absolute_tolerance_newton"
                ],
                1.0e-20,
            )
            self.assertEqual(
                report["force_potential_consistency"]["relative_tolerance"],
                2.0e-7,
            )
            comparison = report["implementation_comparison"]
            self.assertTrue(comparison["vectorized_vs_scalar"]["passed"])
            self.assertEqual(
                comparison["vectorized_vs_scalar"]["relative_tolerance"],
                1.0e-14,
            )
            self.assertEqual(
                comparison["pchip_vs_trapezoid"][
                    "physical_equivalence_verdict"
                ],
                "NOT_ESTABLISHED_NO_INDEPENDENT_PHYSICAL_ERROR_BUDGET",
            )
            self.assertEqual(status["workflow_status"], "TEST_ONLY_PASSED")
            self.assertTrue(status["overall_passed"])
            loaded_potential, loaded_metadata = (
                load_effective_potential_artifact(data_path, metadata_path)
            )
            self.assertEqual(
                loaded_potential.physical_status,
                TEST_ONLY_NOT_CALIBRATED,
            )
            self.assertEqual(
                loaded_metadata["data_file_sha256"], sha256_file(data_path)
            )

    def test_external_test_only_and_incomplete_physical_tables_are_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            force_csv = root / "pair_force.csv"
            table = test_only_pair_force_table()
            write_pair_force_csv(force_csv, table)
            force_hash = sha256_file(force_csv)
            metadata = pair_force_metadata_payload(
                table,
                data_file_sha256=force_hash,
                scaling_provenance=test_only_scaling_provenance(),
                short_range_closure_provenance=(
                    test_only_short_range_provenance()
                ),
                workflow_status="TEST_ONLY_PASSED",
                calibration_status=TEST_ONLY_NOT_CALIBRATED,
            )
            test_metadata = root / "test_metadata.json"
            test_metadata.write_text(json.dumps(metadata), encoding="utf-8")
            with patch("builtins.print"):
                return_code = build_effective_potential.main(
                    [
                        "--input-force-csv",
                        str(force_csv),
                        "--input-metadata",
                        str(test_metadata),
                        "--output-directory",
                        str(root / "external_test_output"),
                    ]
                )
            self.assertEqual(return_code, 2)
            report = _read_json(
                root / "external_test_output" / "effective_potential_status.json"
            )
            self.assertIn("explicit --test-only-fixture", report["reason"])

            scaling = test_only_scaling_provenance()
            scaling.update(
                {
                    "source_semantics": (
                        PairForceScaling.UNSCALED_SINGLE_PAIR.value
                    ),
                    "scaling_applied_before_ingestion": True,
                    "source_unscaled_artifact_sha256": "a" * 64,
                    "scaled_force_table_sha256": force_hash,
                    "applied_by": "independent scaling workflow v1",
                    "source": "calibrated contact dataset",
                    "calibration_id": "CAL-SET-001",
                }
            )
            short_range = test_only_short_range_provenance()
            short_range.update(
                {
                    "validation_status": "PASSED",
                    "minimum_distance_m": 0.0,
                    "source_artifact_sha256": "b" * 64,
                    "method": "validated repulsive core closure",
                    "source": "calibrated short-range closure study",
                    "calibration_id": "CAL-SET-001",
                }
            )
            physical = dict(metadata)
            physical.update(
                {
                    "workflow_status": "PHYSICAL_PASSED",
                    "physical_status": "CALIBRATED_PHYSICAL_PAIR_DATA",
                    "calibration_status": "CALIBRATED_LAB_SET",
                    "calibration_id": "CAL-SET-001",
                    "time_scale_status": "SATISFIED",
                    "artifact_id": "CALIBRATED_PAIR_TABLE_SET_1",
                    "source": "calibrated two-gel contact study",
                    "solver_status": "ALL_SAMPLES_CONVERGED:TestFEM@1.0",
                    "native_scaling": (
                        PairForceScaling.UNSCALED_SINGLE_PAIR.value
                    ),
                    "scaling_conversion": {
                        "source_dataset_id": "CALIBRATED_PAIR_TABLE_SET_1",
                        "population_count": 1,
                        "population_count_provenance": (
                            "calibrated experiment uses one-particle reference count"
                        ),
                        "source_scaling": (
                            PairForceScaling.UNSCALED_SINGLE_PAIR.value
                        ),
                        "target_scaling": (
                            PairForceScaling.KAC_NORMALIZED_PROBABILITY.value
                        ),
                    },
                    "scaling_provenance": scaling,
                    "short_range_closure_provenance": short_range,
                }
            )
            physical_metadata = root / "physical_metadata.json"
            physical_metadata.write_text(json.dumps(physical), encoding="utf-8")
            with patch("builtins.print"):
                return_code = build_effective_potential.main(
                    [
                        "--input-force-csv",
                        str(force_csv),
                        "--input-metadata",
                        str(physical_metadata),
                        "--output-directory",
                        str(root / "incomplete_physical_output"),
                    ]
                )
            self.assertEqual(return_code, 2)
            report = _read_json(
                root
                / "incomplete_physical_output"
                / "effective_potential_status.json"
            )
            self.assertIn("physical_provenance is required", report["reason"])

            nonphysical_labels = dict(physical)
            nonphysical_labels.update(
                {
                    "physical_status": "CALIBRATED_PHYSICAL_TEST_ONLY",
                    "calibration_status": "CALIBRATED_TEST_ONLY",
                }
            )
            nonphysical_metadata = root / "nonphysical_labels.json"
            nonphysical_metadata.write_text(
                json.dumps(nonphysical_labels), encoding="utf-8"
            )
            with patch("builtins.print"):
                return_code = build_effective_potential.main(
                    [
                        "--input-force-csv",
                        str(force_csv),
                        "--input-metadata",
                        str(nonphysical_metadata),
                        "--output-directory",
                        str(root / "nonphysical_output"),
                    ]
                )
            self.assertEqual(return_code, 2)
            report = _read_json(
                root / "nonphysical_output" / "effective_potential_status.json"
            )
            self.assertIn("nonphysical", report["reason"])

            failed_solver = dict(physical)
            failed_solver["solver_status"] = "FAILED"
            failed_solver_metadata = root / "failed_solver.json"
            failed_solver_metadata.write_text(
                json.dumps(failed_solver), encoding="utf-8"
            )
            with patch("builtins.print"):
                return_code = build_effective_potential.main(
                    [
                        "--input-force-csv",
                        str(force_csv),
                        "--input-metadata",
                        str(failed_solver_metadata),
                        "--output-directory",
                        str(root / "failed_solver_output"),
                    ]
                )
            self.assertEqual(return_code, 2)
            report = _read_json(
                root / "failed_solver_output" / "effective_potential_status.json"
            )
            self.assertIn("ALL_SAMPLES_CONVERGED", report["reason"])

            contradictory = dict(physical)
            contradictory["physical_provenance"] = {
                "hydrogel_parameters": {
                    "network_density_times_solvent_volume": 0.025,
                    "flory_huggins_chi": 0.31,
                    "initial_polymer_volume_fraction": 0.18,
                    "delta_chemical_potential_over_kbt": 0.12,
                    "thermal_energy_density_pa": 2.4e5,
                    "calibration_status": "CALIBRATED_LAB_SET",
                    "calibration_id": "CAL-SET-001",
                    "source": "calibrated rheology and swelling dataset",
                    "units": {
                        "network_density_times_solvent_volume": "1",
                        "flory_huggins_chi": "1",
                        "initial_polymer_volume_fraction": "1",
                        "delta_chemical_potential_over_kbt": "1",
                        "thermal_energy_density_pa": "Pa",
                    },
                },
                "geometry": {
                    "model": "two calibrated spheres",
                    "particle_1_undeformed_radius_m": 2.0e-6,
                    "particle_2_undeformed_radius_m": 2.0e-6,
                    "source": "calibrated microscopy geometry",
                    "calibration_id": "CAL-SET-001",
                },
                "equilibrium_assumptions": {
                    "model": "quasi-static gel equilibrium",
                    "assumptions": "fixed bath chemical potential per sweep",
                    "source": "calibrated equilibrium protocol",
                },
                "contact_solver": {
                    "name": "TestFEM",
                    "version": "1.0",
                    "configuration_id": "contact-config-1",
                    "implementation_revision": "solver-revision-1",
                    "method": "mixed finite element contact solve",
                },
                "mesh_and_resolution": {
                    "method": "three-level mesh refinement",
                    "selected_resolution_id": "mesh-level-3",
                    "convergence_study_artifact_sha256": "c" * 64,
                    "resolution_count": 3,
                    "convergence_passed": True,
                    "maximum_relative_change": 0.01,
                    "upper_limit": 0.02,
                },
                "boundary_and_contact_conditions": {
                    "mechanical_boundary_conditions": "fixed rigid modes",
                    "solvent_bath_boundary_conditions": "fixed bath potential",
                    "contact_law": "frictionless nonpenetrating contact",
                    "reaction_force_sign_convention": "positive repulsion",
                    "source": "calibrated contact protocol",
                },
                "time_scale_assessment": {
                    "tau_gel_s": 0.01,
                    "tau_swarm_s": 1.0,
                    "ratio": 0.01,
                    "maximum_ratio": 0.1,
                    "passed": True,
                    "source": "calibrated relaxation experiment",
                },
                "distance_sweep": {
                    "sample_count": 17,
                    "minimum_distance_m": 0.0,
                    "reference_distance_m": 4.0e-6,
                    "spacing_or_sampling_rule": "uniform centre distance",
                    "source": "calibrated sweep protocol",
                },
                "upstream_validation": {
                    "overall_passed": True,
                    "report_sha256": "d" * 64,
                    "validation_errors": {"force_relative_error": 2.0},
                    "acceptance_thresholds": {"force_relative_error": 1.0},
                    "gates": {"force_relative_error": True},
                },
            }
            contradictory_metadata = root / "contradictory.json"
            contradictory_metadata.write_text(
                json.dumps(contradictory), encoding="utf-8"
            )
            with patch("builtins.print"):
                return_code = build_effective_potential.main(
                    [
                        "--input-force-csv",
                        str(force_csv),
                        "--input-metadata",
                        str(contradictory_metadata),
                        "--output-directory",
                        str(root / "contradictory_output"),
                    ]
                )
            self.assertEqual(return_code, 2)
            report = _read_json(
                root / "contradictory_output" / "effective_potential_status.json"
            )
            self.assertIn("contradicts error/threshold", report["reason"])

    def test_strict_metadata_reader_rejects_nonfinite_and_duplicate_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cases = {
                "nonfinite.json": '{"value": NaN}',
                "duplicate.json": '{"value": 1, "value": 2}',
            }
            for filename, content in cases.items():
                path = root / filename
                path.write_text(content, encoding="utf-8")
                with self.subTest(filename=filename):
                    with self.assertRaises(ValueError):
                        read_json_object(path)

            table = test_only_pair_force_table()
            extra_column_csv = root / "extra_column.csv"
            write_pair_force_csv(extra_column_csv, table)
            lines = extra_column_csv.read_text(encoding="utf-8").splitlines()
            lines[1] = lines[1] + ",unexpected"
            extra_column_csv.write_text("\n".join(lines), encoding="utf-8")
            metadata = pair_force_metadata_payload(
                table,
                data_file_sha256=sha256_file(extra_column_csv),
                scaling_provenance=test_only_scaling_provenance(),
                short_range_closure_provenance=(
                    test_only_short_range_provenance()
                ),
                workflow_status="TEST_ONLY_PASSED",
                calibration_status=TEST_ONLY_NOT_CALIBRATED,
            )
            extra_column_metadata = root / "extra_column.json"
            extra_column_metadata.write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "extra columns"):
                load_pair_force_table(
                    extra_column_csv, extra_column_metadata
                )


class ParticleMVComparisonScriptTests(unittest.TestCase):
    def test_missing_physical_potential_fails_closed_without_numerics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            with patch("builtins.print"):
                return_code = compare_particles_mv.main(
                    ["--output-directory", str(output_directory)]
                )

            self.assertEqual(return_code, 2)
            status = _read_json(
                output_directory / "particles_mv_comparison_status.json"
            )
            self.assertEqual(status["workflow_status"], "BLOCKED")
            self.assertEqual(status["physical_status"], "NOT_EVALUATED")
            self.assertFalse(status["numerical_results_generated"])
            self.assertEqual(status["generated_data_files"], [])
            self.assertEqual(
                set(status["missing_inputs"]),
                {
                    "--effective-potential-csv",
                    "--effective-potential-metadata",
                },
            )
            self.assertEqual(
                list(output_directory.glob("*particles_mv_comparison.json")),
                [],
            )

    def test_runtime_failure_is_reported_as_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            with (
                patch("builtins.print"),
                patch.object(
                    compare_particles_mv,
                    "_run_comparison",
                    side_effect=RuntimeError("solver stopped"),
                ),
            ):
                return_code = compare_particles_mv.main(
                    [
                        "--test-only-fixture",
                        "--output-directory",
                        str(output_directory),
                    ]
                )
            self.assertEqual(return_code, 2)
            status = _read_json(
                output_directory / "particles_mv_comparison_status.json"
            )
            self.assertEqual(status["workflow_status"], "BLOCKED")
            self.assertIn("solver stopped", status["reason"])

    def test_fixture_uses_one_kac_potential_and_passes_existing_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            with patch("builtins.print"):
                return_code = compare_particles_mv.main(
                    [
                        "--test-only-fixture",
                        "--output-directory",
                        str(output_directory),
                    ]
                )

            self.assertEqual(return_code, 0)
            report = _read_json(
                output_directory / "TEST_ONLY_particles_mv_comparison.json"
            )
            status = _read_json(
                output_directory / "particles_mv_comparison_status.json"
            )
            self.assertTrue(report["test_only_fixture"])
            self.assertTrue(report["overall_passed"])
            self.assertEqual(report["workflow_status"], "TEST_ONLY_PASSED")
            self.assertEqual(
                report["physical_status"], TEST_ONLY_NOT_CALIBRATED
            )
            self.assertEqual(
                report["calibration_status"], TEST_ONLY_NOT_CALIBRATED
            )
            self.assertIn(
                "NOT_HYDROGEL_PHYSICAL_VALIDATION",
                report["validation_scope"],
            )
            self.assertEqual(report["configuration"]["seed"], 20260824)

            potential = report["potential"]
            self.assertTrue(
                potential[
                    "same_python_object_used_by_particles_and_continuum"
                ]
            )
            self.assertTrue(potential["force_potential_validation_passed"])
            self.assertEqual(
                potential["scaling"],
                PairForceScaling.KAC_NORMALIZED_PROBABILITY.value,
            )

            checks = report["checks"]
            self.assertTrue(all(check["passed"] for check in checks.values()))
            self.assertTrue(
                checks["same_effective_pair_potential_instance"]["passed"]
            )
            self.assertEqual(
                checks["continuum_mass_error"]["upper_limit"], 1.0e-12
            )
            self.assertEqual(
                checks["no_material_negative_mass_clipping"]["upper_limit"],
                0.0,
            )
            self.assertEqual(
                checks["passive_free_energy_nonincrease"][
                    "upper_limit_joule"
                ],
                1.0e-30,
            )
            self.assertEqual(
                checks["particle_MV_relative_L2"]["upper_limit"], 0.25
            )
            self.assertEqual(
                checks["particle_MV_JS_divergence"]["upper_limit_nats"],
                0.05,
            )
            self.assertEqual(
                checks["particle_MV_centroid_trajectory"]["upper_limit_m"],
                2.0e-7,
            )
            self.assertEqual(
                checks["particle_MV_covariance"]["upper_limit"], 0.35
            )

            metrics = report["metrics"]
            self.assertLessEqual(
                metrics["maximum_absolute_continuum_mass_error"], 1.0e-12
            )
            self.assertEqual(metrics["total_clipped_negative_mass"], 0.0)
            self.assertLessEqual(
                metrics["continuum_maximum_positive_energy_increment_joule"],
                1.0e-30,
            )
            self.assertLess(
                metrics["continuum_final_free_energy_joule"],
                metrics["continuum_initial_free_energy_joule"],
            )
            self.assertLessEqual(metrics["relative_L2_density_error"], 0.25)
            self.assertLessEqual(metrics["JS_divergence_nats"], 0.05)
            self.assertLessEqual(
                metrics["centroid_trajectory_RMSE_m"], 2.0e-7
            )
            self.assertLessEqual(
                metrics["maximum_covariance_relative_error"], 0.35
            )
            self.assertEqual(status["workflow_status"], "TEST_ONLY_PASSED")
            self.assertTrue(status["overall_passed"])
            self.assertEqual(
                status["generated_data_files"],
                ["TEST_ONLY_particles_mv_comparison.json"],
            )

    def test_fixture_can_consume_validated_builder_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            build_directory = root / "build"
            comparison_directory = root / "comparison"
            with patch("builtins.print"):
                build_return_code = build_effective_potential.main(
                    [
                        "--test-only-fixture",
                        "--output-directory",
                        str(build_directory),
                    ]
                )
                compare_return_code = compare_particles_mv.main(
                    [
                        "--test-only-fixture",
                        "--effective-potential-csv",
                        str(
                            build_directory
                            / "TEST_ONLY_effective_potential.csv"
                        ),
                        "--effective-potential-metadata",
                        str(
                            build_directory
                            / "TEST_ONLY_effective_potential_metadata.json"
                        ),
                        "--output-directory",
                        str(comparison_directory),
                    ]
                )
            self.assertEqual(build_return_code, 0)
            self.assertEqual(compare_return_code, 0)
            report = _read_json(
                comparison_directory
                / "TEST_ONLY_particles_mv_comparison.json"
            )
            self.assertEqual(
                report["potential"]["input_mode"],
                "VALIDATED_TEST_ONLY_EFFECTIVE_ARTIFACT",
            )
            self.assertIsNotNone(
                report["potential"]["source_metadata_sha256"]
            )
            self.assertTrue(report["overall_passed"])

    def test_same_seed_reproduces_core_metrics_and_trajectories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reports = []
            for run_number in (1, 2):
                output_directory = root / f"run_{run_number}"
                with patch("builtins.print"):
                    return_code = compare_particles_mv.main(
                        [
                            "--test-only-fixture",
                            "--seed",
                            "20260824",
                            "--output-directory",
                            str(output_directory),
                        ]
                    )
                self.assertEqual(return_code, 0)
                reports.append(
                    _read_json(
                        output_directory
                        / "TEST_ONLY_particles_mv_comparison.json"
                    )
                )

            reproducible_fields = (
                "configuration",
                "potential",
                "physical_parameters",
                "overall_passed",
                "checks",
                "metrics",
                "trajectories",
                "gate_provenance",
            )
            for field in reproducible_fields:
                self.assertEqual(reports[0][field], reports[1][field], field)


if __name__ == "__main__":
    unittest.main()
