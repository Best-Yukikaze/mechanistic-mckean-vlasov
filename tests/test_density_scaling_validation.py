from __future__ import annotations

import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import (
    build_effective_potential,
    compare_particles_mv,
    validate_density_scaling,
)
from scripts._phase6_common import (
    CONTINUUM_READY_CLASSIFICATION,
    PARTICLE_ONLY_SHORT_RANGE_UNRESOLVED,
    effective_potential_from_table,
    load_effective_potential_artifact,
    read_json_object,
    test_only_pair_force_table,
)
from mechanistic_mv.mechanics.pair_interaction import (
    PairForceScaling,
    PairForceTable,
    convert_single_pair_table_to_kac,
)


def _assert_finite(test_case: unittest.TestCase, value: object) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite(test_case, item)
    elif isinstance(value, list):
        for item in value:
            _assert_finite(test_case, item)
    elif value is None:
        test_case.fail("successful validation JSON must not contain null")
    elif isinstance(value, bool) or isinstance(value, str):
        return
    elif isinstance(value, (int, float)):
        test_case.assertTrue(math.isfinite(value))
    else:
        test_case.fail(f"unsupported JSON value type: {type(value).__name__}")


class DensityScalingValidationScriptTests(unittest.TestCase):
    def test_requires_explicit_test_only_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "result.json"
            with patch("builtins.print"):
                return_code = validate_density_scaling.main(
                    ["--output", str(output)]
                )
            self.assertEqual(return_code, 2)
            self.assertFalse(output.exists())
            status = read_json_object(
                output.parent / "density_scaling_validation_status.json"
            )
            self.assertEqual(status["workflow_status"], "BLOCKED")
            self.assertEqual(status["generated_data_files"], [])
            self.assertFalse(status["numerical_results_generated"])
            self.assertIn("--test-only-fixture", status["reason"])

    def test_report_schema_is_strict_finite_and_test_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "result.json"
            with patch("builtins.print"):
                return_code = validate_density_scaling.main(
                    [
                        "--test-only-fixture",
                        "--seed",
                        "20260824",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(return_code, 0)
            report = read_json_object(output)
            self.assertEqual(
                report["schema_name"],
                "mechanistic_mv.density_scaling_validation",
            )
            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(
                report["artifact_type"], "DENSITY_SCALING_VALIDATION"
            )
            self.assertEqual(report["workflow_status"], "TEST_ONLY_PASSED")
            self.assertEqual(
                report["physical_status"], "TEST_ONLY_NOT_FINAL_PHYSICS"
            )
            self.assertTrue(report["test_only_fixture"])
            self.assertTrue(report["overall_passed"])
            self.assertEqual(
                report["calibration_status"], "TEST_ONLY_NOT_CALIBRATED"
            )
            self.assertEqual(
                report["overall_conclusion"],
                "TEST_ONLY_PROBABILITY_AND_NUMBER_CONVENTIONS_EQUIVALENT",
            )
            self.assertEqual(
                report["parameters"]["fixture_name"],
                "TEST_ONLY_GAUSSIAN_REPULSION_NOT_HYDROGEL",
            )
            self.assertEqual(report["parameters"]["rng_bit_generator"], "PCG64")
            physical_parameters = report["parameters"]["physical_parameters"]
            for name in (
                "particle_mass_kg",
                "drag_coefficient_kg_per_s",
                "temperature_kelvin",
                "mobility_m_per_newton_second",
                "diffusion_m2_per_s",
                "thermal_energy_joule",
            ):
                self.assertIn(name, physical_parameters)
            self.assertEqual(report["parameters"]["continuum_cfl_safety"], 0.83)
            self.assertEqual(report["parameters"]["particle_pair_chunk_size"], 3)
            self.assertTrue(report["scientific_limitations"])
            self.assertIn(
                "NOT_HYDROGEL_PHYSICAL_VALIDATION",
                report["validation_scope"],
            )
            self.assertEqual(len(report["deterministic_evidence_sha256"]), 64)
            _assert_finite(self, report)

    def test_same_seed_reproduces_deterministic_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reports = []
            for name in ("first.json", "second.json"):
                output = root / name
                with patch("builtins.print"):
                    return_code = validate_density_scaling.main(
                        [
                            "--test-only-fixture",
                            "--seed",
                            "20260824",
                            "--output",
                            str(output),
                        ]
                    )
                self.assertEqual(return_code, 0)
                reports.append(read_json_object(output))
            for key in (
                "formulas",
                "parameters",
                "thresholds",
                "checks",
                "rejection_gates",
                "overall_passed",
                "deterministic_evidence_sha256",
            ):
                self.assertEqual(reports[0][key], reports[1][key])

    def test_reported_conventions_satisfy_all_numerical_gates(self) -> None:
        report = validate_density_scaling.run_validation(20260824)
        required_checks = {
            "empirical_probability_mass",
            "empirical_number_mass",
            "initial_probability_mass",
            "initial_number_mass",
            "empirical_number_equals_N_probability",
            "initial_number_equals_N_probability",
            "kac_potential_scale",
            "particle_force_equivalence",
            "one_step_langevin_equivalence",
            "particle_total_energy_equivalence",
            "direct_convolution_convention_equivalence",
            "fft_convolution_convention_equivalence",
            "probability_direct_fft_equivalence",
            "number_direct_fft_equivalence",
            "x_face_number_flux_equals_N_probability_flux",
            "y_face_number_flux_equals_N_probability_flux",
            "one_fvm_step_number_equals_N_probability",
            "continuum_total_free_energy_scaling",
        }
        self.assertTrue(required_checks.issubset(report["checks"]))
        for name, check in report["checks"].items():
            with self.subTest(check=name):
                self.assertTrue(check["passed"])
                if "maximum_absolute_error" in check:
                    self.assertLessEqual(
                        check["maximum_absolute_error"],
                        check["maximum_allowed_absolute_error"],
                    )
        formulas = report["formulas"]
        self.assertEqual(formulas["density"], "n(x)=N*rho(x)")
        self.assertEqual(formulas["pair_potential"], "W_Kac(r)=N*W_pair(r)")
        self.assertEqual(report["parameters"]["population_count"], 6)

    def test_required_contract_mismatches_are_rejected(self) -> None:
        report = validate_density_scaling.run_validation(20260824)
        required = {
            "scaling_mismatch",
            "population_mismatch",
            "probability_mass_mismatch",
            "number_mass_mismatch",
            "particle_only_short_range",
        }
        self.assertEqual(set(report["rejection_gates"]), required)
        for name, gate in report["rejection_gates"].items():
            with self.subTest(gate=name):
                self.assertTrue(gate["passed"])
                self.assertEqual(gate["expected_outcome"], "REJECTED")
                self.assertEqual(gate["observed_outcome"], "REJECTED")
                self.assertTrue(gate["exception_type"])
                self.assertTrue(gate["reason"])
        short_range = report["rejection_gates"][
            "particle_only_short_range"
        ]
        self.assertFalse(short_range["continuum_ready"])
        self.assertEqual(
            short_range["short_range_classification"],
            PARTICLE_ONLY_SHORT_RANGE_UNRESOLVED,
        )


class ShortRangeArtifactTests(unittest.TestCase):
    @staticmethod
    def _build(root: Path, minimum_distance_m: float) -> tuple[Path, Path]:
        arguments = [
            "--test-only-fixture",
            "--test-only-minimum-distance-m",
            str(minimum_distance_m),
            "--output-directory",
            str(root),
        ]
        with patch("builtins.print"):
            return_code = build_effective_potential.main(arguments)
        if return_code != 0:
            raise AssertionError(f"builder returned {return_code}")
        return (
            root / "TEST_ONLY_effective_potential.csv",
            root / "TEST_ONLY_effective_potential_metadata.json",
        )

    def test_positive_r_min_is_preserved_and_marked_particle_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_path, metadata_path = self._build(root, 0.5e-6)
            with data_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            distances = [float(row["center_distance_m"]) for row in rows]
            self.assertEqual(len(distances), 17)
            self.assertEqual(distances[0], 0.5e-6)
            self.assertNotIn(0.0, distances)
            metadata = read_json_object(metadata_path)
            status = read_json_object(root / "effective_potential_status.json")
            for payload in (metadata, status):
                self.assertEqual(
                    payload["minimum_supported_distance_m"], 0.5e-6
                )
                self.assertFalse(payload["continuum_ready"])
                self.assertEqual(
                    payload["short_range_classification"],
                    PARTICLE_ONLY_SHORT_RANGE_UNRESOLVED,
                )
                self.assertEqual(
                    payload["continuum_admission_status"], "BLOCKED"
                )
            potential, loaded_metadata = load_effective_potential_artifact(
                data_path, metadata_path
            )
            self.assertEqual(potential.minimum_supported_distance_m, 0.5e-6)
            self.assertFalse(potential.continuum_ready)
            self.assertEqual(
                loaded_metadata["short_range_classification"],
                PARTICLE_ONLY_SHORT_RANGE_UNRESOLVED,
            )

    def test_zero_r_min_is_continuum_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_path, metadata_path = self._build(root, 0.0)
            metadata = read_json_object(metadata_path)
            self.assertEqual(metadata["minimum_supported_distance_m"], 0.0)
            self.assertTrue(metadata["continuum_ready"])
            self.assertEqual(
                metadata["short_range_classification"],
                CONTINUUM_READY_CLASSIFICATION,
            )
            potential, _ = load_effective_potential_artifact(
                data_path, metadata_path
            )
            self.assertTrue(potential.continuum_ready)

    def test_loader_rejects_contradictory_short_range_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_path, metadata_path = self._build(root, 0.5e-6)
            original = read_json_object(metadata_path)
            cases = {
                "minimum_supported_distance_m": 0.0,
                "continuum_ready": True,
                "short_range_classification": CONTINUUM_READY_CLASSIFICATION,
                "continuum_admission_status": "PASSED",
                "usage_scope": "PARTICLES_AND_CONTINUUM",
            }
            for field, value in cases.items():
                changed = dict(original)
                changed[field] = value
                changed_path = root / f"bad_{field}.json"
                changed_path.write_text(json.dumps(changed), encoding="utf-8")
                with self.subTest(field=field):
                    with self.assertRaisesRegex(ValueError, "contradicts"):
                        load_effective_potential_artifact(
                            data_path, changed_path
                        )

    def test_loader_rejects_population_scaling_evidence_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_path, metadata_path = self._build(root, 0.0)
            changed = read_json_object(metadata_path)
            pair_metadata = changed["pair_force_metadata"]
            pair_metadata["native_scaling"] = (
                PairForceScaling.UNSCALED_SINGLE_PAIR.value
            )
            pair_metadata["scaling_conversion"] = {
                "source_dataset_id": pair_metadata["dataset_id"],
                "population_count": 5,
                "population_count_provenance": "TEST_ONLY fixed N=5",
                "source_scaling": PairForceScaling.UNSCALED_SINGLE_PAIR.value,
                "target_scaling": (
                    PairForceScaling.KAC_NORMALIZED_PROBABILITY.value
                ),
            }
            changed["scaling_provenance"].update(
                {
                    "source_semantics": (
                        PairForceScaling.UNSCALED_SINGLE_PAIR.value
                    ),
                    "population_or_concentration_value": 4.0,
                    "force_multiplier": 4.0,
                    "energy_multiplier": 4.0,
                }
            )
            changed_path = root / "population_mismatch.json"
            changed_path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disagrees"):
                load_effective_potential_artifact(data_path, changed_path)

    def test_compare_blocks_particle_only_before_numerics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact_directory = root / "artifact"
            data_path, metadata_path = self._build(
                artifact_directory, 0.5e-6
            )
            output_directory = root / "comparison"
            with (
                patch("builtins.print"),
                patch.object(compare_particles_mv, "_run_comparison") as run,
            ):
                return_code = compare_particles_mv.main(
                    [
                        "--test-only-fixture",
                        "--effective-potential-csv",
                        str(data_path),
                        "--effective-potential-metadata",
                        str(metadata_path),
                        "--output-directory",
                        str(output_directory),
                    ]
                )
            self.assertEqual(return_code, 2)
            run.assert_not_called()
            status = read_json_object(
                output_directory / "particles_mv_comparison_status.json"
            )
            self.assertEqual(status["workflow_status"], "BLOCKED")
            self.assertFalse(status["numerical_results_generated"])
            self.assertEqual(status["generated_data_files"], [])
            self.assertEqual(
                status["short_range_classification"],
                PARTICLE_ONLY_SHORT_RANGE_UNRESOLVED,
            )
            self.assertIn(
                PARTICLE_ONLY_SHORT_RANGE_UNRESOLVED,
                status["reason"],
            )
            self.assertEqual(
                list(output_directory.glob("*particles_mv_comparison.json")),
                [],
            )

    def test_compare_rejects_population_mismatch_before_solver(self) -> None:
        direct_table = test_only_pair_force_table()
        unscaled_metadata = replace(
            direct_table.metadata,
            scaling=PairForceScaling.UNSCALED_SINGLE_PAIR,
            native_scaling=PairForceScaling.UNSCALED_SINGLE_PAIR,
        )
        unscaled_table = PairForceTable(
            direct_table.center_distance_m,
            direct_table.radial_force_newton / 7.0,
            unscaled_metadata,
        )
        converted = convert_single_pair_table_to_kac(
            unscaled_table,
            population_count=7,
            population_count_provenance="TEST_ONLY fixed N=7",
        )
        potential = effective_potential_from_table(converted)
        with patch.object(compare_particles_mv, "McKeanVlasovSolver") as solver:
            with self.assertRaisesRegex(ValueError, "population_count"):
                compare_particles_mv._run_comparison(
                    potential,
                    {"particle_count": 5},
                )
        solver.assert_not_called()


if __name__ == "__main__":
    unittest.main()
