from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mechanistic_mv.continuum.flux import DriftFluxScheme

from scripts import run_second_order_flux_validation


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"expected a JSON object in {path}")
    return value


class SecondOrderFluxValidationTests(unittest.TestCase):
    def test_actual_physics_api_selects_declared_first_and_second_order_schemes(
        self,
    ) -> None:
        discovery = run_second_order_flux_validation.discover_flux_api()

        self.assertEqual(
            discovery.reference.identifier,
            DriftFluxScheme.FIRST_ORDER_UPWIND.name,
        )
        self.assertIsNotNone(discovery.candidate)
        assert discovery.candidate is not None
        self.assertIsInstance(discovery.candidate.value, DriftFluxScheme)
        self.assertIn("SECOND_ORDER", discovery.candidate.identifier)
        self.assertEqual(
            discovery.reference.constructor_keyword, "drift_flux_scheme"
        )
        self.assertEqual(
            discovery.status, "READY_FOR_TEST_ONLY_COMPARISON"
        )

    def test_missing_selectable_scheme_api_fails_closed(self) -> None:
        class LegacySolver:
            def __init__(self, grid: object, parameters: object, pair: object) -> None:
                del grid, parameters, pair

        discovery = run_second_order_flux_validation.discover_flux_api(LegacySolver)
        self.assertEqual(
            discovery.status,
            run_second_order_flux_validation.BLOCKED_NEW_FLUX_API_UNAVAILABLE,
        )
        self.assertIsNone(discovery.candidate)
        self.assertEqual(
            discovery.reference.identifier,
            "FIRST_ORDER_UPWIND_LEGACY_DEFAULT",
        )

    def test_test_only_report_applies_fixed_second_order_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "second_order_flux_validation.json"
            with patch("builtins.print"):
                return_code = run_second_order_flux_validation.main(
                    ["--output", str(output)]
                )

            report = _read_json(output)
            self.assertEqual(
                report["model_status"], "TEST_ONLY_NOT_FINAL_PHYSICS"
            )
            self.assertEqual(report["workflow_status"], "COMPLETED_TEST_ONLY")
            self.assertEqual(return_code, 0 if report["overall_passed"] else 2)
            self.assertTrue(report["all_json_values_finite"])
            scope = report["scope"]
            candidate_method = scope["candidate_method"]
            self.assertEqual(
                candidate_method["scheme_name"],
                report["scheme_api"]["candidate"]["identifier"],
            )
            formula_source = candidate_method["formula_source"]
            self.assertEqual(
                formula_source["module"], "mechanistic_mv.continuum.flux"
            )
            self.assertIn("compute_face_fluxes", formula_source["flux_assembly_symbol"])
            self.assertIn("flux.py", formula_source["source_file"])
            time_contract = candidate_method["time_step_contract_source"]
            self.assertIn("mckean_vlasov.py", time_contract["solver_file"])
            self.assertIn("outflow rate", time_contract["CFL_api_documentation"])
            self.assertIn("SSP-RK2", time_contract["flux_api_documentation"])
            self.assertIn(
                "not a claim of a formal temporal convergence order",
                report["diagnostic_interpretation"]["time_refinement"],
            )
            spatial_design = report["test_only_configuration"][
                "spatial_refinement_time_design"
            ]
            self.assertEqual(spatial_design["grid_sizes"], [32, 64, 128])
            self.assertEqual(spatial_design["step_counts"], [32, 128, 512])
            self.assertTrue(spatial_design["requires_one_internal_substep"])
            self.assertIn("O(h^4)", spatial_design["formula"])
            boundary_tail = report["test_only_configuration"][
                "boundary_tail_reference"
            ]
            self.assertIn("no-flux", boundary_tail["finite_volume_boundary_condition"])
            self.assertLess(
                boundary_tail["final_time"][
                    "conservative_probability_mass_outside_rectangle_upper_bound"
                ],
                1.0e-12,
            )

            candidate_checks = report["candidate_scheme_checks"]
            observed_order = report["candidate_scheme"]["grid_refinement"][-1][
                "observed_order_from_previous_grid"
            ]
            self.assertEqual(
                candidate_checks["fine_grid_observed_order_at_least_1p8"],
                observed_order >= 1.8,
            )
            self.assertEqual(
                report["acceptance_thresholds"]["candidate_fine_grid_observed_order"],
                1.8,
            )
            self.assertIn(
                "pre-set near-second-order acceptance line",
                report["acceptance_thresholds"][
                    "candidate_fine_grid_observed_order_rationale"
                ],
            )
            self.assertTrue(candidate_checks["all_mass_and_positivity_gates_pass"])
            self.assertTrue(candidate_checks["grid_error_decreases"])
            self.assertTrue(
                candidate_checks[
                    "spatial_time_schedule_is_h_squared_and_single_substep"
                ]
            )
            grid_records = report["candidate_scheme"]["grid_refinement"]
            self.assertEqual(
                [record["outer_steps"] for record in grid_records], [32, 128, 512]
            )
            self.assertTrue(
                all(
                    record["exactly_one_internal_substep_per_requested_step"]
                    and record["requested_step_within_observed_CFL_bound"]
                    for record in grid_records
                )
            )
            dt_over_h_squared = [
                record["spatial_time_refinement"]["dt_over_h_squared_s_per_m2"]
                for record in grid_records
            ]
            self.assertAlmostEqual(dt_over_h_squared[0], dt_over_h_squared[1])
            self.assertAlmostEqual(dt_over_h_squared[1], dt_over_h_squared[2])
            cfl_errors = [
                item["relative_L2_to_finest_CFL"]
                for item in report["candidate_scheme"]["adaptive_CFL_refinement"][:-1]
            ]
            self.assertEqual(
                candidate_checks["adaptive_CFL_error_decreases"],
                all(earlier > later for earlier, later in zip(cfl_errors[:-1], cfl_errors[1:])),
            )
            self.assertTrue(
                report["comparison_checks"][
                    "candidate_finest_grid_error_not_larger_than_reference"
                ]
            )

            for study_name in ("first_order_reference", "candidate_scheme"):
                study = report[study_name]
                for series_name in ("grid_refinement", "adaptive_CFL_refinement"):
                    for record in study[series_name]:
                        self.assertLessEqual(
                            record["maximum_absolute_mass_error"], 2.0e-12
                        )
                        self.assertGreaterEqual(
                            record["minimum_density_per_m2"], 0.0
                        )
                        self.assertEqual(record["clipped_negative_mass"], 0.0)
                        self.assertIn("free_energy_behavior", record)


if __name__ == "__main__":
    unittest.main()
