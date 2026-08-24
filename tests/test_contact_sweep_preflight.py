from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from mechanistic_mv.mechanics.pair_interaction import (
    ContactInputPurpose,
    ContactProblemInput,
    ContactProblemReadiness,
    RequiredInputReport,
    RequiredInputStatus,
    ShortRangeInputStatus,
    SolverInputStatus,
)

from scripts import run_pair_contact_sweep


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"expected a JSON object in {path}")
    return value


def _required_field_count() -> int:
    return sum(
        len(paths)
        for paths in ContactProblemInput.required_input_field_paths().values()
    )


class PairContactSweepPreflightTests(unittest.TestCase):
    def assert_no_contact_result_artifacts(self, output_directory: Path) -> None:
        self.assertEqual(list(output_directory.rglob("*.csv")), [])
        self.assertEqual(list(output_directory.rglob("*.png")), [])

    def test_no_manifest_is_field_level_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            with patch("builtins.print"):
                return_code = run_pair_contact_sweep.main(
                    ["--output-directory", str(output_directory)]
                )

            self.assertEqual(return_code, 2)
            status = _read_object(
                output_directory / "pair_contact_sweep_status.json"
            )
            self.assertEqual(status["workflow_status"], "BLOCKED")
            self.assertEqual(status["input_manifest_status"], "MISSING")
            self.assertEqual(status["input_readiness"], "BLOCKED")
            self.assertEqual(
                status["contact_fem_backend_status"],
                run_pair_contact_sweep.CONTACT_FEM_BACKEND_UNAVAILABLE,
            )
            self.assertEqual(
                status["contact_results_status"],
                run_pair_contact_sweep.CONTACT_RESULTS_NOT_GENERATED,
            )
            required_inputs = status["required_inputs"]
            self.assertIsInstance(required_inputs, list)
            self.assertEqual(len(required_inputs), _required_field_count())
            self.assertTrue(
                all(
                    item["presence_status"] == "MISSING_KEY"
                    for item in required_inputs
                )
            )
            self.assertEqual(status["generated_data_files"], [])
            self.assertFalse(status["physical_results_present"])
            self.assert_no_contact_result_artifacts(output_directory)

    def test_template_contains_only_null_parameter_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_directory = root / "status"
            template_path = root / "contact_input_template.json"
            with patch("builtins.print"):
                return_code = run_pair_contact_sweep.main(
                    [
                        "--write-input-template",
                        str(template_path),
                        "--output-directory",
                        str(output_directory),
                    ]
                )

            self.assertEqual(return_code, 0)
            template = _read_object(template_path)
            self.assertEqual(
                template["template_status"],
                run_pair_contact_sweep.INPUT_TEMPLATE_NOT_PHYSICAL,
            )
            self.assertFalse(template["physical_values_supplied"])
            contact_problem = template["contact_problem"]
            self.assertIsInstance(contact_problem, dict)
            leaves: list[object] = []

            def collect(value: object) -> None:
                if isinstance(value, dict):
                    for child in value.values():
                        collect(child)
                else:
                    leaves.append(value)

            collect(contact_problem)
            self.assertEqual(len(leaves), _required_field_count())
            self.assertTrue(all(value is None for value in leaves))
            status = _read_object(
                output_directory / "pair_contact_sweep_status.json"
            )
            self.assertEqual(
                status["physical_status"],
                run_pair_contact_sweep.INPUT_TEMPLATE_NOT_PHYSICAL,
            )
            self.assertEqual(status["generated_data_files"], [])
            self.assertEqual(
                status["generated_parameter_collection_files"],
                [str(template_path)],
            )
            self.assert_no_contact_result_artifacts(output_directory)

    def test_bad_json_is_rejected_with_hash_of_exact_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_directory = root / "status"
            manifest_path = root / "bad.json"
            raw = b'{"schema_name":"bad",}'
            manifest_path.write_bytes(raw)
            with patch("builtins.print"):
                return_code = run_pair_contact_sweep.main(
                    [
                        "--input-manifest",
                        str(manifest_path),
                        "--output-directory",
                        str(output_directory),
                    ]
                )

            self.assertEqual(return_code, 2)
            status = _read_object(
                output_directory / "pair_contact_sweep_status.json"
            )
            self.assertEqual(status["input_manifest_status"], "REJECTED")
            self.assertEqual(
                status["input_manifest_sha256"], hashlib.sha256(raw).hexdigest()
            )
            self.assertIn("strict UTF-8 JSON", status["reason"])
            self.assert_no_contact_result_artifacts(output_directory)

    def test_test_only_manifest_is_never_admitted_as_physical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_directory = root / "status"
            manifest_path = root / "test_only.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_name": run_pair_contact_sweep.MANIFEST_SCHEMA_NAME,
                        "schema_version": 1,
                        "input_purpose": (
                            ContactInputPurpose.TEST_ONLY_CONTRACT.value
                        ),
                        "contact_problem": {},
                    }
                ),
                encoding="utf-8",
            )
            with patch("builtins.print"):
                return_code = run_pair_contact_sweep.main(
                    [
                        "--input-manifest",
                        str(manifest_path),
                        "--output-directory",
                        str(output_directory),
                    ]
                )

            self.assertEqual(return_code, 2)
            status = _read_object(
                output_directory / "pair_contact_sweep_status.json"
            )
            self.assertEqual(status["workflow_status"], "BLOCKED")
            self.assertEqual(status["input_manifest_status"], "REJECTED")
            self.assertIn("TEST_ONLY", status["reason"])
            self.assertEqual(status["generated_data_files"], [])
            self.assert_no_contact_result_artifacts(output_directory)

    def test_partial_physical_manifest_reports_present_and_missing_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_directory = root / "status"
            manifest_path = root / "partial.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_name": run_pair_contact_sweep.MANIFEST_SCHEMA_NAME,
                        "schema_version": 1,
                        "input_purpose": ContactInputPurpose.PHYSICAL.value,
                        "contact_problem": {
                            "geometry": {"first_radius_m": None}
                        },
                    }
                ),
                encoding="utf-8",
            )
            expected_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            with patch("builtins.print"):
                return_code = run_pair_contact_sweep.main(
                    [
                        "--input-manifest",
                        str(manifest_path),
                        "--output-directory",
                        str(output_directory),
                    ]
                )

            self.assertEqual(return_code, 2)
            status = _read_object(
                output_directory / "pair_contact_sweep_status.json"
            )
            self.assertEqual(status["input_manifest_sha256"], expected_hash)
            self.assertEqual(status["input_readiness"], "BLOCKED")
            field_reports = {
                item["field_path"]: item for item in status["required_inputs"]
            }
            self.assertEqual(
                field_reports["geometry.first_radius_m"]["presence_status"],
                "EXPLICIT_NULL",
            )
            self.assertEqual(
                field_reports["geometry.second_radius_m"]["presence_status"],
                "MISSING_KEY",
            )
            self.assert_no_contact_result_artifacts(output_directory)

    def test_all_null_declaration_does_not_bypass_strong_types(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_directory = root / "status"
            manifest_path = root / "all_null.json"
            contact_problem = run_pair_contact_sweep._template_payload()[
                "contact_problem"
            ]
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_name": run_pair_contact_sweep.MANIFEST_SCHEMA_NAME,
                        "schema_version": 1,
                        "input_purpose": ContactInputPurpose.PHYSICAL.value,
                        "contact_problem": contact_problem,
                    }
                ),
                encoding="utf-8",
            )
            with patch("builtins.print"):
                return_code = run_pair_contact_sweep.main(
                    [
                        "--input-manifest",
                        str(manifest_path),
                        "--output-directory",
                        str(output_directory),
                    ]
                )

            self.assertEqual(return_code, 2)
            status = _read_object(
                output_directory / "pair_contact_sweep_status.json"
            )
            self.assertEqual(status["workflow_status"], "BLOCKED")
            self.assertEqual(status["input_manifest_status"], "REJECTED")
            self.assertEqual(status["generated_data_files"], [])
            self.assert_no_contact_result_artifacts(output_directory)

    def test_verified_input_readiness_still_cannot_generate_fem_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_directory = root / "status"
            manifest_path = root / "declared_physical.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_name": run_pair_contact_sweep.MANIFEST_SCHEMA_NAME,
                        "schema_version": 1,
                        "input_purpose": ContactInputPurpose.PHYSICAL.value,
                        "contact_problem": {},
                    }
                ),
                encoding="utf-8",
            )
            reports = tuple(
                RequiredInputReport(
                    requirement_id=requirement_id,
                    input_field_paths=paths,
                    status=RequiredInputStatus.READY,
                    reason="input and source record are verified",
                )
                for requirement_id, paths in (
                    ContactProblemInput.required_input_field_paths().items()
                )
            )
            readiness = ContactProblemReadiness(
                solver_input_status=SolverInputStatus.READY_FOR_CONTACT_SOLVER,
                short_range_status=ShortRangeInputStatus.PARTICLE_ONLY,
                required_inputs=reports,
                blocking_reasons=(),
                physical_inputs_verified=True,
            )
            problem = SimpleNamespace(
                purpose=ContactInputPurpose.PHYSICAL,
                evaluate_readiness=lambda: readiness,
            )
            with (
                patch.object(
                    run_pair_contact_sweep,
                    "_decode_contact_problem",
                    return_value=(problem, {}),
                ),
                patch("builtins.print"),
            ):
                return_code = run_pair_contact_sweep.main(
                    [
                        "--input-manifest",
                        str(manifest_path),
                        "--output-directory",
                        str(output_directory),
                    ]
                )

            self.assertEqual(return_code, 2)
            status = _read_object(
                output_directory / "pair_contact_sweep_status.json"
            )
            self.assertEqual(
                status["input_readiness"],
                SolverInputStatus.READY_FOR_CONTACT_SOLVER.value,
            )
            self.assertEqual(status["workflow_status"], "BLOCKED")
            self.assertEqual(
                status["contact_fem_backend_status"],
                run_pair_contact_sweep.CONTACT_FEM_BACKEND_UNAVAILABLE,
            )
            self.assertEqual(
                status["contact_results_status"],
                run_pair_contact_sweep.CONTACT_RESULTS_NOT_GENERATED,
            )
            self.assertFalse(status["physical_results_present"])
            self.assertFalse(status["pair_force_table_generated"])
            self.assertEqual(status["generated_data_files"], [])
            self.assert_no_contact_result_artifacts(output_directory)


if __name__ == "__main__":
    unittest.main()
