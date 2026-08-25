from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from typing import Any

from scripts import run_mg10f_transport_derivation as derivation
from scripts._phase6_common import read_json_object


def _json_payload() -> dict[str, Any]:
    value = json.loads(derivation.DEFAULT_REGISTRY.read_text(encoding="utf-8"))
    if not isinstance(value, dict):  # pragma: no cover - protects the fixture itself
        raise AssertionError("the repository MG10F fixture must be a JSON object")
    return value


def _parameter(payload: dict[str, Any], name: str) -> dict[str, Any]:
    parameters = payload["parameters"]
    if not isinstance(parameters, list):  # pragma: no cover - fixture assertion
        raise AssertionError("parameters must be a list")
    for parameter in parameters:
        if isinstance(parameter, dict) and parameter.get("name") == name:
            return parameter
    raise AssertionError(f"parameter {name!r} was not found")


class MG10FTransportDerivationTests(unittest.TestCase):
    def _run_payload(
        self,
        root: Path,
        payload: dict[str, Any],
        *,
        suffix: str = ".json",
    ) -> tuple[int, Path]:
        registry = root / f"registry{suffix}"
        registry.parent.mkdir(parents=True, exist_ok=True)
        registry.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        output = root / "output"
        return (
            derivation.main(
                ["--registry", str(registry), "--output-directory", str(output)]
            ),
            output,
        )

    def _assert_only_blocked_report(self, output: Path) -> dict[str, Any]:
        self.assertTrue(output.is_dir())
        self.assertEqual(
            {path.name for path in output.iterdir()},
            {derivation.DERIVED_TRANSPORT_FILENAME},
        )
        report = read_json_object(output / derivation.DERIVED_TRANSPORT_FILENAME)
        self.assertEqual(report["workflow_status"], derivation.BLOCKED_WORKFLOW_STATUS)
        self.assertEqual(report["physical_status"], "BLOCKED")
        self.assertEqual(report["calibration_status"], "BLOCKED")
        self.assertEqual(report["transport_status"], "BLOCKED")
        self.assertFalse(report["physical_model_ready"])
        self.assertEqual(report["downstream_status"], derivation._downstream_statuses())
        self.assertNotIn("derived_transport", report)
        return report

    def test_valid_nominal_registry_exports_conditional_transport_and_no_calibration(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "output"
            return_code = derivation.main(["--output-directory", str(output)])

            self.assertEqual(return_code, 0)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {derivation.DERIVED_TRANSPORT_FILENAME},
            )
            report = read_json_object(output / derivation.DERIVED_TRANSPORT_FILENAME)
            self.assertEqual(report["workflow_status"], derivation.WORKFLOW_STATUS)
            self.assertEqual(report["model_status"], derivation.MODEL_STATUS)
            self.assertEqual(report["physical_status"], derivation.PHYSICAL_STATUS)
            self.assertEqual(report["calibration_status"], derivation.CALIBRATION_STATUS)
            self.assertEqual(report["transport_status"], derivation.TRANSPORT_STATUS)
            self.assertTrue(report["workflow_completed"])
            self.assertFalse(report["physical_model_ready"])
            self.assertEqual(report["downstream_status"], derivation._downstream_statuses())
            self.assertEqual(
                report["blocked_calibration_parameters"],
                list(derivation._EXPECTED_BLOCKED_CALIBRATION_PARAMETERS),
            )
            self.assertEqual(report["outputs_written"], [derivation.DERIVED_TRANSPORT_FILENAME])
            self.assertEqual(
                report["downstream_status"],
                {
                    "hydrogel": "BLOCKED",
                    "contact_fem": "BLOCKED",
                    "F_pair": "BLOCKED",
                    "W_eff": "BLOCKED",
                    "mckean_vlasov": "BLOCKED",
                    "controller": "NOT_IN_SCOPE",
                },
            )
            self.assertEqual(
                report["source_verification"]["status"],
                "ALL_USABLE_REGISTRY_INPUT_SOURCES_UNVERIFIED",
            )
            self.assertEqual(
                report["source_verification"]["verified_usable_parameters"], []
            )
            self.assertTrue(report["source_verification"]["unverified_usable_parameters"])
            file_identity = report["registry"]["file_identity"]
            self.assertTrue(file_identity["uses_canonical_repository_config"])
            self.assertEqual(
                file_identity["path"], str(derivation.DEFAULT_REGISTRY.resolve())
            )
            self.assertEqual(
                file_identity["sha256"],
                hashlib.sha256(derivation.DEFAULT_REGISTRY.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                set(report["inputs"]),
                {"temperature_20c", "hydrodynamic_radius_20c", "water_viscosity_20c"},
            )
            self.assertTrue(
                all(
                    item["source_location"] is None
                    and item["verification_status"] == "SOURCE_LOCATION_NEEDS_VERIFICATION"
                    for item in report["inputs"].values()
                )
            )
            self.assertEqual(
                report["uncertainty_assessment"]["status"], "PARTIAL_R_H_ONLY"
            )
            self.assertEqual(
                report["uncertainty_assessment"]["included_uncertainties"],
                ["hydrodynamic_radius_20c"],
            )
            self.assertIn(
                "water_viscosity_20c",
                report["uncertainty_assessment"]["excluded_uncertainties"],
            )

            derived = {entry["name"]: entry for entry in report["derived_transport"]}
            self.assertEqual(
                set(derived),
                {
                    "stokes_drag_coefficient",
                    "mobility",
                    "thermal_energy",
                    "diffusion",
                },
            )
            self.assertTrue(
                math.isclose(
                    derived["stokes_drag_coefficient"]["value"],
                    8.74479908045579e-9,
                    rel_tol=1.0e-14,
                )
            )
            self.assertTrue(
                math.isclose(
                    derived["mobility"]["value"], 1.14353685e8, rel_tol=1.0e-8
                )
            )
            self.assertTrue(
                math.isclose(
                    derived["thermal_energy"]["value"],
                    4.0473725435e-21,
                    rel_tol=1.0e-14,
                )
            )
            self.assertTrue(
                math.isclose(
                    derived["diffusion"]["value"], 4.62831965e-13, rel_tol=1.0e-8
                )
            )
            expected_relative_radius_uncertainty = 7.0e-9 / 4.63e-7
            self.assertTrue(
                math.isclose(
                    derived["stokes_drag_coefficient"]["uncertainty"],
                    derived["stokes_drag_coefficient"]["value"]
                    * expected_relative_radius_uncertainty,
                    rel_tol=1.0e-14,
                )
            )
            self.assertTrue(
                math.isclose(
                    derived["mobility"]["uncertainty"],
                    derived["mobility"]["value"] * expected_relative_radius_uncertainty,
                    rel_tol=1.0e-14,
                )
            )
            self.assertTrue(
                math.isclose(
                    derived["diffusion"]["uncertainty"],
                    derived["diffusion"]["value"] * expected_relative_radius_uncertainty,
                    rel_tol=1.0e-14,
                )
            )
            self.assertEqual(
                derived["stokes_drag_coefficient"]["uncertainty_status"],
                "PARTIAL_PROPAGATED_FROM_R_H_ONLY",
            )
            self.assertEqual(
                derived["mobility"]["uncertainty_status"],
                "PARTIAL_PROPAGATED_FROM_R_H_ONLY",
            )
            self.assertEqual(
                derived["diffusion"]["uncertainty_status"],
                "PARTIAL_PROPAGATED_FROM_R_H_ONLY",
            )
            self.assertTrue(
                all(
                    entry["verification_status"]
                    == "CONDITIONAL_NOMINAL_DERIVED_FROM_UNVERIFIED_INPUTS"
                    for entry in derived.values()
                )
            )


    def test_non_json_suffix_fails_closed_without_success_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            return_code, output = self._run_payload(
                Path(temporary_directory), _json_payload(), suffix=".yaml"
            )
            self.assertEqual(return_code, 2)
            report = self._assert_only_blocked_report(output)
            self.assertIn(".json suffix", report["reason"])

    def test_malformed_duplicate_nonfinite_and_nonobject_json_fail_closed(self) -> None:
        bad_documents = {
            "malformed": "{not valid JSON",
            "duplicate": '{"schema_version": 1, "schema_version": 1}',
            "nonfinite": '{"schema_version": NaN}',
            "root_array": "[]",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for name, text in bad_documents.items():
                with self.subTest(name=name):
                    registry = root / f"{name}.json"
                    registry.write_text(text, encoding="utf-8")
                    output = root / f"{name}_output"
                    return_code = derivation.main(
                        [
                            "--registry",
                            str(registry),
                            "--output-directory",
                            str(output),
                        ]
                    )
                    self.assertEqual(return_code, 2)
                    self._assert_only_blocked_report(output)

    def test_wrong_missing_and_nonpositive_transport_inputs_fail_closed(self) -> None:
        mutations: dict[str, Any] = {
            "wrong_unit": lambda payload: _parameter(payload, "temperature_20c").update(
                {"unit": "degC"}
            ),
            "missing_input": lambda payload: payload.update(
                {
                    "parameters": [
                        parameter
                        for parameter in payload["parameters"]
                        if parameter["name"] != "water_viscosity_20c"
                    ]
                }
            ),
            "nonpositive_input": lambda payload: _parameter(
                payload, "hydrodynamic_radius_20c"
            ).update({"value": 0.0}),
            "missing_radius_uncertainty": lambda payload: _parameter(
                payload, "hydrodynamic_radius_20c"
            ).update({"uncertainty": None}),
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    payload = _json_payload()
                    mutate(payload)
                    return_code, output = self._run_payload(root / name, payload)
                    self.assertEqual(return_code, 2)
                    self._assert_only_blocked_report(output)

    def test_status_gate_rejects_registry_promotion_and_calibration_fill(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            falsely_promoted = _json_payload()
            falsely_promoted["registry_status"] = "VERIFIED"
            return_code, output = self._run_payload(root / "promoted", falsely_promoted)
            self.assertEqual(return_code, 2)
            report = self._assert_only_blocked_report(output)
            self.assertIn("registry_status", report["reason"])

            falsely_calibrated = _json_payload()
            for name in derivation._EXPECTED_BLOCKED_CALIBRATION_PARAMETERS:
                _parameter(falsely_calibrated, name).update(
                    {"value": 0.5, "verification_status": "UNVERIFIED"}
                )
            return_code, output = self._run_payload(root / "calibrated", falsely_calibrated)
            self.assertEqual(return_code, 2)
            report = self._assert_only_blocked_report(output)
            self.assertIn("explicit blocked", report["reason"])


if __name__ == "__main__":
    unittest.main()
