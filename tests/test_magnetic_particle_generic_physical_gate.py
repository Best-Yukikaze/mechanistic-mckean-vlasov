from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_magnetic_particle_generic_physical_gate as validation
from scripts import run_mv_physical_validation as legacy_validation


class MagneticParticleGenericPhysicalGateTests(unittest.TestCase):
    def test_legacy_generic_cli_module_reexports_the_canonical_entrypoint(self) -> None:
        self.assertIs(legacy_validation.main, validation.main)

    def _copy_provenance(self, root: Path) -> Path:
        target = root / "generic__source_provenance__r1.json"
        shutil.copy2(validation.DEFAULT_PROVENANCE, target)
        return target

    @staticmethod
    def _json(path: Path) -> dict[str, object]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):  # pragma: no cover - fixture guard
            raise AssertionError("expected JSON object")
        return value

    def _run(self, root: Path, provenance: Path) -> tuple[int, Path]:
        output = root / "out"
        with patch("builtins.print"):
            return_code = validation.main(
                ["--provenance", str(provenance), "--output-dir", str(output)]
            )
        return return_code, output

    def test_source_register_is_complete_and_preserves_fail_closed_transfers(self) -> None:
        payload = validation.load_provenance(validation.DEFAULT_PROVENANCE)
        sources = payload["sources"]
        records = payload["scalar_records"]
        self.assertEqual(len(sources), 6)
        self.assertGreaterEqual(len(records), 25)
        checks = {item["transfer_id"]: item for item in payload["transfer_checks"]}
        self.assertEqual(
            checks["S1B_TO_S1_2026_FORCE"]["status"],
            "REJECTED_MISSING_BATCH_AND_FIELD_COMPATIBILITY",
        )
        audit = payload["s4_digitization_audit"]
        self.assertFalse(audit["admissible_as_particle_number_density"])
        self.assertEqual(audit["fabricated_numeric_records"], 0)

    def test_si_velocity_and_thesis_force_are_calculated_without_cross_source_promotion(self) -> None:
        payload = validation.load_provenance(validation.DEFAULT_PROVENANCE)
        phase_a = validation.build_phase_a(payload)
        rows = phase_a["observations"]
        first = rows[0]
        self.assertAlmostEqual(first["velocity_m_per_s"], 0.42e-3 / 3600.0, places=20)
        self.assertIsNone(first["force_newton"])
        self.assertIsNone(first["mobility_m_per_N_s"])
        self.assertIsNone(first["einstein_diffusivity_m2_per_s"])
        candidate_force = phase_a["s1b_candidate_force"]
        expected = 0.281 * 3.69e-25 * 0.55 * 45.0 / (4.0e-7 * math.pi)
        self.assertAlmostEqual(candidate_force["value_newton"], expected, places=30)
        self.assertEqual(
            candidate_force["transfer_to_s1_2026"],
            "REJECTED_MISSING_BATCH_AND_FIELD_COMPATIBILITY",
        )
        self.assertEqual(phase_a["acceptance_gate"]["result"], "NOT_EVALUABLE")
        self.assertFalse(
            phase_a["descriptive_velocity_cv"]["8nm"]["constant_mobility_gate_applied"]
        )

    def test_default_report_stops_before_physics_and_has_one_allowed_decision(self) -> None:
        payload = validation.load_provenance(validation.DEFAULT_PROVENANCE)
        report = validation.build_report(payload, validation.DEFAULT_PROVENANCE)
        self.assertIn(report["final_decision"], validation.FINAL_OUTCOMES)
        self.assertEqual(
            report["final_decision"],
            "CURRENT_2D_MV_PHYSICAL_REDUCTION_INVALID",
        )
        self.assertTrue(report["final_decision_is_exclusive"])
        self.assertEqual(
            report["phase_a"]["status"],
            "BLOCKED_SAME_BATCH_MAGNETIC_FORCE_UNAVAILABLE",
        )
        self.assertEqual(
            report["phase_b"]["status"], "BLOCKED_PHASE_A_TRANSPORT_NOT_LOCKED"
        )
        self.assertEqual(
            report["phase_c"]["physics_2d_closure"]["status"],
            "CURRENT_2D_MV_PHYSICAL_REDUCTION_INVALID",
        )
        self.assertFalse(report["scope"]["parameter_fit_performed"])
        self.assertFalse(report["scope"]["magnetic_simulation_run"])
        self.assertFalse(report["scope"]["gym_rl_dqn_run"])
        self.assertFalse(report["scope"]["training_run"])

    def test_cli_writes_all_phase_artifacts_without_fake_joint_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            provenance = self._copy_provenance(root)
            return_code, output = self._run(root, provenance)
            self.assertEqual(return_code, 0)
            expected_stems = (
                "generic__transport_closure__r1",
                "generic__magnetic_drift__r1",
                "generic__dipolar_interaction__r1",
                "generic__joint_mv__r1",
            )
            for stem in expected_stems:
                self.assertTrue((output / f"{stem}.json").is_file())
                self.assertTrue((output / f"{stem}.csv").is_file())
                self.assertTrue((output / f"{stem}.md").is_file())
            report = self._json(output / f"{validation.REPORT_ARTIFACT_STEM}.json")
            self.assertEqual(report["provenance"]["sha256_role"], "OPTIONAL_ARCHIVAL_PROVENANCE_NOT_AN_ADMISSION_GATE")
            self.assertEqual(report["phase_d"]["observation"]["digitized_records"], [])
            self.assertEqual(report["phase_d"]["observation"]["fabricated_records"], 0)
            with (output / f"{validation.PHASE_ARTIFACT_STEMS['phase_a']}.csv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(row["force_newton"] == "" for row in rows))

    def test_cross_source_promotion_and_fake_s4_numeric_data_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for case in ("s1b_promotion", "fake_s4"):
                with self.subTest(case=case):
                    case_root = root / case
                    case_root.mkdir()
                    provenance = self._copy_provenance(case_root)
                    payload = self._json(provenance)
                    if case == "s1b_promotion":
                        payload["transfer_checks"][0]["status"] = "ADMITTED"
                    else:
                        payload["s4_digitization_audit"]["fabricated_numeric_records"] = 1
                    provenance.write_text(json.dumps(payload), encoding="utf-8")
                    return_code, output = self._run(case_root, provenance)
                    self.assertEqual(return_code, 2)
                    report = self._json(output / f"{validation.REPORT_ARTIFACT_STEM}.json")
                    self.assertEqual(report["evidence_status"], "BLOCKED_INVALID_SOURCE_PROVENANCE")
                    self.assertEqual(report["physical_execution"], "NOT_RUN")

    def test_nonfinite_or_bad_json_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for case, content in (
                ("nan", '{"schema": NaN}'),
                ("broken", "{"),
            ):
                with self.subTest(case=case):
                    case_root = root / case
                    case_root.mkdir()
                    provenance = case_root / "source.json"
                    provenance.write_text(content, encoding="utf-8")
                    return_code, output = self._run(case_root, provenance)
                    self.assertEqual(return_code, 2)
                    report = self._json(output / f"{validation.REPORT_ARTIFACT_STEM}.json")
                    self.assertEqual(report["evidence_status"], "BLOCKED_INVALID_SOURCE_PROVENANCE")
                    self.assertFalse(report["scope"]["magnetic_simulation_run"])


if __name__ == "__main__":
    unittest.main()
