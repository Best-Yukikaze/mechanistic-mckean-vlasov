"""Direct source/arithmetical checks only; no PDE, Gym or training runs."""
from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_lyons_constitutive_precheck_v2 as audit


class LyonsConstitutivePrecheckV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = audit.load_source(audit.DEFAULT_SOURCE)

    def run_cli(self, folder: Path, source: Path) -> tuple[int, dict]:
        with patch("builtins.print"):
            code = audit.main(["--source", str(source), "--output-dir", str(folder)])
        report = json.loads((folder / "lyons_constitutive_precheck_v2.json").read_text(encoding="utf-8"))
        return code, report

    def test_normal_evidence_exit_two_is_scientific_insufficiency(self) -> None:
        # Historical evidence must remain byte-for-byte untouched by this CLI.
        historical = {p: p.read_bytes() for p in audit.DEFAULT_OUTPUT.glob("*v1.*")}
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            code, report = self.run_cli(folder, audit.DEFAULT_SOURCE)
            self.assertEqual(code, 2)
            self.assertEqual(report["status"], audit.INSUFFICIENT)
            self.assertEqual(report["input_status"], "VALID_SOURCE_REGISTER_WITH_INCOMPLETE_PHYSICAL_DATA")
            self.assertEqual(report["project_status"], audit.PROJECT_STATUS)
            self.assertEqual(report["quantities"], self.source["quantities"])
            self.assertEqual(report["sources"], self.source["sources"])
            self.assertEqual(report["lineage"], self.source["lineage"])
            self.assertTrue(all(v is False for v in report["scope"].values()))
            md = (folder / "lyons_constitutive_precheck_v2.md").read_text(encoding="utf-8")
            for label in (audit.INSUFFICIENT, audit.PROJECT_STATUS, "NOT SENT", "0.281", "0.289", "0.23 T", "0.55 T"):
                self.assertIn(label, md)
            self.assertFalse(report["author_request"]["sent"])
            self.assertIn(report["author_request"]["draft"], md)
            with (folder / report["generated_data_files"][0]).open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 16)
            for csv_row, json_row in zip(rows, report["digitized_front"]["rows"]):
                self.assertEqual(csv_row["provenance"], "DIGITIZED_AUTHOR_MANUSCRIPT_NOT_RAW_DATA")
                for key in ("time_h", "distance_to_base_mm", "pixel_x", "pixel_y"):
                    self.assertEqual(float(csv_row[key]), json_row[key])
            # This catches NaN/Infinity in all nested numeric output fields.
            json.dumps(report, allow_nan=False)
        for path, original in historical.items():
            self.assertEqual(path.read_bytes(), original)

    def test_conditional_force_arithmetic_not_admitted_transport(self) -> None:
        report = audit.build_report(self.source)
        for case, chi in zip(report["conditional_recipes"], (0.289, 0.281)):
            expected_force = chi * 3.69e-25 * 0.55 * 45 / (4e-7 * math.pi)
            self.assertAlmostEqual(case["force_N"] / expected_force, 1.0, places=14)
            expected_m = (0.37e-3 / 3600) / expected_force
            self.assertAlmostEqual(case["M_front_apparent_m_per_N_s"] / expected_m, 1.0, places=14)
            self.assertAlmostEqual(case["M_front_velocity_SD_only_m_per_N_s"] / expected_m, 0.02 / 0.37)
            self.assertIsNone(case["force_complete_standard_uncertainty_N"])
            self.assertIsNone(case["M_eff_admitted_m_per_N_s"])
            self.assertIsNone(case["D_Einstein_admitted_m2_per_s"])
            self.assertIsNone(case["D0_at_transport_temperature_m2_per_s"])
            self.assertGreater(case["apparent_D_Einstein_over_D0_if_same_T"], 9)
        self.assertIsNone(report["constant_mobility_test"]["CV_M_across_force_levels"])
        self.assertEqual(report["front_to_drift"]["decision"], "UNVERIFIED_DO_NOT_LOCK_M_OR_D")
        self.assertFalse(report["digitized_front"]["raw_data_recovered"])
        self.assertFalse(report["digitized_front"]["constitutive_admission"])

    def test_no_hash_requirement_and_reproducible_report(self) -> None:
        source = copy.deepcopy(self.source)
        source.pop("source_sha256")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            loaded = audit.load_source(path)
        with patch.object(audit.subprocess, "check_output", return_value="fixed-head\n"):
            one, two = audit.build_report(loaded), audit.build_report(loaded)
        self.assertEqual(one, two)
        self.assertIsNone(one["source_sha256"]["value"])
        self.assertEqual(one["source_sha256"]["role"], "OPTIONAL_NOT_A_GATE")

    def test_bad_source_fails_closed_without_data_csv(self) -> None:
        malformed = ["{", '{"schema": NaN}', '{"schema": 1, "schema": 2}']
        for mutate in (
            lambda p: p["lineage"].update(coating="PEG2000"),
            lambda p: p["quantities"]["nominal_magnet_B"].update(value=0.23),
            lambda p: p["quantities"]["chi_thesis"].update(value=0.289),
            lambda p: p["quantities"]["cited_gradient"].update(unit="T"),
            lambda p: p["quantities"]["front_velocity"].update(value=float("inf")),
            lambda p: p["correction"].update(observable="v_exp / d_hyd"),
            lambda p: p["admission"].update(physical_parameter_lock_allowed=True),
            lambda p: p["figure_digitization"]["calibration"].update(time_h=[0, 0]),
        ):
            source = copy.deepcopy(self.source)
            mutate(source)
            malformed.append(json.dumps(source))
        for index, text in enumerate(malformed):
            with self.subTest(case=index), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "bad.json"
                path.write_text(text, encoding="utf-8")
                folder = Path(tmp) / "out"
                code, report = self.run_cli(folder, path)
                self.assertEqual(code, 2)
                self.assertEqual(report["input_status"], "INVALID_SOURCE")
                self.assertEqual(report["status"], audit.INSUFFICIENT)
                self.assertEqual(report["generated_data_files"], [])
                self.assertEqual(list(folder.glob("*.csv")), [])
                self.assertTrue((folder / "lyons_constitutive_precheck_v2.md").is_file())

    def test_missing_source_writes_status_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "out"
            code, report = self.run_cli(folder, Path(tmp) / "absent.json")
            self.assertEqual(code, 2)
            self.assertEqual(report["input_status"], "INVALID_SOURCE")
            self.assertEqual(report["generated_data_files"], [])


if __name__ == "__main__":
    unittest.main()
