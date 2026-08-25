from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
import math
from pathlib import Path
import tempfile
import unittest

from mechanistic_mv.mechanics.hydrogel import HydrogelParameters
from mechanistic_mv.mechanics.mg10f_parameters import (
    MG10FParameterRegistry,
    MaterialNotReadyError,
    PhysicalParameter,
    VerificationStatus,
    default_mg10f_config_path,
    load_mg10f_registry,
)


class MG10FParameterRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = load_mg10f_registry()
        self.raw_config = json.loads(
            default_mg10f_config_path().read_text(encoding="utf-8")
        )

    @staticmethod
    def _record(config: dict[str, object], name: str) -> dict[str, object]:
        parameters = config["parameters"]
        assert isinstance(parameters, list)
        for record in parameters:
            assert isinstance(record, dict)
            if record["name"] == name:
                return record
        raise AssertionError(f"test configuration has no record {name!r}")

    @staticmethod
    def _load_temporary_json(text: str) -> MG10FParameterRegistry:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "physics_mg10f.json"
            path.write_text(text, encoding="utf-8")
            return load_mg10f_registry(path)

    def test_nominal_values_units_and_blocked_calibration_are_explicit(self) -> None:
        temperature = self.registry.parameter("temperature_20c")
        radius = self.registry.parameter("hydrodynamic_radius_20c")
        modulus = self.registry.parameter("young_modulus_20c")
        molecular_volume = self.registry.parameter("water_molecular_volume")
        self.assertEqual(temperature.value, 293.15)
        self.assertEqual(temperature.unit, "K")
        self.assertEqual(radius.value, 4.63e-7)
        self.assertEqual(radius.uncertainty, 7.0e-9)
        self.assertEqual(radius.unit, "m")
        self.assertIn("reference contact radius", radius.notes)
        self.assertEqual(modulus.value, 166000.0)
        self.assertEqual(modulus.uncertainty, 24000.0)
        self.assertEqual(modulus.unit, "Pa")
        self.assertEqual(molecular_volume.value, 2.99e-29)
        self.assertEqual(molecular_volume.unit, "m^3/molecule")
        self.assertEqual(
            self.registry.parameter("network_density_times_solvent_volume").value,
            None,
        )
        self.assertEqual(
            self.registry.parameter("initial_polymer_volume_fraction").value,
            None,
        )
        self.assertEqual(
            self.registry.parameter("delta_chemical_potential_over_kbt").value,
            None,
        )
        self.assertEqual(
            self.registry.blocked_calibration_parameters,
            (
                "network_density_times_solvent_volume",
                "initial_polymer_volume_fraction",
                "delta_chemical_potential_over_kbt",
            ),
        )

    def test_registry_and_records_are_immutable_and_json_serializable(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            self.registry.registry_status = "VERIFIED"  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            self.registry.parameter("temperature_20c").value = 300.0  # type: ignore[misc]
        exported = self.registry.as_jsonable()
        reloaded = json.loads(json.dumps(exported))
        self.assertEqual(reloaded["registry_id"], self.registry.registry_id)
        self.assertEqual(reloaded["parameters"][0]["name"], "material")
        self.assertIn("derived_transport_parameters", reloaded)
        self.assertEqual(reloaded["material_readiness"]["hydrogel_conversion"], "BLOCKED")

    def test_nominal_transport_uses_stokes_einstein_formulas(self) -> None:
        derived = {
            parameter.name: parameter
            for parameter in self.registry.nominal_transport_parameters()
        }
        self.assertTrue(
            math.isclose(
                float(derived["stokes_drag_coefficient"].value),
                8.74479908045579e-9,
                rel_tol=1.0e-14,
            )
        )
        self.assertTrue(
            math.isclose(
                float(derived["mobility"].value),
                1.14353685e8,
                rel_tol=1.0e-8,
            )
        )
        self.assertTrue(
            math.isclose(
                float(derived["diffusion"].value),
                4.62831965e-13,
                rel_tol=1.0e-8,
            )
        )
        self.assertTrue(
            math.isclose(
                float(derived["thermal_energy"].value),
                4.0473725435e-21,
                rel_tol=1.0e-14,
            )
        )
        self.assertTrue(
            math.isclose(
                float(derived["thermal_energy_density"].value),
                1.353636302173913e8,
                rel_tol=1.0e-14,
            )
        )
        self.assertTrue(
            all(
                parameter.verification_status
                is VerificationStatus.CONDITIONAL_NOMINAL_DERIVED_FROM_UNVERIFIED_INPUTS
                for parameter in derived.values()
            )
        )
        radius_relative_uncertainty = 7.0e-9 / 4.63e-7
        for name in ("stokes_drag_coefficient", "mobility", "diffusion"):
            parameter = derived[name]
            self.assertTrue(
                math.isclose(
                    float(parameter.uncertainty),
                    abs(float(parameter.value)) * radius_relative_uncertainty,
                    rel_tol=1.0e-14,
                )
            )
            self.assertIn("not a total uncertainty", parameter.notes)
        self.assertIsNone(derived["thermal_energy"].uncertainty)

    def test_missing_source_location_cannot_promote_a_value_to_verified(self) -> None:
        radius = self.registry.parameter("hydrodynamic_radius_20c")
        self.assertEqual(
            radius.verification_status,
            VerificationStatus.SOURCE_LOCATION_NEEDS_VERIFICATION,
        )
        with self.assertRaisesRegex(ValueError, "requires a source_location"):
            replace(radius, verification_status=VerificationStatus.VERIFIED)
        with self.assertRaises(MaterialNotReadyError):
            self.registry.assert_verified_material_ready()
        self.assertFalse(self.registry.is_verified_material_ready)

    def test_missing_calibration_values_block_material_readiness_and_hydrogel_conversion(
        self,
    ) -> None:
        self.assertFalse(self.registry.is_material_ready)
        with self.assertRaisesRegex(MaterialNotReadyError, "missing calibration values"):
            self.registry.assert_verified_material_ready()
        with self.assertRaisesRegex(
            MaterialNotReadyError, "cannot construct HydrogelParameters"
        ):
            HydrogelParameters(**self.registry.to_hydrogel_parameters())

    def test_hydrodynamic_radius_cannot_be_requested_as_contact_reference_radius(self) -> None:
        with self.assertRaisesRegex(MaterialNotReadyError, "not a reference contact radius"):
            self.registry.reference_contact_radius_m()

    def test_chi_utility_uses_the_registered_temperature_dependent_coefficients(self) -> None:
        self.assertTrue(
            math.isclose(
                self.registry.flory_huggins_chi(0.0, 293.15),
                0.233024,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        )
        self.assertTrue(
            math.isclose(
                self.registry.flory_huggins_chi(1.0, 293.15),
                1.472789,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        )

    def test_null_values_must_stay_blocked(self) -> None:
        blocked = self.registry.parameter("network_density_times_solvent_volume")
        raw = blocked.as_jsonable()
        raw["verification_status"] = "UNVERIFIED"
        with self.assertRaisesRegex(ValueError, "null parameter value must be BLOCKED"):
            PhysicalParameter.from_jsonable(raw)

    def test_nominal_json_edits_cannot_promote_material_or_calibration_contract(self) -> None:
        promoted_status = deepcopy(self.raw_config)
        promoted_status["registry_status"] = "VERIFIED"
        with self.assertRaisesRegex(ValueError, "must retain registry_status"):
            MG10FParameterRegistry.from_jsonable(promoted_status)

        filled = deepcopy(self.raw_config)
        self._record(filled, "network_density_times_solvent_volume")["value"] = 0.1
        self._record(filled, "network_density_times_solvent_volume")[
            "verification_status"
        ] = "UNVERIFIED"
        self._record(filled, "initial_polymer_volume_fraction")["value"] = 0.2
        self._record(filled, "initial_polymer_volume_fraction")[
            "verification_status"
        ] = "UNVERIFIED"
        self._record(filled, "delta_chemical_potential_over_kbt")["value"] = 0.0
        self._record(filled, "delta_chemical_potential_over_kbt")[
            "verification_status"
        ] = "UNVERIFIED"
        filled_registry = MG10FParameterRegistry.from_jsonable(filled)
        self.assertFalse(filled_registry.is_material_ready)
        with self.assertRaisesRegex(MaterialNotReadyError, "cannot be promoted"):
            filled_registry.assert_verified_material_ready()

        invalid_unit = deepcopy(self.raw_config)
        self._record(invalid_unit, "network_density_times_solvent_volume")["unit"] = "m"
        with self.assertRaisesRegex(ValueError, "must use unit"):
            MG10FParameterRegistry.from_jsonable(invalid_unit)
        invalid_role = deepcopy(self.raw_config)
        self._record(invalid_role, "initial_polymer_volume_fraction")[
            "observable_role"
        ] = "OTHER"
        with self.assertRaisesRegex(ValueError, "invalid observable_role"):
            MG10FParameterRegistry.from_jsonable(invalid_role)
        invalid_provenance = deepcopy(self.raw_config)
        self._record(invalid_provenance, "delta_chemical_potential_over_kbt")[
            "provenance_type"
        ] = "MEASURED"
        with self.assertRaisesRegex(ValueError, "CALIBRATED provenance"):
            MG10FParameterRegistry.from_jsonable(invalid_provenance)
        invalid_type = deepcopy(self.raw_config)
        self._record(invalid_type, "network_density_times_solvent_volume")[
            "value"
        ] = True
        self._record(invalid_type, "network_density_times_solvent_volume")[
            "verification_status"
        ] = "UNVERIFIED"
        with self.assertRaisesRegex(ValueError, "must be numeric or null"):
            MG10FParameterRegistry.from_jsonable(invalid_type)

    def test_json_loader_rejects_duplicate_nonfinite_and_unknown_content(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            self._load_temporary_json('{"schema_version": 1, "schema_version": 1}')
        for token in ("NaN", "Infinity", "-Infinity"):
            with self.assertRaisesRegex(ValueError, "non-finite JSON constant"):
                self._load_temporary_json(f'{{"schema_version": {token}}}')

        unknown_root = deepcopy(self.raw_config)
        unknown_root["unexpected"] = "not allowed"
        with self.assertRaisesRegex(ValueError, "unexpected keys"):
            self._load_temporary_json(json.dumps(unknown_root))
        unknown_record = deepcopy(self.raw_config)
        self._record(unknown_record, "temperature_20c")["unexpected"] = "not allowed"
        with self.assertRaisesRegex(ValueError, "unexpected keys"):
            self._load_temporary_json(json.dumps(unknown_record))

    def test_stokes_inputs_require_strict_si_finite_positive_values(self) -> None:
        wrong_unit = deepcopy(self.raw_config)
        self._record(wrong_unit, "hydrodynamic_radius_20c")["unit"] = "nm"
        with self.assertRaisesRegex(MaterialNotReadyError, "must use SI unit"):
            MG10FParameterRegistry.from_jsonable(wrong_unit).nominal_transport_parameters()
        nonpositive = deepcopy(self.raw_config)
        self._record(nonpositive, "water_viscosity_20c")["value"] = 0.0
        with self.assertRaisesRegex(MaterialNotReadyError, "strictly positive"):
            MG10FParameterRegistry.from_jsonable(nonpositive).nominal_transport_parameters()

    def test_missing_transport_input_exports_structured_blocked_report(self) -> None:
        missing_input = deepcopy(self.raw_config)
        parameters = missing_input["parameters"]
        assert isinstance(parameters, list)
        missing_input["parameters"] = [
            record
            for record in parameters
            if isinstance(record, dict) and record["name"] != "water_viscosity_20c"
        ]
        exported = MG10FParameterRegistry.from_jsonable(missing_input).as_jsonable()
        self.assertEqual(exported["transport_readiness"]["status"], "BLOCKED")
        self.assertEqual(exported["derived_transport_parameters"], [])
        self.assertIn(
            {"parameter": "water_viscosity_20c", "reason": "MISSING_PARAMETER"},
            exported["transport_readiness"]["blockers"],
        )
        json.dumps(exported, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
