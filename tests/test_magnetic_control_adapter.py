"""Focused Controller-contract tests for the active magnetic validation path."""

from __future__ import annotations

import inspect
from pathlib import Path
import unittest

import numpy as np

from mechanistic_mv.mechanics.controlled_potential import (
    BoundMagneticFieldSnapshot,
    CoilCurrentCommand,
    LinearMagneticControlSourcePayload,
    MagneticFieldControlAdapter,
    MagneticSourceReference,
    SOURCE_PARAMETERIZED_MAGNETIC_CONTROL_STATUS,
    SourceBackedAffineCoilMagneticFieldMap,
    TestOnlyUniformFieldPotential,
)
from mechanistic_mv.mechanics.magnetic_particle_potential import (
    MU0_N_PER_A2,
    LinearMagneticParticle,
    MagneticParticlePotential,
)


class MagneticControlAdapterTests(unittest.TestCase):
    def _field_map(self) -> SourceBackedAffineCoilMagneticFieldMap:
        particle_law = LinearMagneticParticle(
            chi_v_dimensionless=1.5,
            particle_volume_m3=2.0e-18,
            source_locator="unit-fixture://linear-particle-law",
            provenance_class="TEST_ONLY_NOT_CALIBRATED",
        )
        source_payload = LinearMagneticControlSourcePayload(
            particle_law=particle_law,
            flux_density_provenance=MagneticSourceReference(
                source_id="unit-field-map",
                locator="unit-fixture://B-map",
                quantity="B(x; I)",
                units="T",
                provenance_class="TEST_ONLY_NOT_CALIBRATED",
            ),
            flux_density_gradient_provenance=MagneticSourceReference(
                source_id="unit-gradient-map",
                locator="unit-fixture://grad-B-map",
                quantity="grad_xy(B)(x; I)",
                units="T/m",
                provenance_class="TEST_ONLY_NOT_CALIBRATED",
            ),
        )
        return SourceBackedAffineCoilMagneticFieldMap(
            source_payload=source_payload,
            zero_current_flux_density_tesla=0.05,
            flux_density_per_ampere_tesla=0.01,
            flux_density_gradient_per_ampere_tesla_per_m=(100.0, -50.0),
            minimum_current_ampere=-0.5,
            maximum_current_ampere=0.5,
        )

    def test_command_field_snapshot_is_consumed_by_canonical_physics_law(self) -> None:
        field_map = self._field_map()
        adapter = MagneticFieldControlAdapter(field_map=field_map)
        command = CoilCurrentCommand(0.2)
        snapshot = adapter.adapt_command(command)
        potential = adapter.physics_potential_for(command)
        points = np.asarray([[1.0e-6, 2.0e-6], [3.0e-6, -1.0e-6]])

        self.assertIsInstance(snapshot, BoundMagneticFieldSnapshot)
        self.assertIsInstance(potential, MagneticParticlePotential)
        self.assertEqual(potential.magnetic_field, snapshot)
        self.assertIs(potential.particle_law, field_map.source_payload.particle_law)

        field = 0.05 + 0.2 * (
            0.01 + np.asarray([100.0, -50.0]) @ points.T
        )
        gradient = np.broadcast_to(np.asarray([20.0, -10.0]), points.shape)
        np.testing.assert_allclose(snapshot.flux_density_tesla(points), field)
        np.testing.assert_allclose(
            snapshot.gradient_flux_density_tesla_per_m(points), gradient
        )

        law = field_map.source_payload.particle_law
        expected_energy = -law.chi_v_dimensionless * law.particle_volume_m3 * field**2 / (
            2.0 * MU0_N_PER_A2
        )
        expected_force = (
            law.chi_v_dimensionless
            * law.particle_volume_m3
            * field[:, np.newaxis]
            * gradient
            / MU0_N_PER_A2
        )
        np.testing.assert_allclose(potential.potential_joule(points), expected_energy)
        np.testing.assert_allclose(potential.force_newton(points), expected_force)

        point = points[0]
        step_m = 1.0e-10
        finite_difference_gradient = np.empty(2)
        for axis in range(2):
            offset = np.zeros(2)
            offset[axis] = step_m
            finite_difference_gradient[axis] = (
                potential.potential_joule(point + offset)
                - potential.potential_joule(point - offset)
            ) / (2.0 * step_m)
        np.testing.assert_allclose(
            potential.force_newton(point),
            -finite_difference_gradient,
            rtol=2.0e-6,
            atol=0.0,
        )

    def test_provenance_payload_is_complete_and_fail_closed(self) -> None:
        field_map = self._field_map()
        metadata = field_map.configuration_as_jsonable()
        payload = metadata["source_payload"]
        self.assertEqual(
            payload["magnetic_law_owner"], "Physics.LinearMagneticParticle"
        )
        self.assertIn("chi_v_dimensionless", payload)
        self.assertIn("magnetic_particle_volume_m3", payload)
        self.assertIn("chi_v_and_linear_magnetization_provenance", payload)
        self.assertEqual(payload["flux_density_provenance"]["units"], "T")
        self.assertEqual(
            payload["flux_density_gradient_provenance"]["units"], "T/m"
        )
        self.assertEqual(
            field_map.physical_status,
            SOURCE_PARAMETERIZED_MAGNETIC_CONTROL_STATUS,
        )

        with self.assertRaises(ValueError):
            MagneticSourceReference(
                source_id="",
                locator="unit-fixture://missing",
                quantity="B",
                units="T",
                provenance_class="TEST_ONLY_NOT_CALIBRATED",
            )
        with self.assertRaises(TypeError):
            LinearMagneticControlSourcePayload(  # type: ignore[arg-type]
                particle_law=object(),
                flux_density_provenance=field_map.source_payload.flux_density_provenance,
                flux_density_gradient_provenance=(
                    field_map.source_payload.flux_density_gradient_provenance
                ),
            )
        with self.assertRaises(TypeError):
            LinearMagneticControlSourcePayload(  # type: ignore[call-arg]
                field_map.source_payload.particle_law,
                field_map.source_payload.flux_density_provenance,
            )
        with self.assertRaises(ValueError):
            MagneticFieldControlAdapter(field_map=field_map).adapt_command(
                CoilCurrentCommand(0.6)
            )

        negative_field_map = SourceBackedAffineCoilMagneticFieldMap(
            source_payload=field_map.source_payload,
            zero_current_flux_density_tesla=0.01,
            flux_density_per_ampere_tesla=-1.0,
            flux_density_gradient_per_ampere_tesla_per_m=(0.0, 0.0),
            minimum_current_ampere=0.0,
            maximum_current_ampere=0.5,
        )
        with self.assertRaises(ValueError):
            negative_field_map.bind_command(CoilCurrentCommand(0.1)).flux_density_tesla(
                np.asarray([[0.0, 0.0]])
            )

    def test_controller_has_no_density_or_transport_state_path(self) -> None:
        field_map = self._field_map()
        adapter = MagneticFieldControlAdapter(field_map=field_map)
        density = np.asarray([[0.2, 0.8], [0.3, 0.7]])
        density_before = density.copy()

        self.assertEqual(
            list(inspect.signature(MagneticFieldControlAdapter.adapt_command).parameters),
            ["self", "command"],
        )
        self.assertEqual(
            list(inspect.signature(MagneticFieldControlAdapter.physics_potential_for).parameters),
            ["self", "command"],
        )
        metadata = adapter.configuration_as_jsonable()
        self.assertFalse(metadata["contains_density"])
        self.assertFalse(metadata["contains_diffusion_or_mobility"])
        self.assertFalse(metadata["contains_pair_interaction"])
        adapter.physics_potential_for(CoilCurrentCommand(0.0)).potential_joule(
            np.asarray([[0.0, 0.0]])
        )
        np.testing.assert_array_equal(density, density_before)

        with self.assertRaises(TypeError):
            MagneticFieldControlAdapter(  # type: ignore[arg-type]
                field_map=TestOnlyUniformFieldPotential(1.0)
            )

    def test_contract_marks_non_magnetic_backends_as_historical(self) -> None:
        contract = (
            Path(__file__).resolve().parents[1] / "docs" / "CONTROL_INTERFACE.md"
        ).read_text(encoding="utf-8")
        self.assertIn("Physics-owned `LinearMagneticParticle`", contract)
        self.assertIn("cannot populate or replace", contract)
        self.assertIn("magnetic-field map, `V_mag` surrogate", contract)
        self.assertIn("no Gym, RL, DQN, task construction, or training", contract)


if __name__ == "__main__":
    unittest.main()
