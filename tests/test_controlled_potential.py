from __future__ import annotations

import unittest

import numpy as np

from mechanistic_mv.mechanics.controlled_potential import (
    TestOnlyUniformFieldPotential,
    ZeroControlledPotential,
)


class ControlledPotentialContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.positions_m = np.asarray(
            [
                [[1.0e-6, 2.0e-6], [3.0e-6, 5.0e-6]],
                [[7.0e-6, 11.0e-6], [13.0e-6, 17.0e-6]],
            ]
        )

    def test_uniform_backend_matches_analytic_potential_force_and_shapes(self) -> None:
        mutable_reference = [2.0e-6, -1.0e-6]
        backend = TestOnlyUniformFieldPotential(
            3.0e-14,
            mutable_reference,  # type: ignore[arg-type]
        )
        mutable_reference[0] = np.nan
        control = np.asarray([0.6, -0.8])

        force_vector = backend.maximum_force_newton * control
        expected_potential = -np.sum(
            (self.positions_m - np.asarray([2.0e-6, -1.0e-6])) * force_vector,
            axis=-1,
        )
        potential = backend.potential_joule(self.positions_m, control)
        force = backend.force_newton(self.positions_m, control)

        self.assertEqual(potential.shape, self.positions_m.shape[:-1])
        self.assertEqual(force.shape, self.positions_m.shape)
        np.testing.assert_allclose(potential, expected_potential, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            force,
            np.broadcast_to(force_vector, self.positions_m.shape),
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(backend.reference_position_m, (2.0e-6, -1.0e-6))
        self.assertEqual(backend.physical_status, "TEST_ONLY_NOT_FINAL_PHYSICS")

    def test_uniform_force_is_negative_spatial_gradient(self) -> None:
        backend = TestOnlyUniformFieldPotential(4.0e-14, (1.0e-6, 2.0e-6))
        point = np.asarray([3.0e-6, 5.0e-6])
        control = np.asarray([0.25, -0.5])
        step_m = 1.0e-10
        gradient = np.empty(2)

        for axis in range(2):
            offset = np.zeros(2)
            offset[axis] = step_m
            gradient[axis] = (
                backend.potential_joule(point + offset, control)
                - backend.potential_joule(point - offset, control)
            ) / (2.0 * step_m)

        np.testing.assert_allclose(
            backend.force_newton(point, control),
            -gradient,
            rtol=1.0e-11,
            atol=0.0,
        )

    def test_uniform_backend_zero_control_and_control_validation(self) -> None:
        backend = TestOnlyUniformFieldPotential(3.0e-14)
        expected_potential = np.zeros(self.positions_m.shape[:-1])
        expected_force = np.zeros_like(self.positions_m)

        for control in (None, np.zeros(2)):
            with self.subTest(control=control):
                np.testing.assert_array_equal(
                    backend.potential_joule(self.positions_m, control),
                    expected_potential,
                )
                np.testing.assert_array_equal(
                    backend.force_newton(self.positions_m, control), expected_force
                )

        backend.force_newton(self.positions_m, np.asarray([1.0, 0.0]))
        invalid_controls = (
            np.asarray([1.01, 0.0]),
            np.zeros(3),
            np.asarray([np.nan, 0.0]),
            np.asarray([np.inf, 0.0]),
        )
        for control in invalid_controls:
            with self.subTest(control=control):
                with self.assertRaises(ValueError):
                    backend.potential_joule(self.positions_m, control)
                with self.assertRaises(ValueError):
                    backend.force_newton(self.positions_m, control)

    def test_uniform_backend_rejects_invalid_configuration_and_positions(self) -> None:
        for maximum_force in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(maximum_force=maximum_force):
                with self.assertRaises(ValueError):
                    TestOnlyUniformFieldPotential(maximum_force)

        for reference in (
            (0.0,),
            (0.0, 0.0, 0.0),
            (np.nan, 0.0),
            (np.inf, 0.0),
        ):
            with self.subTest(reference=reference):
                with self.assertRaises(ValueError):
                    TestOnlyUniformFieldPotential(1.0, reference)  # type: ignore[arg-type]

        backend = TestOnlyUniformFieldPotential(1.0)
        invalid_positions = (
            np.asarray([0.0]),
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([np.nan, 0.0]),
            np.asarray([np.inf, 0.0]),
        )
        for positions in invalid_positions:
            with self.subTest(positions=positions):
                with self.assertRaises(ValueError):
                    backend.potential_joule(positions, np.zeros(2))
                with self.assertRaises(ValueError):
                    backend.force_newton(positions, np.zeros(2))

    def test_zero_backend_is_fail_closed_on_both_methods(self) -> None:
        backend = ZeroControlledPotential()
        for control in (None, np.zeros(2)):
            with self.subTest(control=control):
                np.testing.assert_array_equal(
                    backend.potential_joule(self.positions_m, control),
                    np.zeros(self.positions_m.shape[:-1]),
                )
                np.testing.assert_array_equal(
                    backend.force_newton(self.positions_m, control),
                    np.zeros_like(self.positions_m),
                )

        invalid_controls = (
            np.asarray([1.0, 0.0]),
            np.zeros(3),
            np.asarray([np.nan, 0.0]),
            np.asarray([np.inf, 0.0]),
        )
        for control in invalid_controls:
            with self.subTest(control=control):
                with self.assertRaises(ValueError):
                    backend.potential_joule(self.positions_m, control)
                with self.assertRaises(ValueError):
                    backend.force_newton(self.positions_m, control)

    def test_backend_identity_cannot_be_overridden_at_construction(self) -> None:
        with self.assertRaises(TypeError):
            ZeroControlledPotential(physical_status="pretend_final")  # type: ignore[call-arg]
        with self.assertRaises(TypeError):
            TestOnlyUniformFieldPotential(
                1.0,
                physical_status="pretend_final",  # type: ignore[call-arg]
            )
        with self.assertRaises(TypeError):
            TestOnlyUniformFieldPotential(
                1.0,
                name="pretend_actuator",  # type: ignore[call-arg]
            )


if __name__ == "__main__":
    unittest.main()
