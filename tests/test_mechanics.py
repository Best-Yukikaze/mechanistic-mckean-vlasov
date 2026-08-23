from __future__ import annotations

import unittest

import numpy as np

from mechanistic_mv.mechanics.energies import discrete_particle_energy_joule
from mechanistic_mv.mechanics.controlled_potential import TestOnlyUniformFieldPotential
from mechanistic_mv.mechanics.geometry import (
    RectangleObstacle,
    RectangularDomain,
    enforce_particle_no_flux,
)
from mechanistic_mv.mechanics.pair_potential import (
    TestOnlyGaussianRepulsion,
    mean_field_pair_force_newton,
)
from mechanistic_mv.mechanics.parameters import PhysicalParameters
from mechanistic_mv.mechanics.particle_model import (
    ParticleMechanicalState,
    deterministic_acceleration_m_per_s2,
    semi_implicit_euler_step,
)


class PhysicalParameterTests(unittest.TestCase):
    def test_einstein_relation_and_overdamped_ratio(self) -> None:
        parameters = PhysicalParameters()
        self.assertAlmostEqual(
            parameters.diffusion_m2_per_s,
            parameters.mobility_m_per_newton_second
            * parameters.thermal_energy_joule,
            places=30,
        )
        self.assertLess(parameters.overdamped_ratio(1.0e-3), 1.0e-3)


class PairPotentialTests(unittest.TestCase):
    def setUp(self) -> None:
        self.potential = TestOnlyGaussianRepulsion(4.0e-21, 1.2e-6)

    def test_force_is_negative_energy_gradient(self) -> None:
        displacement = np.asarray([0.7e-6, -0.4e-6])
        step = 1.0e-10
        gradient = np.empty(2)
        for axis in range(2):
            offset = np.zeros(2)
            offset[axis] = step
            gradient[axis] = (
                self.potential.potential_joule(displacement + offset)
                - self.potential.potential_joule(displacement - offset)
            ) / (2.0 * step)
        np.testing.assert_allclose(
            self.potential.force_newton(displacement), -gradient, rtol=2.0e-8
        )

    def test_pair_force_is_antisymmetric_and_chunk_independent(self) -> None:
        positions = np.asarray(
            [[1.0e-6, 2.0e-6], [2.0e-6, 2.5e-6], [4.0e-6, 3.0e-6]]
        )
        force_small = mean_field_pair_force_newton(
            positions, self.potential, chunk_size=1
        )
        force_large = mean_field_pair_force_newton(
            positions, self.potential, chunk_size=16
        )
        np.testing.assert_allclose(force_small, force_large, rtol=1.0e-14, atol=0.0)
        np.testing.assert_allclose(np.sum(force_small, axis=0), 0.0, atol=1.0e-29)

    def test_controlled_potential_force_is_negative_gradient(self) -> None:
        backend = TestOnlyUniformFieldPotential(3.0e-14, (2.0e-6, 1.0e-6))
        point = np.asarray([[4.0e-6, 5.0e-6]])
        control = np.asarray([0.6, -0.8])
        step = 1.0e-10
        gradient = np.empty(2)
        for axis in range(2):
            offset = np.zeros_like(point)
            offset[0, axis] = step
            gradient[axis] = (
                backend.potential_joule(point + offset, control)[0]
                - backend.potential_joule(point - offset, control)[0]
            ) / (2.0 * step)
        np.testing.assert_allclose(
            backend.force_newton(point, control)[0], -gradient, rtol=2.0e-12
        )


class MechanicalModelTests(unittest.TestCase):
    def test_second_order_mother_model_interface_and_energy(self) -> None:
        parameters = PhysicalParameters()
        potential = TestOnlyGaussianRepulsion(
            parameters.thermal_energy_joule, 1.0e-6
        )
        state = ParticleMechanicalState(
            np.asarray([[5.0e-6, 5.0e-6], [6.0e-6, 5.0e-6]]),
            np.zeros((2, 2)),
        )
        acceleration = deterministic_acceleration_m_per_s2(
            state, parameters, potential
        )
        self.assertEqual(acceleration.shape, (2, 2))
        self.assertGreater(acceleration[1, 0], 0.0)
        advanced = semi_implicit_euler_step(state, acceleration, 1.0e-9)
        self.assertTrue(np.all(np.isfinite(advanced.positions_m)))
        self.assertTrue(
            np.isfinite(discrete_particle_energy_joule(state, parameters, potential))
        )

    def test_swept_obstacle_contact_blocks_tunnelling(self) -> None:
        domain = RectangularDomain((0.0, 10.0), (0.0, 10.0))
        obstacle = RectangleObstacle((4.0, 6.0), (3.0, 7.0))
        previous = np.asarray([[2.0, 5.0]])
        proposed = np.asarray([[8.0, 5.0]])
        constrained, collisions = enforce_particle_no_flux(
            previous, proposed, domain, (obstacle,)
        )
        self.assertEqual(collisions, 1)
        self.assertLess(constrained[0, 0], 4.0)
        self.assertFalse(obstacle.contains(constrained)[0])


if __name__ == "__main__":
    unittest.main()
