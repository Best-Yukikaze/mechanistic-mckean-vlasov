from __future__ import annotations

import unittest

import numpy as np

from mechanistic_mv.mechanics.energies import discrete_particle_energy_joule
from mechanistic_mv.mechanics.controlled_potential import (
    TestOnlyUniformFieldPotential,
    ZeroControlledPotential,
)
from mechanistic_mv.mechanics.external_force import HarmonicTestPotential
from mechanistic_mv.mechanics.geometry import (
    RectangleObstacle,
    RectangularDomain,
    enforce_particle_no_flux,
    reflect_outer_walls,
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

    def test_discrete_total_energy_gradient_matches_total_force(self) -> None:
        parameters = PhysicalParameters()
        pair = TestOnlyGaussianRepulsion(
            parameters.thermal_energy_joule, 1.1e-6
        )
        external = HarmonicTestPotential((5.0e-6, 5.0e-6), 2.0e-9)
        controlled = TestOnlyUniformFieldPotential(3.0e-15)
        control = np.asarray([0.3, -0.2])
        positions = np.asarray(
            [[4.0e-6, 4.5e-6], [5.2e-6, 4.8e-6], [6.0e-6, 5.7e-6]]
        )
        state = ParticleMechanicalState(positions, np.zeros_like(positions))
        force = parameters.particle_mass_kg * deterministic_acceleration_m_per_s2(
            state,
            parameters,
            pair,
            external=external,
            controlled_potential=controlled,
            control=control,
        )
        step = 1.0e-10
        energy_gradient = np.empty_like(positions)
        for particle in range(positions.shape[0]):
            for axis in range(2):
                offset = np.zeros_like(positions)
                offset[particle, axis] = step
                plus = ParticleMechanicalState(
                    positions + offset, np.zeros_like(positions)
                )
                minus = ParticleMechanicalState(
                    positions - offset, np.zeros_like(positions)
                )
                energy_gradient[particle, axis] = (
                    discrete_particle_energy_joule(
                        plus,
                        parameters,
                        pair,
                        external=external,
                        controlled_potential=controlled,
                        control=control,
                    )
                    - discrete_particle_energy_joule(
                        minus,
                        parameters,
                        pair,
                        external=external,
                        controlled_potential=controlled,
                        control=control,
                    )
                ) / (2.0 * step)
        np.testing.assert_allclose(force, -energy_gradient, rtol=2.0e-8, atol=1.0e-24)

    def test_null_control_backend_rejects_nonzero_command(self) -> None:
        backend = ZeroControlledPotential()
        points = np.asarray([[1.0e-6, 2.0e-6]])
        np.testing.assert_array_equal(
            backend.force_newton(points, np.zeros(2)), np.zeros_like(points)
        )
        with self.assertRaisesRegex(ValueError, "physical potential backend"):
            backend.potential_joule(points, np.asarray([1.0, 0.0]))


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

    def test_oblique_obstacle_reflection_preserves_tangential_motion(self) -> None:
        domain = RectangularDomain((0.0, 10.0), (0.0, 10.0))
        obstacle = RectangleObstacle((4.0, 6.0), (3.0, 7.0))
        constrained, collisions = enforce_particle_no_flux(
            np.asarray([[2.0, 4.0]]),
            np.asarray([[7.0, 5.5]]),
            domain,
            (obstacle,),
        )
        np.testing.assert_allclose(constrained, np.asarray([[1.0, 5.5]]), atol=1e-12)
        self.assertEqual(collisions, 1)

    def test_one_step_resolves_obstacle_then_outer_wall_reflection(self) -> None:
        domain = RectangularDomain((0.0, 10.0), (0.0, 10.0))
        obstacle = RectangleObstacle((4.0, 6.0), (3.0, 7.0))
        constrained, collisions = enforce_particle_no_flux(
            np.asarray([[2.0, 5.0]]),
            np.asarray([[11.0, 5.0]]),
            domain,
            (obstacle,),
        )
        np.testing.assert_allclose(constrained, np.asarray([[3.0, 5.0]]), atol=1e-12)
        self.assertEqual(collisions, 2)

    def test_large_random_displacements_never_penetrate_solid_or_domain(self) -> None:
        domain = RectangularDomain((0.0, 10.0), (0.0, 10.0))
        obstacle = RectangleObstacle((4.0, 6.0), (3.0, 7.0))
        rng = np.random.default_rng(20260824)
        previous = rng.uniform(0.5, 9.5, size=(200, 2))
        previous = previous[~obstacle.contains(previous)]
        proposed = previous + rng.normal(0.0, 14.0, size=previous.shape)
        constrained, collisions = enforce_particle_no_flux(
            previous, proposed, domain, (obstacle,)
        )
        self.assertGreater(collisions, 0)
        self.assertTrue(np.all(constrained >= 0.0))
        self.assertTrue(np.all(constrained <= 10.0))
        self.assertFalse(np.any(obstacle.contains(constrained)))

    def test_outer_corner_counts_both_reflected_faces(self) -> None:
        domain = RectangularDomain((0.0, 10.0), (0.0, 10.0))
        constrained, collisions = enforce_particle_no_flux(
            np.asarray([[5.0, 5.0]]),
            np.asarray([[12.0, 12.0]]),
            domain,
        )
        np.testing.assert_allclose(constrained, np.asarray([[8.0, 8.0]]), atol=1e-12)
        self.assertEqual(collisions, 2)

    def test_obstacle_generator_applies_to_every_particle(self) -> None:
        domain = RectangularDomain((0.0, 10.0), (0.0, 10.0))
        obstacle = RectangleObstacle((4.0, 6.0), (3.0, 7.0))
        previous = np.asarray([[2.0, 4.0], [2.0, 6.0]])
        proposed = np.asarray([[8.0, 4.0], [8.0, 6.0]])
        constrained, collisions = enforce_particle_no_flux(
            previous, proposed, domain, (item for item in (obstacle,))
        )
        self.assertEqual(collisions, 2)
        self.assertTrue(np.all(constrained[:, 0] < 4.0))
        with self.assertRaisesRegex(ValueError, "starts inside"):
            enforce_particle_no_flux(
                np.asarray([[5.0, 5.0]]),
                np.asarray([[7.0, 5.0]]),
                domain,
                (obstacle,),
            )
        with self.assertRaisesRegex(ValueError, "outside the fluid domain"):
            enforce_particle_no_flux(
                np.asarray([[-1.0, 5.0]]),
                np.asarray([[1.0, 5.0]]),
                domain,
            )

    def test_outer_wall_reflection_handles_corner_overshoot(self) -> None:
        domain = RectangularDomain((0.0, 10.0), (0.0, 10.0))
        reflected = reflect_outer_walls(np.asarray([[-1.0, 12.0]]), domain)
        np.testing.assert_allclose(reflected, np.asarray([[1.0, 8.0]]))

    def test_no_obstacle_in_domain_fast_path_returns_independent_copy(self) -> None:
        domain = RectangularDomain((0.0, 10.0), (0.0, 10.0))
        previous = np.asarray([[2.0, 3.0], [7.0, 8.0]])
        proposed = np.asarray([[2.5, 3.5], [6.5, 7.5]])
        constrained, collisions = enforce_particle_no_flux(
            previous, proposed, domain
        )
        np.testing.assert_array_equal(constrained, proposed)
        self.assertFalse(np.shares_memory(constrained, proposed))
        self.assertEqual(collisions, 0)


if __name__ == "__main__":
    unittest.main()
