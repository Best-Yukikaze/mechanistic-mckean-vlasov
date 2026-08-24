from __future__ import annotations

import unittest

import numpy as np

from mechanistic_mv.mechanics.geometry import (
    CartesianGrid,
    RectangleObstacle,
    RectangularDomain,
)
from mechanistic_mv.mechanics.pair_potential import ZeroPairPotential
from mechanistic_mv.mechanics.parameters import PhysicalParameters
from mechanistic_mv.particle_sim.empirical_density import empirical_density
from mechanistic_mv.particle_sim.langevin_interacting import overdamped_langevin_step


class ParticleSimulationTests(unittest.TestCase):
    def test_brownian_increment_mean_and_variance(self) -> None:
        parameters = PhysicalParameters()
        count = 40_000
        initial = np.full((count, 2), 0.5)
        dt_s = 2.0e-4
        advanced, diagnostics = overdamped_langevin_step(
            initial,
            parameters,
            ZeroPairPotential(),
            RectangularDomain((0.0, 1.0), (0.0, 1.0)),
            dt_s=dt_s,
            rng=np.random.default_rng(1234),
        )
        displacement = advanced - initial
        expected_component_variance = 2.0 * parameters.diffusion_m2_per_s * dt_s
        np.testing.assert_allclose(
            np.mean(displacement, axis=0),
            0.0,
            atol=0.02 * np.sqrt(expected_component_variance),
        )
        np.testing.assert_allclose(
            np.var(displacement, axis=0),
            expected_component_variance,
            rtol=0.025,
        )
        self.assertEqual(diagnostics.collision_count, 0)

    def test_empirical_density_has_exact_unit_mass(self) -> None:
        grid = CartesianGrid(RectangularDomain((0.0, 1.0), (0.0, 1.0)), 8, 6)
        positions = np.random.default_rng(8).uniform(0.0, 1.0, size=(500, 2))
        density = empirical_density(positions, grid)
        self.assertAlmostEqual(float(np.sum(density) * grid.cell_area_m2), 1.0)

    def test_brownian_particles_respect_obstacle_no_flux_over_many_steps(self) -> None:
        parameters = PhysicalParameters()
        domain = RectangularDomain((0.0, 10.0e-6), (0.0, 10.0e-6))
        obstacle = RectangleObstacle((4.0e-6, 6.0e-6), (3.0e-6, 7.0e-6))
        rng = np.random.default_rng(20260824)
        positions = np.column_stack(
            (
                rng.uniform(3.2e-6, 3.9e-6, size=300),
                rng.uniform(3.2e-6, 6.8e-6, size=300),
            )
        )
        total_collisions = 0
        for _ in range(12):
            positions, diagnostics = overdamped_langevin_step(
                positions,
                parameters,
                ZeroPairPotential(),
                domain,
                dt_s=0.25,
                rng=rng,
                obstacles=(obstacle,),
            )
            total_collisions += diagnostics.collision_count
            self.assertTrue(np.all(positions >= 0.0))
            self.assertTrue(np.all(positions <= 10.0e-6))
            self.assertFalse(np.any(obstacle.contains(positions)))
        self.assertGreater(total_collisions, 0)
        grid = CartesianGrid(domain, 20, 20)
        density = empirical_density(positions, grid)
        self.assertAlmostEqual(float(np.sum(density) * grid.cell_area_m2), 1.0)


if __name__ == "__main__":
    unittest.main()
