from __future__ import annotations

import unittest

import numpy as np

from mechanistic_mv.continuum.diagnostics import (
    density_moments,
    jensen_shannon_divergence,
)
from mechanistic_mv.continuum.initial_conditions import gaussian_density
from mechanistic_mv.continuum.mckean_vlasov import McKeanVlasovSolver
from mechanistic_mv.mechanics.geometry import CartesianGrid, RectangularDomain
from mechanistic_mv.mechanics.external_force import HarmonicTestPotential
from mechanistic_mv.mechanics.pair_potential import TestOnlyGaussianRepulsion
from mechanistic_mv.mechanics.parameters import PhysicalParameters
from mechanistic_mv.particle_sim.empirical_density import empirical_density
from mechanistic_mv.particle_sim.langevin_interacting import overdamped_langevin_step


class ParticleContinuumAgreementTests(unittest.TestCase):
    def test_interacting_particle_and_continuum_moments_agree(self) -> None:
        parameters = PhysicalParameters()
        domain = RectangularDomain((0.0, 20.0e-6), (0.0, 20.0e-6))
        grid = CartesianGrid(domain, 18, 18)
        potential = TestOnlyGaussianRepulsion(
            parameters.thermal_energy_joule, 1.2e-6
        )
        external = HarmonicTestPotential((10.0e-6, 10.0e-6), 6.0e-9)
        continuum_solver = McKeanVlasovSolver(
            grid, parameters, potential, external=external
        )
        density = gaussian_density(grid, (8.8e-6, 10.4e-6), 1.4e-6)
        rng = np.random.default_rng(20260824)
        particles = rng.normal(
            np.asarray([8.8e-6, 10.4e-6]), 1.4e-6, size=(700, 2)
        )
        dt = 0.01
        for _ in range(20):
            density, _ = continuum_solver.step(density, dt)
            particles, _ = overdamped_langevin_step(
                particles,
                parameters,
                potential,
                domain,
                dt_s=dt,
                rng=rng,
                pair_chunk_size=128,
                external=external,
            )
        particle_density = empirical_density(particles, grid)
        continuum_moments = density_moments(density, grid)
        particle_moments = density_moments(particle_density, grid)
        np.testing.assert_allclose(
            particle_moments.mean_m, continuum_moments.mean_m, atol=2.5e-7
        )
        np.testing.assert_allclose(
            np.diag(particle_moments.covariance_m2),
            np.diag(continuum_moments.covariance_m2),
            rtol=0.20,
        )
        self.assertLess(
            jensen_shannon_divergence(particle_density, density, grid), 0.16
        )


if __name__ == "__main__":
    unittest.main()
