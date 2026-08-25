from __future__ import annotations

import unittest

import numpy as np

from mechanistic_mv.continuum.diagnostics import (
    density_moments,
    jensen_shannon_divergence,
    relative_l2_error,
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


class DistributionDiagnosticValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = CartesianGrid(
            RectangularDomain((0.0, 4.0e-6), (0.0, 4.0e-6)), 4, 4
        )
        self.uniform = np.full(
            (self.grid.ny, self.grid.nx),
            1.0 / (self.grid.nx * self.grid.ny * self.grid.cell_area_m2),
        )

    def _assert_both_metrics_reject(
        self,
        first: np.ndarray,
        second: np.ndarray,
        message: str,
    ) -> None:
        metrics = (
            relative_l2_error,
            jensen_shannon_divergence,
        )
        for metric in metrics:
            with self.subTest(metric=metric.__name__):
                with self.assertRaisesRegex(ValueError, message):
                    metric(first, second, self.grid)

    def test_nonfinite_negative_and_zero_mass_densities_are_rejected(self) -> None:
        invalid_cases = (
            ("NaN", np.nan, "finite values"),
            ("infinity", np.inf, "finite values"),
            ("negative", -1.0, "non-negative"),
            ("zero mass", 0.0, "positive finite discrete mass"),
        )
        for name, invalid_value, message in invalid_cases:
            with self.subTest(case=name):
                invalid = (
                    np.zeros_like(self.uniform)
                    if name == "zero mass"
                    else self.uniform.copy()
                )
                if name != "zero mass":
                    invalid[0, 0] = invalid_value
                self._assert_both_metrics_reject(invalid, self.uniform, message)
                self._assert_both_metrics_reject(self.uniform, invalid, message)

    def test_shape_and_discrete_mass_mismatch_are_rejected(self) -> None:
        wrong_shape = np.ones(self.grid.nx * self.grid.ny, dtype=np.float64)
        self._assert_both_metrics_reject(
            wrong_shape, self.uniform, "(candidate|first) density must have grid shape"
        )
        unequal_mass = 1.01 * self.uniform
        self._assert_both_metrics_reject(
            unequal_mass, self.uniform, "discrete masses must agree"
        )

    def test_equal_mass_densities_keep_existing_metric_values(self) -> None:
        shifted = self.uniform.copy()
        shifted[0, 0] *= 2.0
        shifted[0, 1] = 0.0
        self.assertAlmostEqual(
            relative_l2_error(shifted, self.uniform, self.grid),
            np.sqrt(1.0 / 8.0),
        )
        expected_js = 0.5 * (
            (np.log(2.0 / 3.0) + np.log(2.0)) / 16.0
            + np.log(4.0 / 3.0) / 8.0
        )
        self.assertAlmostEqual(
            jensen_shannon_divergence(shifted, self.uniform, self.grid),
            expected_js,
        )


if __name__ == "__main__":
    unittest.main()
