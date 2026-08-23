from __future__ import annotations

import unittest

import numpy as np

from mechanistic_mv.continuum.convolution import (
    FFTPairConvolver,
    direct_pair_convolution_joule,
)
from mechanistic_mv.continuum.chemical_potential import chemical_potential_joule
from mechanistic_mv.continuum.free_energy import free_energy_components
from mechanistic_mv.continuum.diagnostics import density_moments
from mechanistic_mv.continuum.initial_conditions import gaussian_density
from mechanistic_mv.continuum.mckean_vlasov import (
    McKeanVlasovSolver,
    conservative_update,
)
from mechanistic_mv.continuum.weak_form import weak_form_residual
from mechanistic_mv.mechanics.controlled_potential import TestOnlyUniformFieldPotential
from mechanistic_mv.mechanics.geometry import (
    CartesianGrid,
    RectangleObstacle,
    RectangularDomain,
)
from mechanistic_mv.mechanics.external_force import HarmonicTestPotential
from mechanistic_mv.mechanics.pair_potential import (
    TestOnlyGaussianRepulsion,
    ZeroPairPotential,
)
from mechanistic_mv.mechanics.parameters import PhysicalParameters


class ConvolutionTests(unittest.TestCase):
    def test_fft_matches_definition_including_cell_area(self) -> None:
        grid = CartesianGrid(RectangularDomain((0.0, 7.0e-6), (0.0, 5.0e-6)), 7, 5)
        potential = TestOnlyGaussianRepulsion(5.0e-21, 1.1e-6)
        density = np.random.default_rng(9).uniform(size=(grid.ny, grid.nx))
        density /= np.sum(density) * grid.cell_area_m2
        reference = direct_pair_convolution_joule(density, grid, potential)
        accelerated = FFTPairConvolver(grid, potential).convolve_joule(density)
        np.testing.assert_allclose(accelerated, reference, rtol=2.0e-13, atol=1.0e-34)

    def test_chemical_potential_matches_free_energy_first_variation(self) -> None:
        grid = CartesianGrid(RectangularDomain((0.0, 6.0e-6), (0.0, 5.0e-6)), 6, 5)
        parameters = PhysicalParameters()
        potential = TestOnlyGaussianRepulsion(
            parameters.thermal_energy_joule, 1.0e-6
        )
        convolver = FFTPairConvolver(grid, potential)
        rng = np.random.default_rng(12)
        density = rng.uniform(0.5, 1.5, size=(grid.ny, grid.nx))
        density /= np.sum(density) * grid.cell_area_m2
        direction = density * rng.uniform(-0.2, 0.2, size=density.shape)
        external = np.zeros_like(density)
        reference_density = 1.0 / (
            grid.nx * grid.ny * grid.cell_area_m2
        )
        interaction = convolver.convolve_joule(density)
        chemical = chemical_potential_joule(
            density,
            parameters.thermal_energy_joule,
            reference_density,
            external,
            interaction,
        )

        def energy(values: np.ndarray) -> float:
            return free_energy_components(
                values,
                grid,
                parameters.thermal_energy_joule,
                reference_density,
                external,
                convolver.convolve_joule(values),
            ).total_joule

        epsilon = 1.0e-5
        numerical = (energy(density + epsilon * direction) - energy(density - epsilon * direction)) / (
            2.0 * epsilon
        )
        variational = float(np.sum(chemical * direction) * grid.cell_area_m2)
        self.assertAlmostEqual(numerical, variational, delta=5.0e-9 * abs(variational))


class FiniteVolumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = PhysicalParameters()
        self.domain = RectangularDomain((0.0, 20.0e-6), (0.0, 20.0e-6))

    def test_pure_diffusion_preserves_mass_positivity_and_symmetry(self) -> None:
        grid = CartesianGrid(self.domain, 40, 40)
        solver = McKeanVlasovSolver(grid, self.parameters, ZeroPairPotential())
        initial = gaussian_density(grid, (10.0e-6, 10.0e-6), 1.0e-6)
        initial_moments = density_moments(initial, grid)
        advanced, diagnostics = solver.step(initial, 0.5)
        final_moments = density_moments(advanced, grid)
        self.assertLess(diagnostics.absolute_mass_error, 2.0e-14)
        self.assertGreaterEqual(float(np.min(advanced)), 0.0)
        np.testing.assert_allclose(
            final_moments.mean_m, initial_moments.mean_m, atol=2.0e-12
        )
        expected_variance = (
            initial_moments.covariance_m2[0, 0]
            + 2.0 * self.parameters.diffusion_m2_per_s * 0.5
        )
        self.assertAlmostEqual(
            final_moments.covariance_m2[0, 0], expected_variance, delta=5.0e-14
        )
        self.assertAlmostEqual(
            final_moments.covariance_m2[0, 0],
            final_moments.covariance_m2[1, 1],
            delta=2.0e-18,
        )

    def test_discrete_weak_form_residual_is_roundoff(self) -> None:
        grid = CartesianGrid(self.domain, 20, 18)
        solver = McKeanVlasovSolver(grid, self.parameters, ZeroPairPotential())
        old = gaussian_density(grid, (8.0e-6, 11.0e-6), (1.5e-6, 2.0e-6))
        fluxes, _ = solver.face_fluxes(old)
        dt = 0.25 * solver.stable_dt_s(fluxes)
        new = conservative_update(old, fluxes, grid, dt)
        x, y = grid.mesh_m()
        eta = 1.0 + x / self.domain.x_limits_m[1] + (y / self.domain.y_limits_m[1]) ** 2
        residual = weak_form_residual(old, new, eta, fluxes, grid, dt)
        scale = max(1.0, abs(np.sum(eta * (new - old)) * grid.cell_area_m2 / dt))
        self.assertLess(abs(residual), 2.0e-13 * scale)

    def test_obstacle_is_zero_density_and_zero_flux(self) -> None:
        grid = CartesianGrid(self.domain, 32, 32)
        obstacle = RectangleObstacle((8.0e-6, 12.0e-6), (7.5e-6, 12.5e-6))
        solver = McKeanVlasovSolver(
            grid, self.parameters, ZeroPairPotential(), obstacles=(obstacle,)
        )
        initial = gaussian_density(
            grid, (5.0e-6, 10.0e-6), 1.2e-6, fluid_mask=solver.fluid_mask
        )
        fluxes, _ = solver.face_fluxes(initial)
        self.assertTrue(np.all(fluxes.x_per_m_s[~solver.face_masks.open_x] == 0.0))
        self.assertTrue(np.all(fluxes.y_per_m_s[~solver.face_masks.open_y] == 0.0))
        advanced, diagnostics = solver.step(initial, 0.5)
        self.assertTrue(np.all(advanced[~solver.fluid_mask] == 0.0))
        self.assertLess(diagnostics.absolute_mass_error, 2.0e-14)

    def test_passive_free_energy_decreases(self) -> None:
        grid = CartesianGrid(self.domain, 32, 32)
        potential = TestOnlyGaussianRepulsion(
            2.0 * self.parameters.thermal_energy_joule, 1.2e-6
        )
        solver = McKeanVlasovSolver(grid, self.parameters, potential)
        density = gaussian_density(grid, (10.0e-6, 10.0e-6), 1.0e-6)
        energies = [solver.free_energy(density).total_joule]
        for _ in range(8):
            density, _ = solver.step(density, 0.05)
            energies.append(solver.free_energy(density).total_joule)
        differences = np.diff(energies)
        self.assertLessEqual(float(np.max(differences)), 2.0e-12 * abs(energies[0]))
        self.assertLess(energies[-1], energies[0])

    def test_external_only_moves_toward_trap_and_dissipates_energy(self) -> None:
        grid = CartesianGrid(self.domain, 32, 32)
        external = HarmonicTestPotential((11.0e-6, 10.0e-6), 1.0e-8)
        solver = McKeanVlasovSolver(
            grid, self.parameters, ZeroPairPotential(), external=external
        )
        density = gaussian_density(grid, (7.0e-6, 10.0e-6), 1.0e-6)
        initial_mean = density_moments(density, grid).mean_m[0]
        initial_energy = solver.free_energy(density).total_joule
        density, diagnostics = solver.step(density, 0.2)
        self.assertGreater(density_moments(density, grid).mean_m[0], initial_mean)
        self.assertLess(solver.free_energy(density).total_joule, initial_energy)
        self.assertGreater(diagnostics.maximum_abs_flux_per_m_s, 0.0)

    def test_complete_mv_with_external_and_pair_terms(self) -> None:
        grid = CartesianGrid(self.domain, 32, 32)
        external = HarmonicTestPotential((10.0e-6, 10.0e-6), 8.0e-9)
        interaction = TestOnlyGaussianRepulsion(
            1.5 * self.parameters.thermal_energy_joule, 1.1e-6
        )
        solver = McKeanVlasovSolver(
            grid, self.parameters, interaction, external=external
        )
        density = gaussian_density(grid, (8.0e-6, 11.0e-6), (1.0e-6, 1.4e-6))
        initial_energy = solver.free_energy(density).total_joule
        density, diagnostics = solver.step(density, 0.25)
        self.assertLess(diagnostics.absolute_mass_error, 2.0e-14)
        self.assertGreaterEqual(diagnostics.minimum_density_per_m2, 0.0)
        self.assertEqual(diagnostics.clipped_negative_mass, 0.0)
        self.assertLess(solver.free_energy(density).total_joule, initial_energy)

    def test_test_control_moves_density_in_force_direction(self) -> None:
        grid = CartesianGrid(self.domain, 32, 32)
        control = TestOnlyUniformFieldPotential(4.0e-14)
        solver = McKeanVlasovSolver(
            grid,
            self.parameters,
            ZeroPairPotential(),
            controlled_potential=control,
        )
        initial = gaussian_density(grid, (8.0e-6, 10.0e-6), 1.0e-6)
        advanced, _ = solver.step(initial, 0.05, control=np.asarray([1.0, 0.0]))
        shift = density_moments(advanced, grid).mean_m - density_moments(initial, grid).mean_m
        self.assertGreater(shift[0], 0.0)
        self.assertLess(abs(shift[1]), 1.0e-14)


if __name__ == "__main__":
    unittest.main()
