from __future__ import annotations

import unittest

import numpy as np

from mechanistic_mv.continuum.convolution import (
    FFTPairConvolver,
    direct_pair_convolution_joule,
)
from mechanistic_mv.continuum.chemical_potential import chemical_potential_joule
from mechanistic_mv.continuum.free_energy import free_energy_components
from mechanistic_mv.continuum.diagnostics import density_moments, relative_l2_error
from mechanistic_mv.continuum.flux import (
    DriftFluxScheme,
    FaceFluxes,
    build_face_masks,
    compute_face_fluxes,
)
from mechanistic_mv.continuum.initial_conditions import gaussian_density
from mechanistic_mv.continuum.mckean_vlasov import (
    McKeanVlasovSolver,
    StepDiagnostics,
    conservative_update,
)
from mechanistic_mv.continuum.weak_form import weak_form_residual
from mechanistic_mv.mechanics.controlled_potential import TestOnlyUniformFieldPotential
from mechanistic_mv.mechanics.density_scaling import PairForceScaling
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

    def test_cached_spectrum_matches_direct_for_repeated_densities(self) -> None:
        grid = CartesianGrid(
            RectangularDomain((0.0, 9.0e-6), (0.0, 6.0e-6)), 9, 6
        )
        potential = TestOnlyGaussianRepulsion(3.0e-21, 0.9e-6)
        convolver = FFTPairConvolver(grid, potential)
        rng = np.random.default_rng(91)
        for _ in range(3):
            density = rng.uniform(0.1, 1.0, size=(grid.ny, grid.nx))
            density /= np.sum(density) * grid.cell_area_m2
            np.testing.assert_allclose(
                convolver.convolve_joule(density),
                direct_pair_convolution_joule(density, grid, potential),
                rtol=3.0e-13,
                atol=3.0e-34,
            )

    def test_nonfinite_pair_backend_is_rejected_by_both_convolutions(self) -> None:
        class NonfinitePairPotential:
            name = "invalid_nonfinite_pair"
            physical_status = "INVALID_TEST_BACKEND"
            pair_force_scaling = PairForceScaling.KAC_NORMALIZED_PROBABILITY
            scaling_population_count = None
            minimum_supported_distance_m = 0.0
            continuum_ready = True

            def potential_joule(self, displacement_m: np.ndarray) -> np.ndarray:
                return np.full(displacement_m.shape[:-1], np.nan)

            def force_newton(self, displacement_m: np.ndarray) -> np.ndarray:
                return np.zeros_like(displacement_m)

        grid = CartesianGrid(
            RectangularDomain((0.0, 4.0e-6), (0.0, 4.0e-6)), 4, 4
        )
        density = np.ones((4, 4)) / (16 * grid.cell_area_m2)
        backend = NonfinitePairPotential()
        with self.assertRaisesRegex(FloatingPointError, "became non-finite"):
            direct_pair_convolution_joule(density, grid, backend)
        with self.assertRaisesRegex(ValueError, "invalid convolution kernel"):
            FFTPairConvolver(grid, backend)

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

    def test_legacy_face_flux_and_step_diagnostics_constructors_remain_valid(self) -> None:
        grid = CartesianGrid(self.domain, 8, 8)
        legacy_fluxes = FaceFluxes(
            np.zeros((grid.ny, grid.nx + 1)),
            np.zeros((grid.ny + 1, grid.nx)),
            2.0e-6,
            3.0e-6,
        )
        self.assertIsNone(legacy_fluxes.outgoing_rate_per_s)
        self.assertEqual(
            legacy_fluxes.drift_flux_scheme, DriftFluxScheme.FIRST_ORDER_UPWIND
        )
        solver = McKeanVlasovSolver(
            grid, self.parameters, ZeroPairPotential(), cfl_safety=0.8
        )
        expected_rate = (
            2.0 * self.parameters.diffusion_m2_per_s
            * (1.0 / grid.dx_m**2 + 1.0 / grid.dy_m**2)
            + 2.0 * 2.0e-6 / grid.dx_m
            + 2.0 * 3.0e-6 / grid.dy_m
        )
        self.assertAlmostEqual(solver.stable_dt_s(legacy_fluxes), 0.8 / expected_rate)
        obsolete_one_sided_rate = (
            2.0 * self.parameters.diffusion_m2_per_s
            * (1.0 / grid.dx_m**2 + 1.0 / grid.dy_m**2)
            + 2.0e-6 / grid.dx_m
            + 3.0e-6 / grid.dy_m
        )
        self.assertLess(
            solver.stable_dt_s(legacy_fluxes), 0.8 / obsolete_one_sided_rate
        )
        for invalid_rate in (
            np.ones((grid.ny - 1, grid.nx)),
            np.full((grid.ny, grid.nx), np.nan),
            -np.ones((grid.ny, grid.nx)),
        ):
            malformed_fluxes = FaceFluxes(
                legacy_fluxes.x_per_m_s,
                legacy_fluxes.y_per_m_s,
                legacy_fluxes.max_abs_velocity_x_m_per_s,
                legacy_fluxes.max_abs_velocity_y_m_per_s,
                invalid_rate,
            )
            with self.assertRaisesRegex(ValueError, "outgoing_rate_per_s"):
                solver.stable_dt_s(malformed_fluxes)

        legacy_diagnostics = StepDiagnostics(
            substeps=1,
            initial_mass=1.0,
            final_mass=1.0,
            minimum_stable_dt_s=0.1,
            minimum_density_per_m2=0.0,
            clipped_negative_mass=0.0,
            maximum_abs_flux_per_m_s=0.0,
        )
        self.assertEqual(
            legacy_diagnostics.drift_flux_scheme,
            DriftFluxScheme.FIRST_ORDER_UPWIND,
        )
        self.assertIsNone(legacy_diagnostics.fixed_control_free_energy_change_joule)

    def test_pure_diffusion_preserves_mass_positivity_and_symmetry(self) -> None:
        grid = CartesianGrid(self.domain, 40, 40)
        solver = McKeanVlasovSolver(grid, self.parameters, ZeroPairPotential())
        initial = gaussian_density(grid, (10.0e-6, 10.0e-6), 1.0e-6)
        initial_moments = density_moments(initial, grid)
        advanced, diagnostics = solver.step(initial, 0.5)
        final_moments = density_moments(advanced, grid)
        self.assertLess(diagnostics.absolute_mass_error, 2.0e-14)
        self.assertGreaterEqual(float(np.min(advanced)), 0.0)
        self.assertEqual(
            diagnostics.drift_flux_scheme, DriftFluxScheme.FIRST_ORDER_UPWIND
        )
        self.assertIsNone(diagnostics.fixed_control_free_energy_change_joule)
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

    def test_scharfetter_gummel_preserves_discrete_gibbs_equilibrium(self) -> None:
        domain = RectangularDomain((0.0, 8.0e-6), (0.0, 4.0e-6))
        grid = CartesianGrid(domain, 8, 4)
        diffusion = 2.0e-12
        mobility = 1.0e8
        q_per_x_face = 0.2
        x_index = np.arange(grid.nx, dtype=np.float64)
        potential_row = q_per_x_face * diffusion / mobility * x_index
        potential = np.broadcast_to(potential_row, (grid.ny, grid.nx)).copy()
        density_row = 1.0e12 * np.exp(-q_per_x_face * x_index)
        density = np.broadcast_to(density_row, (grid.ny, grid.nx)).copy()
        masks = build_face_masks(np.ones((grid.ny, grid.nx), dtype=bool))
        fluxes = compute_face_fluxes(
            density,
            potential,
            grid,
            diffusion_m2_per_s=diffusion,
            mobility_m_per_newton_second=mobility,
            face_masks=masks,
            drift_flux_scheme=DriftFluxScheme.SECOND_ORDER_SCHARFETTER_GUMMEL,
        )
        np.testing.assert_allclose(fluxes.x_per_m_s[:, 1:-1], 0.0, atol=1.0e-8)
        np.testing.assert_allclose(fluxes.y_per_m_s[1:-1, :], 0.0, atol=0.0)

    def test_scharfetter_gummel_is_finite_at_small_and_large_potential_jumps(self) -> None:
        grid = CartesianGrid(self.domain, 4, 4)
        diffusion = 1.0e-12
        mobility = 1.0e8
        potential_row = np.array((0.0, 1.0e-32, 1.0e-17, 0.0))
        potential = np.broadcast_to(potential_row, (grid.ny, grid.nx)).copy()
        masks = build_face_masks(np.ones((grid.ny, grid.nx), dtype=bool))
        fluxes = compute_face_fluxes(
            np.ones((grid.ny, grid.nx), dtype=np.float64),
            potential,
            grid,
            diffusion_m2_per_s=diffusion,
            mobility_m_per_newton_second=mobility,
            face_masks=masks,
            drift_flux_scheme=DriftFluxScheme.SECOND_ORDER_SCHARFETTER_GUMMEL,
        )
        self.assertTrue(np.all(np.isfinite(fluxes.x_per_m_s)))
        self.assertTrue(np.all(np.isfinite(fluxes.y_per_m_s)))
        self.assertTrue(np.all(np.isfinite(fluxes.outgoing_rate_per_s)))
        with self.assertRaisesRegex(ValueError, "requires diffusion"):
            compute_face_fluxes(
                np.ones((grid.ny, grid.nx), dtype=np.float64),
                potential,
                grid,
                diffusion_m2_per_s=0.0,
                mobility_m_per_newton_second=mobility,
                face_masks=masks,
                drift_flux_scheme=DriftFluxScheme.SECOND_ORDER_SCHARFETTER_GUMMEL,
            )

    def test_outgoing_rate_counts_barrier_outflow_on_every_open_face(self) -> None:
        domain = RectangularDomain((0.0, 4.0), (0.0, 4.0))
        grid = CartesianGrid(domain, 4, 4)
        density = np.ones((4, 4), dtype=np.float64)
        potential = np.zeros((4, 4), dtype=np.float64)
        potential[1, 1] = 1.0
        masks = build_face_masks(np.ones((4, 4), dtype=bool))
        fluxes = compute_face_fluxes(
            density,
            potential,
            grid,
            diffusion_m2_per_s=0.0,
            mobility_m_per_newton_second=1.0,
            face_masks=masks,
        )
        self.assertEqual(float(fluxes.outgoing_rate_per_s[1, 1]), 4.0)
        self.assertGreater(
            fluxes.maximum_outgoing_rate_per_s,
            fluxes.max_abs_velocity_x_m_per_s / grid.dx_m
            + fluxes.max_abs_velocity_y_m_per_s / grid.dy_m,
        )
        solver = McKeanVlasovSolver(
            grid, self.parameters, ZeroPairPotential(), cfl_safety=0.9
        )
        self.assertAlmostEqual(solver.stable_dt_s(fluxes), 0.9 / 4.0)

    def test_second_order_scharfetter_gummel_step_is_nonnegative_conservative_and_no_flux(self) -> None:
        grid = CartesianGrid(self.domain, 32, 32)
        obstacle = RectangleObstacle((8.0e-6, 12.0e-6), (7.5e-6, 12.5e-6))
        external = HarmonicTestPotential((11.0e-6, 10.0e-6), 1.0e-8)
        solver = McKeanVlasovSolver(
            grid,
            self.parameters,
            ZeroPairPotential(),
            obstacles=(obstacle,),
            external=external,
            drift_flux_scheme=DriftFluxScheme.SECOND_ORDER_SCHARFETTER_GUMMEL,
            record_fixed_control_free_energy=True,
        )
        initial = gaussian_density(
            grid, (5.0e-6, 10.0e-6), 1.2e-6, fluid_mask=solver.fluid_mask
        )
        initial_energy = solver.free_energy(initial).total_joule
        with self.assertRaisesRegex(ValueError, "requires dt_s"):
            solver.face_fluxes(initial)
        explicit_dt_s = 1.0e-2
        with self.assertRaisesRegex(ValueError, "exceeds the explicit"):
            solver.face_fluxes(initial, dt_s=1.0)
        raw_fluxes, _ = solver.face_fluxes(initial, dt_s=explicit_dt_s)
        self.assertEqual(
            raw_fluxes.drift_flux_scheme,
            DriftFluxScheme.SECOND_ORDER_SCHARFETTER_GUMMEL,
        )
        self.assertTrue(np.all(raw_fluxes.x_per_m_s[~solver.face_masks.open_x] == 0.0))
        self.assertTrue(np.all(raw_fluxes.y_per_m_s[~solver.face_masks.open_y] == 0.0))
        self.assertTrue(
            np.all(raw_fluxes.outgoing_rate_per_s[~solver.fluid_mask] == 0.0)
        )
        one_update = conservative_update(initial, raw_fluxes, grid, explicit_dt_s)
        self.assertGreaterEqual(float(np.min(one_update)), 0.0)
        self.assertAlmostEqual(
            float(np.sum(one_update) * grid.cell_area_m2),
            float(np.sum(initial) * grid.cell_area_m2),
            places=14,
        )

        advanced, diagnostics = solver.step(initial, 0.5)
        self.assertLess(diagnostics.absolute_mass_error, 2.0e-14)
        self.assertGreaterEqual(float(np.min(advanced)), 0.0)
        self.assertEqual(diagnostics.clipped_negative_mass, 0.0)
        self.assertEqual(
            diagnostics.drift_flux_scheme,
            DriftFluxScheme.SECOND_ORDER_SCHARFETTER_GUMMEL,
        )
        self.assertIsNotNone(diagnostics.fixed_control_free_energy_change_joule)
        self.assertAlmostEqual(
            diagnostics.fixed_control_free_energy_change_joule,
            solver.free_energy(advanced).total_joule - initial_energy,
            places=30,
        )

    def test_pure_diffusion_grid_refinement_reduces_density_error(self) -> None:
        errors = []
        final_time_s = 0.2
        initial_variance_m2 = 1.0e-12
        for size in (24, 48, 96):
            grid = CartesianGrid(self.domain, size, size)
            solver = McKeanVlasovSolver(
                grid, self.parameters, ZeroPairPotential()
            )
            initial = gaussian_density(
                grid, (10.0e-6, 10.0e-6), np.sqrt(initial_variance_m2)
            )
            advanced, _ = solver.step(initial, final_time_s)
            x, y = grid.mesh_m()
            analytic_variance = (
                initial_variance_m2
                + 2.0 * self.parameters.diffusion_m2_per_s * final_time_s
            )
            analytic = np.exp(
                -(
                    (x - 10.0e-6) ** 2 + (y - 10.0e-6) ** 2
                )
                / (2.0 * analytic_variance)
            )
            analytic /= np.sum(analytic) * grid.cell_area_m2
            errors.append(relative_l2_error(advanced, analytic, grid))
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])
        self.assertLess(errors[2], 1.0e-3)
        self.assertGreater(np.log2(errors[1] / errors[2]), 1.5)

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

    def test_cfl_refinement_converges_for_complete_mv(self) -> None:
        grid = CartesianGrid(self.domain, 32, 32)
        external = HarmonicTestPotential((10.0e-6, 10.0e-6), 8.0e-9)
        interaction = TestOnlyGaussianRepulsion(
            1.5 * self.parameters.thermal_energy_joule, 1.1e-6
        )
        initial = gaussian_density(
            grid, (8.0e-6, 11.0e-6), (1.0e-6, 1.4e-6)
        )
        solutions = []
        for safety in (0.9, 0.45, 0.225, 0.1125):
            solver = McKeanVlasovSolver(
                grid,
                self.parameters,
                interaction,
                external=external,
                cfl_safety=safety,
            )
            solution, _ = solver.step(initial, 0.5)
            solutions.append(solution)
        errors = [
            relative_l2_error(solution, solutions[-1], grid)
            for solution in solutions[:-1]
        ]
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])
        self.assertLess(errors[2], 2.0e-3)

    def test_controlled_potential_is_evaluated_once_per_requested_step(self) -> None:
        class CountingPotential:
            name = "counting_test_potential"
            physical_status = "TEST_ONLY_NOT_FINAL_PHYSICS"

            def __init__(self) -> None:
                self.calls = 0

            def potential_joule(
                self, positions_m: np.ndarray, control: np.ndarray | None
            ) -> np.ndarray:
                self.calls += 1
                return np.zeros(positions_m.shape[0])

            def force_newton(
                self, positions_m: np.ndarray, control: np.ndarray | None
            ) -> np.ndarray:
                return np.zeros_like(positions_m)

        grid = CartesianGrid(self.domain, 48, 48)
        backend = CountingPotential()
        solver = McKeanVlasovSolver(
            grid,
            self.parameters,
            ZeroPairPotential(),
            controlled_potential=backend,
        )
        density = gaussian_density(grid, (10.0e-6, 10.0e-6), 1.0e-6)
        _, diagnostics = solver.step(density, 0.5, control=np.zeros(2))
        self.assertGreater(diagnostics.substeps, 1)
        self.assertEqual(backend.calls, 1)

    def test_invalid_external_potential_is_rejected_at_construction(self) -> None:
        class NonfiniteExternalPotential:
            name = "invalid_nonfinite_external"

            def potential_joule(self, positions_m: np.ndarray) -> np.ndarray:
                return np.full(positions_m.shape[:-1], np.nan)

            def force_newton(self, positions_m: np.ndarray) -> np.ndarray:
                return np.zeros_like(positions_m)

        grid = CartesianGrid(self.domain, 8, 8)
        with self.assertRaisesRegex(ValueError, "one finite value"):
            McKeanVlasovSolver(
                grid,
                self.parameters,
                ZeroPairPotential(),
                external=NonfiniteExternalPotential(),
            )

    def test_invalid_controlled_potential_output_is_rejected(self) -> None:
        class InvalidControlledPotential:
            name = "invalid_controlled_shape"
            physical_status = "INVALID_TEST_BACKEND"

            def potential_joule(
                self, positions_m: np.ndarray, control: np.ndarray | None
            ) -> np.ndarray:
                return np.zeros((positions_m.shape[0], 1))

            def force_newton(
                self, positions_m: np.ndarray, control: np.ndarray | None
            ) -> np.ndarray:
                return np.zeros_like(positions_m)

        grid = CartesianGrid(self.domain, 8, 8)
        solver = McKeanVlasovSolver(
            grid,
            self.parameters,
            ZeroPairPotential(),
            controlled_potential=InvalidControlledPotential(),
        )
        density = gaussian_density(grid, (10.0e-6, 10.0e-6), 1.0e-6)
        with self.assertRaisesRegex(ValueError, "one finite value"):
            solver.step(density, 0.01, control=np.zeros(2))

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
