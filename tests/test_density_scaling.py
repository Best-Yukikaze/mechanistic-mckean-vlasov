from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import unittest

import numpy as np

from mechanistic_mv.continuum.convolution import (
    FFTPairConvolver,
    direct_pair_convolution_joule,
)
from mechanistic_mv.continuum.initial_conditions import gaussian_density
from mechanistic_mv.continuum.mckean_vlasov import McKeanVlasovSolver
from mechanistic_mv.mechanics.density_scaling import (
    DensityConvention,
    PairForceScaling,
)
from mechanistic_mv.mechanics.energies import discrete_particle_energy_joule
from mechanistic_mv.mechanics.geometry import (
    CartesianGrid,
    RectangularDomain,
)
from mechanistic_mv.mechanics.hydrogel import (
    TEST_ONLY_NOT_CALIBRATED,
    TimeScaleStatus,
)
from mechanistic_mv.mechanics.pair_interaction import (
    HydrogelEffectivePairPotential,
    PairDataValidationStatus,
    PairForceMetadata,
    PairForceTable,
    convert_single_pair_table_to_kac,
)
from mechanistic_mv.mechanics.pair_potential import (
    TestOnlyGaussianRepulsion,
    mean_field_pair_force_newton,
)
from mechanistic_mv.mechanics.parameters import PhysicalParameters
from mechanistic_mv.mechanics.particle_model import ParticleMechanicalState
from mechanistic_mv.particle_sim.empirical_density import empirical_density
from mechanistic_mv.particle_sim.langevin_interacting import (
    overdamped_langevin_step,
)


def test_only_single_pair_table(
    *,
    minimum_distance_m: float = 0.0,
    reference_force_tolerance_newton: float = 0.0,
) -> PairForceTable:
    reference_distance_m = 4.0e-6
    distance = np.linspace(minimum_distance_m, reference_distance_m, 17)
    force = 1.5e-12 * (1.0 - distance / reference_distance_m) ** 2
    metadata = PairForceMetadata(
        dataset_id="TEST_ONLY_NOT_CALIBRATED_single_pair_table",
        source="analytic TEST_ONLY single-pair fixture; not contact-FEM data",
        physical_status=TEST_ONLY_NOT_CALIBRATED,
        solver_status="TEST_ONLY_NOT_A_CONTACT_SOLVER",
        validation_status=PairDataValidationStatus.PASSED,
        time_scale_status=TimeScaleStatus.UNVERIFIED,
        scaling=PairForceScaling.UNSCALED_SINGLE_PAIR,
        reference_distance_m=reference_distance_m,
        reference_force_tolerance_newton=reference_force_tolerance_newton,
    )
    return PairForceTable(distance, force, metadata)


def hydrogel_potential(
    force_data: PairForceTable,
    density_convention: DensityConvention,
) -> HydrogelEffectivePairPotential:
    return HydrogelEffectivePairPotential(
        force_data,
        derivative_absolute_tolerance_newton=1.0e-20,
        derivative_relative_tolerance=2.0e-7,
        finite_difference_step_fraction=1.0e-4,
        density_convention=density_convention,
    )


class DensityRepresentationTests(unittest.TestCase):
    def test_empirical_and_initial_density_integrate_to_one_or_population(self) -> None:
        population = 9
        grid = CartesianGrid(
            RectangularDomain((0.0, 1.0), (0.0, 1.0)),
            8,
            6,
        )
        positions = np.random.default_rng(20260824).uniform(
            0.0,
            1.0,
            size=(population, 2),
        )
        probability = empirical_density(positions, grid)
        number = empirical_density(
            positions,
            grid,
            density_convention=DensityConvention.NUMBER,
        )
        self.assertAlmostEqual(
            float(np.sum(probability) * grid.cell_area_m2),
            1.0,
        )
        self.assertAlmostEqual(
            float(np.sum(number) * grid.cell_area_m2),
            population,
        )
        np.testing.assert_array_equal(number, population * probability)

        gaussian_probability = gaussian_density(grid, (0.4, 0.6), 0.15)
        gaussian_number = gaussian_density(
            grid,
            (0.4, 0.6),
            0.15,
            density_convention=DensityConvention.NUMBER,
            population_count=population,
        )
        np.testing.assert_allclose(
            gaussian_number,
            population * gaussian_probability,
            rtol=5.0e-16,
            atol=0.0,
        )
        with self.assertRaisesRegex(ValueError, "requires population_count"):
            gaussian_density(
                grid,
                (0.4, 0.6),
                0.15,
                density_convention=DensityConvention.NUMBER,
            )


class SinglePairToKacConversionTests(unittest.TestCase):
    def test_conversion_multiplies_values_and_records_immutable_evidence(self) -> None:
        population = 7
        source = test_only_single_pair_table(
            reference_force_tolerance_newton=2.0e-18
        )
        converted = convert_single_pair_table_to_kac(
            source,
            population_count=population,
            population_count_provenance=(
                "TEST_ONLY population equals the explicit particle fixture size"
            ),
        )
        np.testing.assert_array_equal(
            converted.center_distance_m,
            source.center_distance_m,
        )
        np.testing.assert_allclose(
            converted.radial_force_newton,
            population * source.radial_force_newton,
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(
            converted.metadata.reference_force_tolerance_newton,
            population * source.metadata.reference_force_tolerance_newton,
        )
        self.assertEqual(
            converted.metadata.scaling,
            PairForceScaling.KAC_NORMALIZED_PROBABILITY,
        )
        self.assertEqual(
            converted.metadata.native_scaling,
            PairForceScaling.UNSCALED_SINGLE_PAIR,
        )
        evidence = converted.metadata.scaling_conversion
        self.assertEqual(evidence.population_count, population)
        self.assertEqual(evidence.source_dataset_id, source.metadata.dataset_id)
        self.assertEqual(evidence.force_multiplier, population)
        single_pair_potential = hydrogel_potential(
            source,
            DensityConvention.NUMBER,
        )
        kac_potential = hydrogel_potential(
            converted,
            DensityConvention.PROBABILITY,
        )
        radii = np.linspace(0.2e-6, 3.8e-6, 11)
        np.testing.assert_allclose(
            kac_potential.radial_potential_joule(radii),
            population * single_pair_potential.radial_potential_joule(radii),
            rtol=1.0e-12,
            atol=1.0e-32,
        )
        with self.assertRaises(FrozenInstanceError):
            evidence.population_count = population + 1
        self.assertEqual(
            source.metadata.scaling,
            PairForceScaling.UNSCALED_SINGLE_PAIR,
        )

    def test_conversion_rejects_relabel_and_invalid_population_or_provenance(self) -> None:
        source = test_only_single_pair_table()
        with self.assertRaisesRegex(ValueError, "explicit numerical conversion"):
            replace(
                source.metadata,
                scaling=PairForceScaling.KAC_NORMALIZED_PROBABILITY,
            )
        for invalid_population in (True, 0, -3, 2.5):
            with self.subTest(population_count=invalid_population):
                with self.assertRaises((TypeError, ValueError)):
                    convert_single_pair_table_to_kac(
                        source,
                        population_count=invalid_population,
                        population_count_provenance="TEST_ONLY explicit population",
                    )
        with self.assertRaisesRegex(ValueError, "must be specific"):
            convert_single_pair_table_to_kac(
                source,
                population_count=4,
                population_count_provenance="unknown",
            )
        converted = convert_single_pair_table_to_kac(
            source,
            population_count=4,
            population_count_provenance="TEST_ONLY explicit population",
        )
        with self.assertRaisesRegex(ValueError, "UNSCALED_SINGLE_PAIR"):
            convert_single_pair_table_to_kac(
                converted,
                population_count=4,
                population_count_provenance="TEST_ONLY repeated conversion",
            )


class ParticleScalingEquivalenceTests(unittest.TestCase):
    def test_particle_force_langevin_and_energy_are_exactly_equivalent(self) -> None:
        population = 4
        parameters = PhysicalParameters()
        pair = TestOnlyGaussianRepulsion(
            0.4 * parameters.thermal_energy_joule,
            1.1e-6,
            density_convention=DensityConvention.NUMBER,
        )
        kac = TestOnlyGaussianRepulsion(
            population * pair.energy_scale_joule,
            pair.length_scale_m,
        )
        positions = np.asarray(
            [
                [3.0e-6, 3.0e-6],
                [4.1e-6, 3.2e-6],
                [3.4e-6, 4.5e-6],
                [5.0e-6, 4.8e-6],
            ]
        )
        probability_force = mean_field_pair_force_newton(positions, kac)
        number_force = mean_field_pair_force_newton(
            positions,
            pair,
            density_convention=DensityConvention.NUMBER,
        )
        np.testing.assert_allclose(
            probability_force,
            number_force,
            rtol=2.0e-16,
            atol=1.0e-30,
        )

        state = ParticleMechanicalState(positions, np.zeros_like(positions))
        probability_energy = discrete_particle_energy_joule(
            state,
            parameters,
            kac,
        )
        number_energy = discrete_particle_energy_joule(
            state,
            parameters,
            pair,
            density_convention=DensityConvention.NUMBER,
        )
        self.assertAlmostEqual(probability_energy, number_energy, places=32)

        domain = RectangularDomain((0.0, 8.0e-6), (0.0, 8.0e-6))
        probability_step, _ = overdamped_langevin_step(
            positions,
            parameters,
            kac,
            domain,
            dt_s=1.0e-3,
            rng=np.random.default_rng(81),
        )
        number_step, _ = overdamped_langevin_step(
            positions,
            parameters,
            pair,
            domain,
            dt_s=1.0e-3,
            rng=np.random.default_rng(81),
            density_convention=DensityConvention.NUMBER,
        )
        np.testing.assert_allclose(
            probability_step,
            number_step,
            rtol=0.0,
            atol=1.0e-21,
        )

    def test_particle_scaling_mismatch_and_kac_population_mismatch_fail(self) -> None:
        positions = np.asarray([[0.0, 0.0], [1.0e-6, 0.0], [2.0e-6, 0.0]])
        single_pair = TestOnlyGaussianRepulsion(
            1.0e-21,
            1.0e-6,
            density_convention=DensityConvention.NUMBER,
        )
        kac = TestOnlyGaussianRepulsion(3.0e-21, 1.0e-6)
        with self.assertRaisesRegex(ValueError, "scaling mismatch"):
            mean_field_pair_force_newton(positions, single_pair)
        with self.assertRaisesRegex(ValueError, "scaling mismatch"):
            mean_field_pair_force_newton(
                positions,
                kac,
                density_convention=DensityConvention.NUMBER,
            )

        converted = convert_single_pair_table_to_kac(
            test_only_single_pair_table(),
            population_count=4,
            population_count_provenance="TEST_ONLY four-particle Kac scale",
        )
        converted_potential = hydrogel_potential(
            converted,
            DensityConvention.PROBABILITY,
        )
        with self.assertRaisesRegex(ValueError, "runtime population"):
            mean_field_pair_force_newton(positions, converted_potential)


class ContinuumScalingEquivalenceTests(unittest.TestCase):
    def test_convolution_flux_step_and_free_energy_scaling_are_equivalent(self) -> None:
        population = 5
        parameters = PhysicalParameters()
        grid = CartesianGrid(
            RectangularDomain((0.0, 6.0e-6), (0.0, 5.0e-6)),
            6,
            5,
        )
        pair = TestOnlyGaussianRepulsion(
            0.3 * parameters.thermal_energy_joule,
            1.0e-6,
            density_convention=DensityConvention.NUMBER,
        )
        kac = TestOnlyGaussianRepulsion(
            population * pair.energy_scale_joule,
            pair.length_scale_m,
        )
        probability = gaussian_density(grid, (2.6e-6, 2.8e-6), 0.8e-6)
        number = population * probability

        direct_probability = direct_pair_convolution_joule(
            probability,
            grid,
            kac,
            population_count=population,
        )
        direct_number = direct_pair_convolution_joule(
            number,
            grid,
            pair,
            density_convention=DensityConvention.NUMBER,
            population_count=population,
        )
        np.testing.assert_allclose(
            direct_probability,
            direct_number,
            rtol=3.0e-15,
            atol=1.0e-35,
        )

        probability_solver = McKeanVlasovSolver(
            grid,
            parameters,
            kac,
            population_count=population,
        )
        number_solver = McKeanVlasovSolver(
            grid,
            parameters,
            pair,
            density_convention=DensityConvention.NUMBER,
            population_count=population,
        )
        self.assertEqual(probability_solver.expected_density_mass, 1.0)
        self.assertEqual(number_solver.expected_density_mass, population)
        self.assertEqual(
            number_solver.reference_density_per_m2,
            population * probability_solver.reference_density_per_m2,
        )
        probability_flux, probability_interaction = probability_solver.face_fluxes(
            probability
        )
        number_flux, number_interaction = number_solver.face_fluxes(number)
        np.testing.assert_allclose(
            probability_interaction,
            number_interaction,
            rtol=3.0e-15,
            atol=1.0e-35,
        )
        np.testing.assert_allclose(
            number_flux.x_per_m_s,
            population * probability_flux.x_per_m_s,
            rtol=3.0e-15,
            atol=1.0e-22,
        )
        np.testing.assert_allclose(
            number_flux.y_per_m_s,
            population * probability_flux.y_per_m_s,
            rtol=3.0e-15,
            atol=1.0e-22,
        )

        probability_next, _ = probability_solver.step(probability, 1.0e-3)
        number_next, _ = number_solver.step(number, 1.0e-3)
        np.testing.assert_allclose(
            number_next,
            population * probability_next,
            rtol=3.0e-15,
            atol=1.0e-7,
        )

        probability_energy = probability_solver.free_energy(probability)
        number_energy = number_solver.free_energy(number)
        self.assertAlmostEqual(
            number_energy.entropy_joule,
            population * probability_energy.entropy_joule,
            places=31,
        )
        self.assertAlmostEqual(
            number_energy.interaction_joule,
            population * probability_energy.interaction_joule,
            places=31,
        )
        self.assertAlmostEqual(
            number_energy.total_joule,
            population * probability_energy.total_joule,
            places=31,
        )

    def test_solver_rejects_scaling_population_and_total_mass_mismatch(self) -> None:
        population = 4
        grid = CartesianGrid(
            RectangularDomain((0.0, 4.0e-6), (0.0, 4.0e-6)),
            4,
            4,
        )
        parameters = PhysicalParameters()
        single_pair = TestOnlyGaussianRepulsion(
            1.0e-21,
            1.0e-6,
            density_convention=DensityConvention.NUMBER,
        )
        kac = TestOnlyGaussianRepulsion(population * 1.0e-21, 1.0e-6)
        with self.assertRaisesRegex(ValueError, "scaling mismatch"):
            McKeanVlasovSolver(grid, parameters, single_pair)
        with self.assertRaisesRegex(ValueError, "scaling mismatch"):
            McKeanVlasovSolver(
                grid,
                parameters,
                kac,
                density_convention=DensityConvention.NUMBER,
                population_count=population,
            )
        with self.assertRaisesRegex(ValueError, "requires population_count"):
            McKeanVlasovSolver(
                grid,
                parameters,
                single_pair,
                density_convention=DensityConvention.NUMBER,
            )

        probability_solver = McKeanVlasovSolver(grid, parameters, kac)
        number_solver = McKeanVlasovSolver(
            grid,
            parameters,
            single_pair,
            density_convention=DensityConvention.NUMBER,
            population_count=population,
        )
        probability = gaussian_density(grid, (2.0e-6, 2.0e-6), 0.7e-6)
        with self.assertRaisesRegex(ValueError, "density mass"):
            probability_solver.step(population * probability, 1.0e-3)
        with self.assertRaisesRegex(ValueError, "density mass"):
            number_solver.step(probability, 1.0e-3)


class ShortRangeAdmissionTests(unittest.TestCase):
    def test_particle_only_table_accepts_valid_distances_and_rejects_short_range(self) -> None:
        minimum_distance = 0.5e-6
        potential = hydrogel_potential(
            test_only_single_pair_table(minimum_distance_m=minimum_distance),
            DensityConvention.NUMBER,
        )
        self.assertFalse(potential.continuum_ready)
        valid_positions = np.asarray(
            [[0.0, 0.0], [0.8e-6, 0.0], [2.0e-6, 0.0]]
        )
        valid_force = mean_field_pair_force_newton(
            valid_positions,
            potential,
            density_convention=DensityConvention.NUMBER,
        )
        self.assertTrue(np.all(np.isfinite(valid_force)))

        with self.assertRaisesRegex(ValueError, "below the validated force table"):
            mean_field_pair_force_newton(
                np.asarray([[0.0, 0.0], [0.25e-6, 0.0]]),
                potential,
                density_convention=DensityConvention.NUMBER,
            )
        with self.assertRaisesRegex(ValueError, "below the validated force table"):
            mean_field_pair_force_newton(
                np.asarray([[0.0, 0.0], [0.0, 0.0]]),
                potential,
                density_convention=DensityConvention.NUMBER,
            )

    def test_particle_only_table_is_rejected_before_continuum_kernel_creation(self) -> None:
        population = 3
        grid = CartesianGrid(
            RectangularDomain((0.0, 3.0e-6), (0.0, 3.0e-6)),
            4,
            4,
        )
        potential = hydrogel_potential(
            test_only_single_pair_table(minimum_distance_m=0.5e-6),
            DensityConvention.NUMBER,
        )
        number_density = np.full(
            (grid.ny, grid.nx),
            population / (grid.nx * grid.ny * grid.cell_area_m2),
        )
        with self.assertRaisesRegex(ValueError, "zero displacement"):
            direct_pair_convolution_joule(
                number_density,
                grid,
                potential,
                density_convention=DensityConvention.NUMBER,
                population_count=population,
            )
        with self.assertRaisesRegex(ValueError, "zero displacement"):
            FFTPairConvolver(
                grid,
                potential,
                density_convention=DensityConvention.NUMBER,
                population_count=population,
            )
        with self.assertRaisesRegex(ValueError, "zero displacement"):
            McKeanVlasovSolver(
                grid,
                PhysicalParameters(),
                potential,
                density_convention=DensityConvention.NUMBER,
                population_count=population,
            )

    def test_table_covering_zero_is_continuum_ready(self) -> None:
        population = 3
        grid = CartesianGrid(
            RectangularDomain((0.0, 3.0e-6), (0.0, 3.0e-6)),
            4,
            4,
        )
        potential = hydrogel_potential(
            test_only_single_pair_table(minimum_distance_m=0.0),
            DensityConvention.NUMBER,
        )
        self.assertTrue(potential.continuum_ready)
        number_density = np.full(
            (grid.ny, grid.nx),
            population / (grid.nx * grid.ny * grid.cell_area_m2),
        )
        convolver = FFTPairConvolver(
            grid,
            potential,
            density_convention=DensityConvention.NUMBER,
            population_count=population,
        )
        interaction = convolver.convolve_joule(number_density)
        self.assertTrue(np.all(np.isfinite(interaction)))


if __name__ == "__main__":
    unittest.main()
