from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from mechanistic_mv.continuum.convolution import (
    FFTPairConvolver,
    direct_pair_convolution_joule,
)
from mechanistic_mv.mechanics.geometry import CartesianGrid, RectangularDomain
from mechanistic_mv.mechanics.hydrogel import (
    HydrogelParameters,
    TEST_ONLY_NOT_CALIBRATED,
    TimeScaleStatus,
    assess_time_scale_separation,
)
from mechanistic_mv.mechanics.pair_interaction import (
    HydrogelContactFEMNotAvailable,
    HydrogelEffectivePairPotential,
    PairDataValidationStatus,
    PairForceMetadata,
    PairForceScaling,
    PairForceTable,
    PchipIntegratedForceLaw,
    TwoSphereGeometry,
    radial_reaction_force_newton,
)
from mechanistic_mv.mechanics.pair_potential import mean_field_pair_force_newton


def test_only_force_table(
    *,
    scaling: PairForceScaling = PairForceScaling.KAC_NORMALIZED_PROBABILITY,
    minimum_distance_m: float = 0.0,
) -> PairForceTable:
    reference = 4.0e-6
    distance = np.linspace(minimum_distance_m, reference, 17)
    normalized = distance / reference
    force = 2.0e-12 * normalized * (1.0 - normalized) ** 2
    metadata = PairForceMetadata(
        dataset_id="TEST_ONLY_NOT_CALIBRATED_smooth_radial_force",
        source="analytic unit-test fixture; not physical contact data",
        physical_status=TEST_ONLY_NOT_CALIBRATED,
        solver_status="TEST_ONLY_NOT_A_CONTACT_SOLVER",
        validation_status=PairDataValidationStatus.PASSED,
        time_scale_status=TimeScaleStatus.UNVERIFIED,
        scaling=scaling,
        reference_distance_m=reference,
        reference_force_tolerance_newton=0.0,
    )
    return PairForceTable(distance, force, metadata)


def test_only_effective_potential() -> HydrogelEffectivePairPotential:
    return HydrogelEffectivePairPotential(
        test_only_force_table(),
        derivative_absolute_tolerance_newton=1.0e-20,
        derivative_relative_tolerance=2.0e-7,
        finite_difference_step_fraction=1.0e-4,
    )


class PairDataAndInterpolationTests(unittest.TestCase):
    def test_pchip_antiderivative_satisfies_force_potential_relation(self) -> None:
        potential = test_only_effective_potential()
        self.assertTrue(potential.validation_report.passed)
        self.assertGreater(potential.validation_report.evaluated_point_count, 0)
        radius = np.linspace(0.2e-6, 3.8e-6, 25)
        step = 1.0e-10
        numerical_force = -(
            potential.radial_potential_joule(radius + step)
            - potential.radial_potential_joule(radius - step)
        ) / (2.0 * step)
        np.testing.assert_allclose(
            numerical_force,
            potential.radial_force_newton(radius),
            rtol=3.0e-6,
            atol=1.0e-20,
        )

    def test_reference_and_extrapolation_behavior_are_explicit(self) -> None:
        table = test_only_force_table(minimum_distance_m=0.5e-6)
        law = PchipIntegratedForceLaw(table)
        self.assertEqual(float(law.potential_joule(table.metadata.reference_distance_m)), 0.0)
        self.assertEqual(float(law.potential_joule(5.0e-6)), 0.0)
        self.assertEqual(float(law.radial_force_newton(5.0e-6)), 0.0)
        with self.assertRaisesRegex(ValueError, "below the validated force table"):
            law.potential_joule(0.4e-6)

        force = table.radial_force_newton.copy()
        force[-1] = 0.5e-18
        metadata = replace(
            table.metadata, reference_force_tolerance_newton=1.0e-18
        )
        endpoint_law = PchipIntegratedForceLaw(
            PairForceTable(table.center_distance_m, force, metadata)
        )
        self.assertEqual(
            float(endpoint_law.radial_force_newton(metadata.reference_distance_m)),
            0.0,
        )

    def test_metadata_and_mean_field_scaling_gate_reject_invalid_data(self) -> None:
        table = test_only_force_table(
            scaling=PairForceScaling.UNSCALED_SINGLE_PAIR
        )
        with self.assertRaisesRegex(ValueError, "unscaled physical single-pair"):
            HydrogelEffectivePairPotential(
                table,
                derivative_absolute_tolerance_newton=1.0e-20,
                derivative_relative_tolerance=2.0e-7,
                finite_difference_step_fraction=1.0e-4,
            )
        valid = test_only_force_table()
        bad_force = valid.radial_force_newton.copy()
        bad_force[-1] = 1.0e-12
        with self.assertRaisesRegex(ValueError, "negligible-force tolerance"):
            PairForceTable(valid.center_distance_m, bad_force, valid.metadata)

        negative_force = valid.radial_force_newton.copy()
        negative_force[4] = -1.0e-18
        with self.assertRaisesRegex(ValueError, "non-negative"):
            PairForceTable(valid.center_distance_m, negative_force, valid.metadata)

        failed_metadata = replace(
            valid.metadata, validation_status=PairDataValidationStatus.FAILED
        )
        with self.assertRaisesRegex(ValueError, "validation_status=PASSED"):
            HydrogelEffectivePairPotential(
                PairForceTable(
                    valid.center_distance_m, valid.radial_force_newton, failed_metadata
                ),
                derivative_absolute_tolerance_newton=1.0e-20,
                derivative_relative_tolerance=2.0e-7,
                finite_difference_step_fraction=1.0e-4,
            )

        physical_unverified = replace(
            valid.metadata,
            physical_status="PHYSICAL_DATA_NOT_YET_TIME_SCALE_VALIDATED",
        )
        with self.assertRaisesRegex(ValueError, "tau_gel"):
            HydrogelEffectivePairPotential(
                PairForceTable(
                    valid.center_distance_m,
                    valid.radial_force_newton,
                    physical_unverified,
                ),
                derivative_absolute_tolerance_newton=1.0e-20,
                derivative_relative_tolerance=2.0e-7,
                finite_difference_step_fraction=1.0e-4,
            )

        with self.assertRaisesRegex(ValueError, "must cover r=0"):
            HydrogelEffectivePairPotential(
                test_only_force_table(minimum_distance_m=0.5e-6),
                derivative_absolute_tolerance_newton=1.0e-20,
                derivative_relative_tolerance=2.0e-7,
                finite_difference_step_fraction=1.0e-4,
            )

    def test_pchip_integral_is_compared_with_direct_trapezoid(self) -> None:
        potential = test_only_effective_potential()
        comparison = (
            potential.validation_report.pchip_vs_trapezoid_maximum_difference_joule
        )
        self.assertTrue(np.isfinite(comparison))
        self.assertGreaterEqual(comparison, 0.0)

        reference = 3.0e-6
        distance = np.linspace(0.0, reference, 9)
        force_scale = 1.5e-12
        force = force_scale * (1.0 - distance / reference)
        metadata = replace(
            test_only_force_table().metadata,
            dataset_id="TEST_ONLY_LINEAR_INTEGRATION",
            reference_distance_m=reference,
        )
        law = PchipIntegratedForceLaw(PairForceTable(distance, force, metadata))
        exact = force_scale * (reference - distance) ** 2 / (2.0 * reference)
        np.testing.assert_allclose(law.potential_joule(distance), exact, atol=1.0e-36)
        np.testing.assert_allclose(
            law.potential_at_nodes_by_trapezoid_joule(), exact, atol=1.0e-36
        )

    def test_scalar_and_vectorized_backend_results_match(self) -> None:
        potential = test_only_effective_potential()
        displacement = np.asarray(
            [[0.4e-6, 0.2e-6], [1.0e-6, -0.5e-6], [2.0e-6, 0.7e-6]]
        )
        vector_energy = potential.potential_joule(displacement)
        vector_force = potential.force_newton(displacement)
        scalar_energy = np.asarray(
            [potential.potential_joule(value) for value in displacement]
        )
        scalar_force = np.stack(
            [potential.force_newton(value) for value in displacement]
        )
        np.testing.assert_allclose(vector_energy, scalar_energy, rtol=1.0e-14)
        np.testing.assert_allclose(vector_force, scalar_force, rtol=1.0e-14)


class PairBridgeTests(unittest.TestCase):
    def test_missing_contact_problem_fails_explicitly(self) -> None:
        geometry = TwoSphereGeometry(
            first_radius_m=1.0e-6,
            second_radius_m=1.1e-6,
            calibration_status=TEST_ONLY_NOT_CALIBRATED,
        )
        parameters = HydrogelParameters(
            network_density_times_solvent_volume=0.025,
            flory_huggins_chi=0.31,
            initial_polymer_volume_fraction=0.18,
            delta_chemical_potential_over_kbt=0.0,
            thermal_energy_density_pa=2.4e5,
            calibration_status=TEST_ONLY_NOT_CALIBRATED,
        )
        time_scale = assess_time_scale_separation(
            tau_gel_s=None, tau_swarm_s=None, required_max_ratio=None
        )
        with self.assertRaisesRegex(NotImplementedError, "contact FEM is unavailable"):
            HydrogelContactFEMNotAvailable().solve_pair_force_table(
                geometry, parameters, np.asarray([2.5e-6, 2.2e-6]), time_scale
            )

    def test_same_validated_potential_serves_particles_and_continuum(self) -> None:
        potential = test_only_effective_potential()
        positions = np.asarray(
            [[0.2e-6, 0.2e-6], [0.7e-6, 0.3e-6], [0.5e-6, 0.8e-6]]
        )
        force = mean_field_pair_force_newton(positions, potential, chunk_size=2)
        self.assertTrue(np.all(np.isfinite(force)))
        np.testing.assert_allclose(np.sum(force, axis=0), 0.0, atol=1.0e-28)

        grid = CartesianGrid(
            RectangularDomain((0.0, 1.0e-6), (0.0, 1.0e-6)), 4, 4
        )
        density = np.arange(1.0, 17.0).reshape(4, 4)
        density /= np.sum(density) * grid.cell_area_m2
        direct = direct_pair_convolution_joule(density, grid, potential)
        accelerated = FFTPairConvolver(grid, potential).convolve_joule(density)
        np.testing.assert_allclose(accelerated, direct, rtol=2.0e-13, atol=1.0e-34)

    def test_reaction_projection_uses_repulsive_sign_convention(self) -> None:
        radial = radial_reaction_force_newton(
            np.asarray([[2.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
            np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        )
        np.testing.assert_array_equal(radial, np.asarray([2.0, -1.0]))


if __name__ == "__main__":
    unittest.main()
