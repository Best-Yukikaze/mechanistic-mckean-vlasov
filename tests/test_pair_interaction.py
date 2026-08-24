from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
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
    ContactSolveStatus,
    HydrogelContactFEMNotAvailable,
    HydrogelEffectivePairPotential,
    MeshOrResolutionMetadata,
    PairContactSample,
    PairContactSweep,
    PairContactSweepMetadata,
    PairDataValidationStatus,
    PairForceMetadata,
    PairForceScaling,
    PairForceTable,
    PchipIntegratedForceLaw,
    QuantityDefinition,
    ScalarDiagnostic,
    TwoSphereGeometry,
    pair_force_table_from_contact_sweep,
    radial_reaction_force_newton,
)
from mechanistic_mv.mechanics.pair_potential import mean_field_pair_force_newton


def test_only_hydrogel_parameters() -> HydrogelParameters:
    return HydrogelParameters(
        network_density_times_solvent_volume=0.025,
        flory_huggins_chi=0.31,
        initial_polymer_volume_fraction=0.18,
        delta_chemical_potential_over_kbt=0.0,
        thermal_energy_density_pa=2.4e5,
        calibration_status=TEST_ONLY_NOT_CALIBRATED,
    )


def test_only_sweep_metadata(
    *,
    validation_status: PairDataValidationStatus = PairDataValidationStatus.PASSED,
) -> PairContactSweepMetadata:
    return PairContactSweepMetadata(
        dataset_id="TEST_ONLY_NOT_CALIBRATED_contact_sweep",
        source="analytic unit-test records; not contact-FEM output",
        physical_status=TEST_ONLY_NOT_CALIBRATED,
        geometry=TwoSphereGeometry(
            first_radius_m=0.4e-6,
            second_radius_m=0.45e-6,
            calibration_status=TEST_ONLY_NOT_CALIBRATED,
        ),
        material_parameters=test_only_hydrogel_parameters(),
        time_scale=assess_time_scale_separation(
            tau_gel_s=1.0e-3,
            tau_swarm_s=1.0,
            required_max_ratio=1.0e-2,
        ),
        solver_name="TEST_ONLY_ANALYTIC_CONTACT_FIXTURE",
        solver_version="TEST_ONLY_VERSION_1",
        solver_configuration_id="TEST_ONLY_CONFIG_A",
        resolution_method="TEST_ONLY_STRUCTURED_DISCRETIZATION",
        mechanical_boundary_conditions=(
            "frictionless normal non-penetration with prescribed center distance"
        ),
        solvent_bath_boundary_conditions=(
            "fixed dimensionless bath chemical potential from test parameters"
        ),
        contact_law="frictionless non-adhesive normal contact",
        radial_reaction_force_sign_convention=(
            "positive radial force separates the two sphere centers"
        ),
        total_free_energy_reference="zero at the declared separated reference distance",
        contact_measure_definition=QuantityDefinition(
            quantity_name="contact_area",
            si_unit_or_one="m^2",
            description="area of active frictionless contact",
        ),
        maximum_deformation_definition=QuantityDefinition(
            quantity_name="maximum_principal_strain",
            si_unit_or_one="1",
            description="largest dimensionless principal strain in either gel",
        ),
        maximum_stress_definition=QuantityDefinition(
            quantity_name="maximum_cauchy_stress",
            si_unit_or_one="Pa",
            description="largest Cauchy-stress magnitude in either gel",
        ),
        validation_status=validation_status,
        reference_distance_m=4.0e-6,
        reference_force_tolerance_newton=0.0,
    )


def test_only_resolution(
    *,
    resolution_id: str,
    solver_configuration_id: str = "TEST_ONLY_CONFIG_A",
) -> MeshOrResolutionMetadata:
    return MeshOrResolutionMetadata(
        method="TEST_ONLY_STRUCTURED_DISCRETIZATION",
        resolution_id=resolution_id,
        solver_configuration_id=solver_configuration_id,
        degrees_of_freedom=1200,
        characteristic_length_m=2.0e-8,
        solver_iterations=7,
        final_residual_dimensionless=1.0e-10,
    )


def test_only_contact_sample(
    center_distance_m: float,
    radial_force_newton: float,
    *,
    resolution_id: str,
) -> PairContactSample:
    return PairContactSample(
        center_distance_m=center_distance_m,
        radial_reaction_force_newton=radial_force_newton,
        total_free_energy_joule=(
            radial_force_newton * (4.0e-6 - center_distance_m) / 3.0
        ),
        contact_measure_si_or_dimensionless=max(
            0.0, (4.0e-6 - center_distance_m) * 1.0e-7
        ),
        maximum_deformation_si_or_dimensionless=max(
            0.0, (4.0e-6 - center_distance_m) / 4.0e-6
        ),
        maximum_stress_pascal=radial_force_newton / 1.0e-12,
        solvent_state_summary=(
            ScalarDiagnostic(
                quantity_name="mean_polymer_volume_fraction",
                value_si_or_dimensionless=0.18,
                si_unit_or_one="1",
                description="domain-mean polymer volume fraction",
            ),
        ),
        solver_status=ContactSolveStatus.CONVERGED,
        mesh_or_resolution_metadata=test_only_resolution(
            resolution_id=resolution_id
        ),
        failure_reason=None,
    )


def test_only_contact_samples() -> tuple[PairContactSample, ...]:
    distance = np.linspace(1.0e-6, 4.0e-6, 5)
    force = 2.0e-12 * (1.0 - distance / distance[-1]) ** 2
    return tuple(
        test_only_contact_sample(
            float(radius),
            float(radial_force),
            resolution_id=f"TEST_ONLY_MESH_{index}",
        )
        for index, (radius, radial_force) in enumerate(zip(distance, force, strict=True))
    )


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


class PairContactSweepContractTests(unittest.TestCase):
    def test_converged_sample_is_complete_structured_and_immutable(self) -> None:
        sample = test_only_contact_samples()[0]
        self.assertGreater(sample.center_distance_m, 0.0)
        self.assertIsNotNone(sample.radial_reaction_force_newton)
        self.assertIsNotNone(sample.total_free_energy_joule)
        self.assertIsNotNone(sample.contact_measure_si_or_dimensionless)
        self.assertIsNotNone(sample.maximum_deformation_si_or_dimensionless)
        self.assertIsNotNone(sample.maximum_stress_pascal)
        self.assertEqual(sample.solver_status, ContactSolveStatus.CONVERGED)
        self.assertEqual(
            sample.solvent_state_summary[0].si_unit_or_one,
            "1",
        )
        self.assertTrue(sample.mesh_or_resolution_metadata.resolution_id)
        with self.assertRaises(FrozenInstanceError):
            sample.center_distance_m = 2.0e-6

    def test_failed_sample_is_preserved_and_rejected_without_partial_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "partial physical outputs"):
            replace(
                test_only_contact_samples()[1],
                solver_status=ContactSolveStatus.FAILED,
                failure_reason="TEST_ONLY_NONLINEAR_SOLVER_DIVERGED",
            )

        failed = PairContactSample(
            center_distance_m=2.5e-6,
            radial_reaction_force_newton=None,
            total_free_energy_joule=None,
            contact_measure_si_or_dimensionless=None,
            maximum_deformation_si_or_dimensionless=None,
            maximum_stress_pascal=None,
            solvent_state_summary=None,
            solver_status=ContactSolveStatus.FAILED,
            mesh_or_resolution_metadata=test_only_resolution(
                resolution_id="TEST_ONLY_FAILED_MESH"
            ),
            failure_reason="TEST_ONLY_NONLINEAR_SOLVER_DIVERGED",
        )
        samples = list(test_only_contact_samples())
        samples[2] = failed
        failed_sweep = PairContactSweep(
            tuple(samples),
            test_only_sweep_metadata(
                validation_status=PairDataValidationStatus.FAILED
            ),
        )
        with self.assertRaisesRegex(ValueError, "non-converged samples at indices 2"):
            pair_force_table_from_contact_sweep(failed_sweep)
        with self.assertRaisesRegex(ValueError, "inconsistent with a failed"):
            PairContactSweep(tuple(samples), test_only_sweep_metadata())

    def test_order_finite_and_common_metadata_are_strict(self) -> None:
        samples = test_only_contact_samples()
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            PairContactSweep(
                (samples[0], samples[2], samples[1], samples[3], samples[4]),
                test_only_sweep_metadata(),
            )
        with self.assertRaisesRegex(ValueError, "maximum_stress_pascal must be finite"):
            replace(samples[0], maximum_stress_pascal=np.nan)
        with self.assertRaisesRegex(ValueError, "must be specific"):
            replace(
                test_only_resolution(resolution_id="TEST_ONLY_MESH"),
                method="unknown",
            )
        incomplete_resolution = replace(
            test_only_resolution(resolution_id="TEST_ONLY_INCOMPLETE_MESH"),
            degrees_of_freedom=None,
            characteristic_length_m=None,
        )
        with self.assertRaisesRegex(ValueError, "degrees_of_freedom"):
            replace(
                samples[0],
                mesh_or_resolution_metadata=incomplete_resolution,
            )
        with self.assertRaisesRegex(ValueError, "iterations and final residual"):
            replace(
                samples[0],
                mesh_or_resolution_metadata=replace(
                    samples[0].mesh_or_resolution_metadata,
                    final_residual_dimensionless=None,
                ),
            )

        inconsistent = list(samples)
        inconsistent[1] = replace(
            inconsistent[1],
            mesh_or_resolution_metadata=test_only_resolution(
                resolution_id="TEST_ONLY_OTHER_CONFIG_MESH",
                solver_configuration_id="TEST_ONLY_CONFIG_B",
            ),
        )
        with self.assertRaisesRegex(ValueError, "sweep solver_configuration_id"):
            PairContactSweep(tuple(inconsistent), test_only_sweep_metadata())

        inconsistent_method = list(samples)
        inconsistent_method[1] = replace(
            inconsistent_method[1],
            mesh_or_resolution_metadata=replace(
                inconsistent_method[1].mesh_or_resolution_metadata,
                method="TEST_ONLY_DIFFERENT_DISCRETIZATION",
            ),
        )
        with self.assertRaisesRegex(ValueError, "sweep resolution_method"):
            PairContactSweep(tuple(inconsistent_method), test_only_sweep_metadata())

        inconsistent_solvent = list(samples)
        inconsistent_solvent[1] = replace(
            inconsistent_solvent[1],
            solvent_state_summary=(
                ScalarDiagnostic(
                    quantity_name="maximum_polymer_volume_fraction",
                    value_si_or_dimensionless=0.2,
                    si_unit_or_one="1",
                    description="maximum polymer volume fraction",
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "schema must be consistent"):
            PairContactSweep(
                tuple(inconsistent_solvent),
                test_only_sweep_metadata(),
            )

    def test_validated_sweep_converts_all_samples_without_rescaling(self) -> None:
        samples = test_only_contact_samples()
        sweep = PairContactSweep(samples, test_only_sweep_metadata())
        table = pair_force_table_from_contact_sweep(sweep)
        np.testing.assert_array_equal(
            table.center_distance_m,
            np.asarray([sample.center_distance_m for sample in samples]),
        )
        np.testing.assert_array_equal(
            table.radial_force_newton,
            np.asarray([sample.radial_reaction_force_newton for sample in samples]),
        )
        self.assertEqual(
            table.metadata.scaling,
            PairForceScaling.UNSCALED_SINGLE_PAIR,
        )
        self.assertEqual(table.metadata.time_scale_status, TimeScaleStatus.SATISFIED)
        self.assertIn("ALL_SAMPLES_CONVERGED", table.metadata.solver_status)
        with self.assertRaises(ValueError):
            table.radial_force_newton[0] = 0.0

        unverified_metadata = replace(
            test_only_sweep_metadata(),
            time_scale=assess_time_scale_separation(
                tau_gel_s=None,
                tau_swarm_s=None,
                required_max_ratio=None,
            ),
        )
        with self.assertRaisesRegex(ValueError, "verified tau_gel"):
            PairContactSweep(samples, unverified_metadata).to_pair_force_table()


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
        with self.assertRaisesRegex(ValueError, "scaling mismatch"):
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

        particle_only = HydrogelEffectivePairPotential(
            test_only_force_table(minimum_distance_m=0.5e-6),
            derivative_absolute_tolerance_newton=1.0e-20,
            derivative_relative_tolerance=2.0e-7,
            finite_difference_step_fraction=1.0e-4,
        )
        self.assertFalse(particle_only.continuum_ready)
        grid = CartesianGrid(
            RectangularDomain((0.0, 2.0e-6), (0.0, 2.0e-6)),
            4,
            4,
        )
        with self.assertRaisesRegex(ValueError, "zero displacement"):
            FFTPairConvolver(grid, particle_only)

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
        parameters = test_only_hydrogel_parameters()
        time_scale = assess_time_scale_separation(
            tau_gel_s=None, tau_swarm_s=None, required_max_ratio=None
        )
        with self.assertRaisesRegex(NotImplementedError, "contact FEM is unavailable"):
            HydrogelContactFEMNotAvailable().solve_contact_sweep(
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
