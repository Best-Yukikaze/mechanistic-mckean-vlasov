from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from mechanistic_mv.mechanics.hydrogel import (
    HydrogelParameters,
    TEST_ONLY_NOT_CALIBRATED,
    TimeScaleStatus,
    assess_time_scale_separation,
    dimensionless_gibbs_free_energy_density,
    first_piola_stress_pa,
    gibbs_conjugate_solvent_content_dimensionless,
    gibbs_free_energy_density_pa,
    independent_helmholtz_chemical_potential_over_kbt,
    independent_helmholtz_free_energy_density_pa,
    reduced_helmholtz_free_energy_density_pa,
    source_dry_solvent_content_times_molecular_volume,
)


def test_only_parameters(*, delta_mu_over_kbt: float = 0.12) -> HydrogelParameters:
    return HydrogelParameters(
        network_density_times_solvent_volume=0.025,
        flory_huggins_chi=0.31,
        initial_polymer_volume_fraction=0.18,
        delta_chemical_potential_over_kbt=delta_mu_over_kbt,
        thermal_energy_density_pa=2.4e5,
        calibration_status=TEST_ONLY_NOT_CALIBRATED,
    )


class HydrogelModelIITests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = test_only_parameters()
        self.deformation = np.asarray(
            [
                [1.08, 0.04, 0.00],
                [0.01, 1.03, 0.02],
                [0.00, 0.01, 1.02],
            ]
        )

    def test_first_piola_matches_gibbs_finite_difference(self) -> None:
        analytic = first_piola_stress_pa(self.deformation, self.parameters)
        step = 1.0e-6
        finite_difference = np.empty((3, 3))
        for row in range(3):
            for column in range(3):
                offset = np.zeros((3, 3))
                offset[row, column] = step
                finite_difference[row, column] = (
                    gibbs_free_energy_density_pa(
                        self.deformation + offset, self.parameters
                    )
                    - gibbs_free_energy_density_pa(
                        self.deformation - offset, self.parameters
                    )
                ) / (2.0 * step)
        np.testing.assert_allclose(
            analytic, finite_difference, rtol=2.0e-8, atol=2.0e-5
        )

    def test_chemical_conjugacy_matches_finite_difference(self) -> None:
        step = 1.0e-6
        plus = replace(
            self.parameters,
            delta_chemical_potential_over_kbt=(
                self.parameters.delta_chemical_potential_over_kbt + step
            ),
        )
        minus = replace(
            self.parameters,
            delta_chemical_potential_over_kbt=(
                self.parameters.delta_chemical_potential_over_kbt - step
            ),
        )
        derivative = -(
            dimensionless_gibbs_free_energy_density(self.deformation, plus)
            - dimensionless_gibbs_free_energy_density(self.deformation, minus)
        ) / (2.0 * step)
        conjugate = gibbs_conjugate_solvent_content_dimensionless(
            self.deformation, self.parameters
        )
        np.testing.assert_allclose(derivative, conjugate, rtol=2.0e-10, atol=1.0e-11)
        np.testing.assert_allclose(
            source_dry_solvent_content_times_molecular_volume(
                self.deformation, self.parameters
            ),
            conjugate / self.parameters.initial_polymer_volume_fraction,
        )

    def test_initial_state_is_stress_free_at_zero_delta_mu(self) -> None:
        parameters = test_only_parameters(delta_mu_over_kbt=0.0)
        self.assertAlmostEqual(
            float(dimensionless_gibbs_free_energy_density(np.eye(3), parameters)),
            0.0,
            places=14,
        )
        np.testing.assert_allclose(
            first_piola_stress_pa(np.eye(3), parameters),
            0.0,
            atol=2.0e-11,
        )

    def test_reduced_legendre_identity_and_independent_closure_failure(self) -> None:
        gibbs = gibbs_free_energy_density_pa(self.deformation, self.parameters)
        conjugate = gibbs_conjugate_solvent_content_dimensionless(
            self.deformation, self.parameters
        )
        expected_helmholtz = gibbs + (
            self.parameters.thermal_energy_density_pa
            * self.parameters.bath_chemical_potential_over_kbt
            * conjugate
        )
        self.assertAlmostEqual(
            float(
                reduced_helmholtz_free_energy_density_pa(
                    self.deformation, self.parameters
                )
            ),
            float(expected_helmholtz),
            places=10,
        )
        with self.assertRaisesRegex(NotImplementedError, "independent Helmholtz"):
            independent_helmholtz_free_energy_density_pa(
                self.deformation, np.asarray(1.0), self.parameters
            )
        with self.assertRaisesRegex(NotImplementedError, "independent F and C"):
            independent_helmholtz_chemical_potential_over_kbt(
                self.deformation, np.asarray(1.0), self.parameters
            )

    def test_parameter_and_deformation_domains_are_strict(self) -> None:
        with self.assertRaises(TypeError):
            HydrogelParameters()  # type: ignore[call-arg]
        with self.assertRaisesRegex(ValueError, "phi0"):
            replace(self.parameters, initial_polymer_volume_fraction=1.0)
        with self.assertRaisesRegex(ValueError, r"det\(F\) must be positive"):
            first_piola_stress_pa(np.diag([-1.0, 1.0, 1.0]), self.parameters)
        with self.assertRaisesRegex(ValueError, r"det\(F\) > phi0"):
            first_piola_stress_pa(np.diag([0.1, 1.0, 1.0]), self.parameters)

    def test_batched_analytic_evaluation_matches_scalar_calls(self) -> None:
        batch = np.stack((self.deformation, np.eye(3), 1.04 * np.eye(3)))
        batch_energy = gibbs_free_energy_density_pa(batch, self.parameters)
        batch_stress = first_piola_stress_pa(batch, self.parameters)
        scalar_energy = np.asarray(
            [gibbs_free_energy_density_pa(value, self.parameters) for value in batch]
        )
        scalar_stress = np.stack(
            [first_piola_stress_pa(value, self.parameters) for value in batch]
        )
        np.testing.assert_allclose(batch_energy, scalar_energy, rtol=1.0e-14)
        np.testing.assert_allclose(batch_stress, scalar_stress, rtol=1.0e-14)


class HydrogelTimeScaleTests(unittest.TestCase):
    def test_missing_time_scale_or_criterion_never_defaults_to_pass(self) -> None:
        missing = assess_time_scale_separation(
            tau_gel_s=None, tau_swarm_s=10.0, required_max_ratio=0.01
        )
        self.assertEqual(missing.status, TimeScaleStatus.UNVERIFIED)
        no_criterion = assess_time_scale_separation(
            tau_gel_s=0.01, tau_swarm_s=10.0, required_max_ratio=None
        )
        self.assertEqual(no_criterion.status, TimeScaleStatus.UNVERIFIED)

    def test_explicit_time_scale_criterion_can_pass_or_fail(self) -> None:
        satisfied = assess_time_scale_separation(
            tau_gel_s=0.01, tau_swarm_s=10.0, required_max_ratio=0.002
        )
        violated = assess_time_scale_separation(
            tau_gel_s=0.03, tau_swarm_s=10.0, required_max_ratio=0.002
        )
        self.assertEqual(satisfied.status, TimeScaleStatus.SATISFIED)
        self.assertTrue(satisfied.quasi_static_pair_reduction_verified)
        self.assertEqual(violated.status, TimeScaleStatus.VIOLATED)
        self.assertFalse(violated.quasi_static_pair_reduction_verified)


if __name__ == "__main__":
    unittest.main()
