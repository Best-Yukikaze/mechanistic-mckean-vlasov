"""Focused unit checks for source-gated magnetic MV mechanics.

The numerical constants in this file are explicit dimensional fixtures only;
they are not loaded as physical-validation source data or used in a continuum
experiment.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from mechanistic_mv.mechanics.magnetic_particle.dipolar_pair import (
    FieldOrientedDipoleInteraction,
    SourceBackedContactGeometry,
    sphere_volume_m3,
)
from mechanistic_mv.mechanics.magnetic_particle.potential import (
    LinearMagneticParticle,
    MagneticParticlePotential,
    TabulatedMagnetizationLaw,
)
from mechanistic_mv.mechanics.magnetic_particle.continuum_admission import (
    CURRENT_2D_MV_PHYSICAL_REDUCTION_INVALID,
    Current2DMVPhysicalReductionInvalid,
    assess_magnetic_2d_closure,
)
from mechanistic_mv.mechanics import magnetic_dipole_interaction as legacy_dipolar
from mechanistic_mv.mechanics import magnetic_particle_potential as legacy_potential
from mechanistic_mv.mechanics import magnetic_validation as legacy_admission


class _LinearMagnitudeField:
    """TEST_ONLY_DIMENSIONAL_UNIT_FIXTURE: B(x)=B0+g*x, in SI units."""

    name = "test_only_linear_magnetic_flux_density"

    def __init__(self, b0_tesla: float, gradient_tesla_per_m: float) -> None:
        self.b0_tesla = b0_tesla
        self.gradient_tesla_per_m = gradient_tesla_per_m

    def flux_density_tesla(self, positions_m: np.ndarray) -> np.ndarray:
        points = np.asarray(positions_m, dtype=np.float64)
        return self.b0_tesla + self.gradient_tesla_per_m * points[..., 0]

    def gradient_flux_density_tesla_per_m(self, positions_m: np.ndarray) -> np.ndarray:
        points = np.asarray(positions_m, dtype=np.float64)
        gradient = np.zeros(points.shape, dtype=np.float64)
        gradient[..., 0] = self.gradient_tesla_per_m
        return gradient


class MagneticParticlePhysicsGuardTests(unittest.TestCase):
    def test_legacy_physics_imports_reexport_the_canonical_api(self) -> None:
        self.assertIs(legacy_potential.LinearMagneticParticle, LinearMagneticParticle)
        self.assertIs(legacy_potential.MagneticParticlePotential, MagneticParticlePotential)
        self.assertIs(legacy_dipolar.FieldOrientedDipoleInteraction, FieldOrientedDipoleInteraction)
        self.assertIs(
            legacy_admission.assess_magnetic_2d_closure,
            assess_magnetic_2d_closure,
        )

    def test_missing_source_blocks_2d_reduction_and_any_continuum_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing_source_provenance_v1.json"
            admission = assess_magnetic_2d_closure(missing)
        self.assertEqual(admission.status, CURRENT_2D_MV_PHYSICAL_REDUCTION_INVALID)
        self.assertFalse(admission.physical_simulation_allowed)
        self.assertIn("rho_2D", admission.unit_audit)
        self.assertIn("W_2D_convolution", admission.unit_audit)
        with self.assertRaises(Current2DMVPhysicalReductionInvalid):
            admission.require_runnable()

    def test_frozen_source_audit_reports_all_physical_prerequisites_before_any_pde_run(self) -> None:
        admission = assess_magnetic_2d_closure()
        self.assertEqual(admission.status, CURRENT_2D_MV_PHYSICAL_REDUCTION_INVALID)
        self.assertFalse(admission.physical_simulation_allowed)
        reasons = "\n".join(admission.missing_or_invalid)
        self.assertIn("slab_thickness_m", reasons)
        self.assertIn("susceptibility or nonlinear moment law", reasons)
        self.assertIn("B(x) and grad(B)(x)", reasons)
        self.assertIn("diffusivity D", reasons)
        self.assertIn("mobility M", reasons)
        self.assertIn("number-density conversion", reasons)

    def test_linear_energy_force_and_nonlinear_route_are_derivative_consistent(self) -> None:
        field = _LinearMagnitudeField(0.040, 1.7e3)
        linear = LinearMagneticParticle(
            chi_v_dimensionless=1.3,
            particle_volume_m3=2.1e-24,
            source_locator="TEST_ONLY_DIMENSIONAL_UNIT_FIXTURE",
            provenance_class="TEST_ONLY_NOT_CALIBRATED",
        )
        potential = MagneticParticlePotential(linear, linear.particle_volume_m3, field)
        point = np.array([[2.0e-6, 0.0]])
        force = potential.force_newton(point)[0, 0]
        epsilon = 1.0e-10
        plus = potential.potential_joule(point + np.array([[epsilon, 0.0]]))[0]
        minus = potential.potential_joule(point - np.array([[epsilon, 0.0]]))[0]
        self.assertAlmostEqual(force, -(plus - minus) / (2.0 * epsilon), delta=abs(force) * 1.0e-7)
        self.assertLess(potential.potential_joule(point)[0], 0.0)

        nonlinear = TabulatedMagnetizationLaw(
            flux_density_samples_tesla=(0.0, 0.04, 0.08),
            magnetization_samples_A_per_m=(0.0, 1.0e4, 1.4e4),
            source_locator="TEST_ONLY_DIMENSIONAL_UNIT_FIXTURE",
            provenance_class="TEST_ONLY_NOT_CALIBRATED",
        )
        nonlinear_potential = MagneticParticlePotential(nonlinear, 2.1e-24, field)
        nonlinear_force = nonlinear_potential.force_newton(point)[0, 0]
        nonlinear_plus = nonlinear_potential.potential_joule(point + np.array([[epsilon, 0.0]]))[0]
        nonlinear_minus = nonlinear_potential.potential_joule(point - np.array([[epsilon, 0.0]]))[0]
        self.assertAlmostEqual(
            nonlinear_force,
            -(nonlinear_plus - nonlinear_minus) / (2.0 * epsilon),
            delta=abs(nonlinear_force) * 1.0e-6,
        )

    def test_field_oriented_dipole_order_contact_and_lambda_scaling(self) -> None:
        diameter_10_m = 10.0e-9
        diameter_40_m = 40.0e-9
        field_tesla = np.asarray(0.040)
        chi_v = 1.0
        law_10 = LinearMagneticParticle(chi_v, sphere_volume_m3(diameter_10_m), "TEST_ONLY_DIMENSIONAL_UNIT_FIXTURE", "TEST_ONLY_NOT_CALIBRATED")
        law_40 = LinearMagneticParticle(chi_v, sphere_volume_m3(diameter_40_m), "TEST_ONLY_DIMENSIONAL_UNIT_FIXTURE", "TEST_ONLY_NOT_CALIBRATED")
        contact_10 = SourceBackedContactGeometry(diameter_10_m, "TEST_ONLY_DIMENSIONAL_UNIT_FIXTURE", "TEST_ONLY_NOT_CALIBRATED")
        contact_40 = SourceBackedContactGeometry(diameter_40_m, "TEST_ONLY_DIMENSIONAL_UNIT_FIXTURE", "TEST_ONLY_NOT_CALIBRATED")
        pair_10 = FieldOrientedDipoleInteraction(float(law_10.dipole_moment_A_m2(field_tesla)), (0.0, 0.0, 1.0), contact_10, "TEST_ONLY_DIMENSIONAL_UNIT_FIXTURE", "TEST_ONLY_NOT_CALIBRATED")
        pair_40 = FieldOrientedDipoleInteraction(float(law_40.dipole_moment_A_m2(field_tesla)), (0.0, 0.0, 1.0), contact_40, "TEST_ONLY_DIMENSIONAL_UNIT_FIXTURE", "TEST_ONLY_NOT_CALIBRATED")
        lambda_ratio = pair_40.coupling_lambda_at_contact(4.0e-21) / pair_10.coupling_lambda_at_contact(4.0e-21)
        self.assertAlmostEqual(lambda_ratio, 64.0, places=10)
        self.assertGreater(pair_40.coupling_lambda_at_contact(4.0e-21), pair_10.coupling_lambda_at_contact(4.0e-21))

        parallel_energy = pair_10.energy_joule(np.array([[0.0, 0.0, 2.0 * diameter_10_m]]))[0]
        transverse_energy = pair_10.energy_joule(np.array([[2.0 * diameter_10_m, 0.0, 0.0]]))[0]
        self.assertLess(parallel_energy, 0.0)
        self.assertGreater(transverse_energy, 0.0)
        with self.assertRaises(ValueError):
            pair_10.energy_joule(np.array([[0.0, 0.0, 0.5 * diameter_10_m]]))


if __name__ == "__main__":
    unittest.main()
