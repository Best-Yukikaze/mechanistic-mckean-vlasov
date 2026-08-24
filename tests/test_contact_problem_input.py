from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
import unittest

import numpy as np

from mechanistic_mv.mechanics.density_scaling import DensityConvention
from mechanistic_mv.mechanics.hydrogel import (
    HydrogelParameters,
    TEST_ONLY_NOT_CALIBRATED,
    TimeScaleAssessment,
    TimeScaleStatus,
    assess_time_scale_separation,
)
from mechanistic_mv.mechanics.pair_interaction import (
    ContactInputPurpose,
    ContactLawAndTolerances,
    ContactProblemInput,
    ContactProblemReadiness,
    ContactResultsStatus,
    ContactSolverSpecification,
    DistanceScanPlan,
    HydrogelContactFEMNotAvailable,
    InputProvenance,
    InputVerificationStatus,
    MeanFieldScalingInput,
    MeanFieldScalingMode,
    MechanicalBoundaryConditions,
    MeshConvergencePlan,
    NormalContactLaw,
    PairAxisDefinition,
    ReactionForceConvention,
    ReactionForceSignConvention,
    RequiredInputStatus,
    ShortRangeClosureInput,
    ShortRangeClosureKind,
    ShortRangeInputStatus,
    SolventBathBoundaryConditions,
    SolverInputStatus,
    TotalFreeEnergyReference,
    TwoSphereGeometry,
)


def _test_only_provenance(name: str) -> InputProvenance:
    return InputProvenance(
        source_id=f"TEST_ONLY_{name}_SOURCE",
        source_description=(
            f"TEST_ONLY contract fixture for {name}; not calibrated physical data"
        ),
        verification_status=InputVerificationStatus.TEST_ONLY_NOT_CALIBRATED,
        verification_record_id=f"TEST_ONLY_{name}_RECORD",
    )


def _test_only_parameters() -> HydrogelParameters:
    return HydrogelParameters(
        network_density_times_solvent_volume=0.025,
        flory_huggins_chi=0.31,
        initial_polymer_volume_fraction=0.18,
        delta_chemical_potential_over_kbt=0.0,
        thermal_energy_density_pa=2.4e5,
        calibration_status=TEST_ONLY_NOT_CALIBRATED,
    )


def _test_only_complete_problem(
    *,
    minimum_supported_distance_m: float = 0.7e-6,
    with_zero_closure: bool = False,
) -> ContactProblemInput:
    parameters = _test_only_parameters()
    if with_zero_closure:
        short_range = ShortRangeClosureInput(
            minimum_supported_distance_m=0.0,
            closure_kind=(
                ShortRangeClosureKind.EXTERNAL_VALIDATED_HYDROGEL_MECHANICS_CLOSURE
            ),
            closure_description=(
                "TEST_ONLY independently supplied zero-distance closure contract"
            ),
            validation_method=(
                "TEST_ONLY force and energy overlap comparison at positive distance"
            ),
            force_match_tolerance_newton=1.0e-18,
            energy_match_tolerance_joule=1.0e-24,
            provenance=_test_only_provenance("SHORT_RANGE"),
        )
    else:
        short_range = ShortRangeClosureInput(
            minimum_supported_distance_m=minimum_supported_distance_m,
            closure_kind=ShortRangeClosureKind.NONE,
            closure_description=None,
            validation_method=None,
            force_match_tolerance_newton=None,
            energy_match_tolerance_joule=None,
            provenance=_test_only_provenance("SHORT_RANGE"),
        )
    return ContactProblemInput(
        purpose=ContactInputPurpose.TEST_ONLY_CONTRACT,
        hydrogel_parameters=parameters,
        hydrogel_parameters_provenance=_test_only_provenance("HYDROGEL"),
        geometry=TwoSphereGeometry(
            first_radius_m=0.4e-6,
            second_radius_m=0.45e-6,
            calibration_status=TEST_ONLY_NOT_CALIBRATED,
        ),
        geometry_provenance=_test_only_provenance("GEOMETRY"),
        mechanical_boundary_conditions=MechanicalBoundaryConditions(
            center_separation_control=(
                "TEST_ONLY prescribed center separation for every scan point"
            ),
            rigid_body_constraint=(
                "TEST_ONLY remove rigid translation while preserving deformation"
            ),
            noncontact_surface_traction=(
                "TEST_ONLY traction-free gel surface outside the contact region"
            ),
            loading_protocol=(
                "TEST_ONLY independent quasi-static equilibrium at each distance"
            ),
            provenance=_test_only_provenance("MECHANICAL_BOUNDARY"),
        ),
        solvent_bath_boundary_conditions=SolventBathBoundaryConditions(
            bath_chemical_potential_over_kbt=(
                parameters.bath_chemical_potential_over_kbt
            ),
            exposed_surface_exchange_condition=(
                "TEST_ONLY prescribed bath chemical potential on exposed surfaces"
            ),
            contact_surface_transport_condition=(
                "TEST_ONLY zero normal solvent flux across the contact interface"
            ),
            initial_solvent_state=(
                "TEST_ONLY initial-swelling equilibrium from Model II parameters"
            ),
            provenance=_test_only_provenance("SOLVENT_BATH"),
        ),
        contact_law=ContactLawAndTolerances(
            normal_contact_model=NormalContactLaw.UNILATERAL_IMPENETRABILITY,
            contact_enforcement_method=(
                "TEST_ONLY mixed complementarity contact discretization"
            ),
            frictionless=True,
            adhesive=False,
            normal_gap_tolerance_m=1.0e-10,
            force_balance_tolerance_newton=1.0e-18,
            provenance=_test_only_provenance("CONTACT_LAW"),
        ),
        solver=ContactSolverSpecification(
            solver_name="TEST_ONLY_CONTACT_SOLVER_NAME",
            solver_version="TEST_ONLY_SOLVER_VERSION",
            implementation_id="TEST_ONLY_SOLVER_IMPLEMENTATION",
            configuration_id="TEST_ONLY_SOLVER_CONFIGURATION",
            nonlinear_algorithm="TEST_ONLY_DAMPED_NEWTON_ALGORITHM",
            linear_solver="TEST_ONLY_DIRECT_LINEAR_SOLVER",
            provenance=_test_only_provenance("SOLVER"),
        ),
        mesh_convergence=MeshConvergencePlan(
            discretization_method="TEST_ONLY_AXISYMMETRIC_FINITE_ELEMENTS",
            element_family="TEST_ONLY_MIXED_DISPLACEMENT_SOLVENT_ELEMENTS",
            characteristic_lengths_m=(0.2e-6, 0.1e-6, 0.05e-6),
            nonlinear_residual_tolerance_dimensionless=1.0e-8,
            relative_force_convergence_tolerance_dimensionless=1.0e-3,
            relative_energy_convergence_tolerance_dimensionless=1.0e-3,
            relative_stress_convergence_tolerance_dimensionless=2.0e-3,
            maximum_nonlinear_iterations=50,
            provenance=_test_only_provenance("MESH_CONVERGENCE"),
        ),
        distance_scan=DistanceScanPlan(
            center_distances_m=(0.7e-6, 0.8e-6, 0.9e-6, 1.2e-6),
            reference_distance_m=1.2e-6,
            reference_force_tolerance_newton=1.0e-18,
            provenance=_test_only_provenance("DISTANCE_SCAN"),
        ),
        reaction_force=ReactionForceConvention(
            integration_boundary=(
                "TEST_ONLY complete exterior boundary of the first gel"
            ),
            traction_quantity="TEST_ONLY first Piola traction in reference area",
            pair_axis_definition=PairAxisDefinition.SECOND_TO_FIRST_CENTER,
            positive_sign_convention=(
                ReactionForceSignConvention.POSITIVE_REPULSION
            ),
            total_free_energy_reference=(
                TotalFreeEnergyReference.ZERO_AT_FINAL_REFERENCE_DISTANCE
            ),
            integration_tolerance_newton=1.0e-18,
            provenance=_test_only_provenance("REACTION_FORCE"),
        ),
        time_scale=assess_time_scale_separation(
            tau_gel_s=1.0e-3,
            tau_swarm_s=1.0,
            required_max_ratio=1.0e-2,
        ),
        time_scale_provenance=_test_only_provenance("TIME_SCALE"),
        mean_field_scaling=MeanFieldScalingInput(
            density_convention=DensityConvention.PROBABILITY,
            scaling_mode=MeanFieldScalingMode.KAC_FROM_SINGLE_PAIR_POPULATION,
            population_count=4,
            source_population_count=4,
            areal_number_density_per_m2=1.0e12,
            representative_area_m2=4.0e-12,
            scaling_definition=(
                "TEST_ONLY F_Kac equals population count times raw single-pair force"
            ),
            provenance=_test_only_provenance("MEAN_FIELD_SCALING"),
        ),
        short_range=short_range,
    )


class ContactProblemInputContractTests(unittest.TestCase):
    def test_completely_missing_problem_is_blocked_and_reports_all_inputs(self) -> None:
        problem = ContactProblemInput(
            purpose=ContactInputPurpose.TEST_ONLY_CONTRACT
        )
        readiness = problem.evaluate_readiness()
        self.assertEqual(readiness.solver_input_status, SolverInputStatus.BLOCKED)
        self.assertEqual(
            readiness.contact_results_status, ContactResultsStatus.NOT_GENERATED
        )
        self.assertEqual(
            readiness.short_range_status, ShortRangeInputStatus.BLOCKED
        )
        self.assertFalse(readiness.effective_pair_potential_validated)
        self.assertEqual(
            set(readiness.missing_inputs),
            set(ContactProblemInput.REQUIRED_INPUT_NAMES),
        )
        reports = {
            item.requirement_id: item for item in readiness.required_inputs
        }
        self.assertIn(
            "hydrogel_parameters.network_density_times_solvent_volume",
            reports["material.hydrogel_parameters"].input_field_paths,
        )
        self.assertIn(
            "mean_field_scaling.provenance.verification_record_id",
            reports[
                "mean_field.population_or_kac_scaling_source"
            ].input_field_paths,
        )
        self.assertEqual(
            tuple(ContactProblemInput.required_input_field_paths()),
            ContactProblemInput.REQUIRED_INPUT_NAMES,
        )

    def test_partial_problem_reports_only_the_removed_group(self) -> None:
        partial = replace(_test_only_complete_problem(), solver=None)
        readiness = partial.evaluate_readiness()
        self.assertEqual(
            readiness.missing_inputs,
            ("solver.identity_and_configuration",),
        )
        self.assertEqual(readiness.unverified_inputs, ())
        payload = partial.to_input_dict()
        self.assertIsNone(payload["solver"])
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(
            set(payload),
            {
                "purpose",
                "hydrogel_parameters",
                "hydrogel_parameters_provenance",
                "geometry",
                "geometry_provenance",
                "mechanical_boundary_conditions",
                "solvent_bath_boundary_conditions",
                "contact_law",
                "solver",
                "mesh_convergence",
                "distance_scan",
                "reaction_force",
                "time_scale",
                "time_scale_provenance",
                "mean_field_scaling",
                "short_range",
                "schema_name",
                "schema_version",
                "readiness",
            },
        )
        self.assertEqual(
            set(payload["readiness"]["required_inputs"][0]),
            {"requirement_id", "input_field_paths", "status", "reason"},
        )
        self.assertIn(
            "physical_continuum_input_ready", payload["readiness"]
        )
        json.dumps(payload, allow_nan=False)

    def test_vague_text_and_test_only_physical_claim_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "vague placeholder"):
            InputProvenance(
                source_id="TBD",
                source_description="specific laboratory record",
                verification_status=InputVerificationStatus.UNVERIFIED,
                verification_record_id=None,
            )
        with self.assertRaisesRegex(ValueError, "vague placeholder"):
            replace(
                _test_only_complete_problem().mechanical_boundary_conditions,
                loading_protocol="provided later",
            )
        with self.assertRaisesRegex(ValueError, "cannot claim VERIFIED"):
            InputProvenance(
                source_id="CALIBRATED_TEST_ONLY_SOURCE",
                source_description="TEST_ONLY source marked as calibrated",
                verification_status=InputVerificationStatus.VERIFIED,
                verification_record_id="CALIBRATED_TEST_ONLY_RECORD",
            )
        with self.assertRaisesRegex(ValueError, "UNVERIFIED.*VERIFIED"):
            InputProvenance(
                source_id="LAB_CONTACT_SOURCE_2026_08",
                source_description="pending verification for laboratory sweep",
                verification_status=InputVerificationStatus.VERIFIED,
                verification_record_id="LAB_CONTACT_REVIEW_2026_08",
            )
        pending_calibration = ContactProblemInput(
            purpose=ContactInputPurpose.PHYSICAL,
            hydrogel_parameters=replace(
                _test_only_parameters(),
                calibration_status="PENDING_CALIBRATION_BATCH_2026_08",
            ),
            hydrogel_parameters_provenance=InputProvenance(
                source_id="LAB_BATCH_2026_08",
                source_description="laboratory material batch record",
                verification_status=InputVerificationStatus.VERIFIED,
                verification_record_id="LAB_BATCH_REVIEW_2026_08",
            ),
        ).evaluate_readiness()
        self.assertEqual(
            pending_calibration.solver_input_status, SolverInputStatus.BLOCKED
        )
        self.assertIn(
            "material.hydrogel_parameters",
            pending_calibration.unverified_inputs,
        )
        with self.assertRaisesRegex(ValueError, "Hertz.*Hydrogel readiness"):
            ContactProblemInput(
                purpose=ContactInputPurpose.PHYSICAL,
                short_range=ShortRangeClosureInput(
                    minimum_supported_distance_m=0.0,
                    closure_kind=(
                        ShortRangeClosureKind.EXTERNAL_VALIDATED_HYDROGEL_MECHANICS_CLOSURE
                    ),
                    closure_description="Hertz continuation to zero separation",
                    validation_method=(
                        "comparison against the positive-distance contact record"
                    ),
                    force_match_tolerance_newton=0.0,
                    energy_match_tolerance_joule=0.0,
                    provenance=InputProvenance(
                        source_id="CLOSURE_RECORD_2026_08",
                        source_description="hydrogel short-range closure record",
                        verification_status=InputVerificationStatus.VERIFIED,
                        verification_record_id="CLOSURE_REVIEW_2026_08",
                    ),
                ),
            )
        with self.assertRaisesRegex(ValueError, "TEST_ONLY values"):
            ContactProblemInput(
                purpose=ContactInputPurpose.PHYSICAL,
                mechanical_boundary_conditions=MechanicalBoundaryConditions(
                    center_separation_control=(
                        "TEST_ONLY prescribed center separation"
                    ),
                    rigid_body_constraint="remove rigid body translation",
                    noncontact_surface_traction="zero exterior traction",
                    loading_protocol="quasi-static distance continuation",
                    provenance=InputProvenance(
                        source_id="MECHANICAL_BC_RECORD_2026_08",
                        source_description="mechanical boundary-condition record",
                        verification_status=InputVerificationStatus.VERIFIED,
                        verification_record_id="MECHANICAL_BC_REVIEW_2026_08",
                    ),
                ),
            )
        with self.assertRaisesRegex(ValueError, "TEST_ONLY calibration"):
            replace(
                _test_only_complete_problem(),
                purpose=ContactInputPurpose.PHYSICAL,
            )
        with self.assertRaisesRegex(ValueError, "vague placeholder"):
            replace(
                _test_only_complete_problem(),
                hydrogel_parameters=replace(
                    _test_only_parameters(), calibration_status="calibrated"
                ),
            )

    def test_time_scale_is_recomputed_and_unverified_or_violated_blocks(self) -> None:
        with self.assertRaisesRegex(TypeError, "time_scale.tau_gel_s"):
            replace(
                _test_only_complete_problem(),
                time_scale=TimeScaleAssessment(
                    status=TimeScaleStatus.SATISFIED,
                    tau_gel_s=True,
                    tau_swarm_s=1.0,
                    required_max_ratio=0.1,
                    ratio=0.01,
                    reason="TEST_ONLY invalid bool time scale",
                ),
            )
        forged = TimeScaleAssessment(
            status=TimeScaleStatus.SATISFIED,
            tau_gel_s=1.0,
            tau_swarm_s=1.0,
            required_max_ratio=0.1,
            ratio=0.01,
            reason="forged satisfied status for contract test",
        )
        violated = replace(_test_only_complete_problem(), time_scale=forged)
        self.assertEqual(violated.time_scale.status, TimeScaleStatus.VIOLATED)
        readiness = violated.evaluate_readiness()
        self.assertEqual(readiness.solver_input_status, SolverInputStatus.BLOCKED)
        self.assertIn(
            "timescale.separation_and_source", readiness.unverified_inputs
        )

        unverified = replace(
            _test_only_complete_problem(),
            time_scale=assess_time_scale_separation(
                tau_gel_s=None,
                tau_swarm_s=None,
                required_max_ratio=None,
            ),
        )
        self.assertEqual(
            unverified.evaluate_readiness().solver_input_status,
            SolverInputStatus.BLOCKED,
        )

    def test_complete_input_is_solver_ready_but_has_no_results(self) -> None:
        problem = _test_only_complete_problem()
        readiness = problem.evaluate_readiness()
        self.assertEqual(
            readiness.solver_input_status,
            SolverInputStatus.READY_FOR_CONTACT_SOLVER,
        )
        self.assertEqual(
            readiness.contact_results_status, ContactResultsStatus.NOT_GENERATED
        )
        self.assertFalse(readiness.physical_inputs_verified)
        self.assertFalse(readiness.effective_pair_potential_validated)
        self.assertFalse(
            any(
                item.status
                in (RequiredInputStatus.MISSING, RequiredInputStatus.UNVERIFIED)
                for item in readiness.required_inputs
            )
        )
        with self.assertRaisesRegex(NotImplementedError, "generated no contact samples"):
            HydrogelContactFEMNotAvailable().solve_contact_problem(problem)

        with self.assertRaisesRegex(ValueError, "every contact requirement"):
            ContactProblemReadiness(
                solver_input_status=SolverInputStatus.READY_FOR_CONTACT_SOLVER,
                short_range_status=ShortRangeInputStatus.CONTINUUM_INPUT_READY,
                required_inputs=(),
                blocking_reasons=(),
                physical_inputs_verified=True,
            )

    def test_short_range_states_are_explicit_and_do_not_validate_weff(self) -> None:
        particle_only = _test_only_complete_problem(
            minimum_supported_distance_m=0.7e-6
        ).evaluate_readiness()
        self.assertEqual(
            particle_only.short_range_status, ShortRangeInputStatus.PARTICLE_ONLY
        )

        continuum_input = _test_only_complete_problem(
            with_zero_closure=True
        ).evaluate_readiness()
        self.assertEqual(
            continuum_input.short_range_status,
            ShortRangeInputStatus.BLOCKED,
        )
        self.assertEqual(
            continuum_input.contact_results_status, ContactResultsStatus.NOT_GENERATED
        )
        self.assertFalse(continuum_input.effective_pair_potential_validated)
        self.assertFalse(continuum_input.physical_inputs_verified)
        self.assertFalse(continuum_input.physical_continuum_input_ready)

        missing_closure = replace(
            _test_only_complete_problem(),
            short_range=ShortRangeClosureInput(
                minimum_supported_distance_m=0.0,
                closure_kind=ShortRangeClosureKind.NONE,
                closure_description=None,
                validation_method=None,
                force_match_tolerance_newton=None,
                energy_match_tolerance_joule=None,
                provenance=_test_only_provenance("MISSING_ZERO_CLOSURE"),
            ),
        ).evaluate_readiness()
        self.assertEqual(
            missing_closure.solver_input_status, SolverInputStatus.BLOCKED
        )
        self.assertEqual(
            missing_closure.short_range_status, ShortRangeInputStatus.BLOCKED
        )

    def test_population_and_concentration_sources_must_agree(self) -> None:
        base = _test_only_complete_problem().mean_field_scaling
        with self.assertRaisesRegex(ValueError, "scaling source population"):
            replace(base, source_population_count=5)
        with self.assertRaisesRegex(ValueError, "number density times"):
            replace(base, representative_area_m2=5.0e-12)
        with self.assertRaisesRegex(TypeError, "positive integer"):
            replace(base, population_count=True)
        with self.assertRaisesRegex(ValueError, "scaling mode"):
            replace(
                base,
                scaling_mode=MeanFieldScalingMode.RAW_SINGLE_PAIR_NUMBER_DENSITY,
            )

    def test_numeric_domains_and_immutability_are_strict(self) -> None:
        with self.assertRaisesRegex(TypeError, "real scalar"):
            replace(
                _test_only_complete_problem(),
                hydrogel_parameters=replace(
                    _test_only_parameters(), flory_huggins_chi="0.31"
                ),
            )
        with self.assertRaisesRegex(TypeError, "real scalar"):
            replace(
                _test_only_complete_problem(),
                geometry=replace(
                    _test_only_complete_problem().geometry,
                    first_radius_m=True,
                ),
            )
        with self.assertRaisesRegex(TypeError, "positive_sign_convention"):
            replace(
                _test_only_complete_problem().reaction_force,
                positive_sign_convention=(
                    "positive radial force attracts the two sphere centers"
                ),
            )
        distances = [0.7e-6, 0.8e-6, 0.9e-6, 1.2e-6]
        plan = DistanceScanPlan(
            center_distances_m=distances,
            reference_distance_m=1.2e-6,
            reference_force_tolerance_newton=1.0e-18,
            provenance=_test_only_provenance("IMMUTABLE_SCAN"),
        )
        distances[0] = 0.1e-6
        self.assertEqual(plan.center_distances_m[0], 0.7e-6)
        with self.assertRaises(FrozenInstanceError):
            plan.reference_distance_m = 2.0e-6
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            replace(plan, center_distances_m=(0.7e-6, 0.9e-6, 0.8e-6, 1.2e-6))
        with self.assertRaisesRegex(ValueError, "must be finite"):
            replace(plan, reference_force_tolerance_newton=np.nan)
        base = _test_only_complete_problem()
        with self.assertRaisesRegex(ValueError, "must equal the first scan distance"):
            replace(
                base,
                short_range=replace(
                    base.short_range, minimum_supported_distance_m=0.6e-6
                ),
            )

    def test_blocked_problem_never_reaches_placeholder_solver(self) -> None:
        problem = ContactProblemInput(
            purpose=ContactInputPurpose.TEST_ONLY_CONTRACT
        )
        with self.assertRaisesRegex(ValueError, "contact problem input is BLOCKED"):
            HydrogelContactFEMNotAvailable().solve_contact_problem(problem)


if __name__ == "__main__":
    unittest.main()
