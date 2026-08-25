"""Export a fail-closed, conditional MG10F Stokes--Einstein derivation.

This Experiment Lab program records only the nominal 293.15 K transport
calculation that follows from the supplied MG10F registry.  It is deliberately
not a Hydrogel calibration, a contact/FEM computation, a pair-potential
construction, a McKean--Vlasov solve, a controller invocation, or training.
All nominal inputs remain source-location-unverified, so a successful program
run means that the arithmetic and provenance export completed—not that the
material model is physically ready.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import sys
from typing import Any

from mechanistic_mv.mechanics.mg10f_parameters import (
    BOLTZMANN_CONSTANT_J_PER_K,
    MG10FParameterRegistry,
    PhysicalParameter,
    VerificationStatus,
)

try:
    from ._phase6_common import provenance, read_json_object, write_json
except ImportError:  # direct ``python scripts/...`` execution
    from _phase6_common import provenance, read_json_object, write_json  # type: ignore[no-redef]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PROJECT_ROOT / "configs" / "physics_mg10f.json"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "validation" / "parameters"

SCHEMA_VERSION = 2
MODEL_STATUS = "TEST_ONLY_NOT_CALIBRATED"
WORKFLOW_STATUS = "CONDITIONAL_TRANSPORT_DERIVATION_COMPLETED"
PHYSICAL_STATUS = "NOMINAL_MG10F_SOURCE_LOCATIONS_UNVERIFIED"
CALIBRATION_STATUS = "BLOCKED_NETWORK_DENSITY_UNCALIBRATED"
TRANSPORT_STATUS = "DERIVED_FROM_NOMINAL_INPUTS_CONDITIONAL"
BLOCKED_WORKFLOW_STATUS = "BLOCKED"

DERIVED_TRANSPORT_FILENAME = "derived_transport.json"

_TRANSPORT_INPUT_CONTRACT: dict[str, str] = {
    "temperature_20c": "K",
    "hydrodynamic_radius_20c": "m",
    "water_viscosity_20c": "Pa s",
}
_EXPECTED_BLOCKED_CALIBRATION_PARAMETERS = (
    "network_density_times_solvent_volume",
    "initial_polymer_volume_fraction",
    "delta_chemical_potential_over_kbt",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help="strict JSON MG10F registry (default: configs/physics_mg10f.json)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="directory for provenance and conditional-transport artifacts",
    )
    return parser


def _downstream_statuses() -> dict[str, str]:
    """Declare the intentionally unrun downstream workflows."""

    return {
        "hydrogel": "BLOCKED",
        "contact_fem": "BLOCKED",
        "F_pair": "BLOCKED",
        "W_eff": "BLOCKED",
        "mckean_vlasov": "BLOCKED",
        "controller": "NOT_IN_SCOPE",
    }


def _finite_positive_numeric_parameter(
    registry: MG10FParameterRegistry, name: str, expected_unit: str
) -> PhysicalParameter:
    """Read one required SI transport input without supplying a fallback."""

    parameter = registry.parameter(name)
    if parameter.unit != expected_unit:
        raise ValueError(
            f"MG10F parameter {name!r} must use unit {expected_unit!r}, "
            f"not {parameter.unit!r}"
        )
    value = parameter.value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"MG10F parameter {name!r} must be a numeric input")
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value <= 0.0:
        raise ValueError(f"MG10F parameter {name!r} must be finite and positive")
    return parameter


def _validate_conditional_nominal_registry(
    registry: MG10FParameterRegistry,
) -> dict[str, PhysicalParameter]:
    """Gate exports so they cannot silently become a physical calibration.

    The reporting contract is specifically for the user-supplied, nominal,
    source-location-unverified MG10F registry.  A registry that claims a
    different readiness state must receive a separately reviewed workflow;
    this script must not relabel it as a calibrated material model.
    """

    if registry.registry_status != MODEL_STATUS:
        raise ValueError(
            "registry_status must be TEST_ONLY_NOT_CALIBRATED for the "
            "conditional MG10F transport workflow"
        )
    if registry.is_material_ready:
        raise ValueError(
            "conditional transport workflow requires unresolved calibration "
            "parameters; material-ready registries are outside this scope"
        )
    if registry.blocked_calibration_parameters != _EXPECTED_BLOCKED_CALIBRATION_PARAMETERS:
        raise ValueError(
            "conditional transport workflow requires the explicit blocked "
            "MG10F calibration set"
        )
    if registry.is_verified_material_ready:
        raise ValueError(
            "source-verified material registries are outside this nominal "
            "unverified transport workflow"
        )
    usable_parameter_names = {
        parameter.name for parameter in registry.parameters if parameter.is_usable
    }
    if set(registry.unverified_usable_parameters) != usable_parameter_names:
        raise ValueError(
            "every usable nominal parameter must remain source-location "
            "unverified in this workflow"
        )

    inputs = {
        name: _finite_positive_numeric_parameter(registry, name, unit)
        for name, unit in _TRANSPORT_INPUT_CONTRACT.items()
    }
    radius_uncertainty = inputs["hydrodynamic_radius_20c"].uncertainty
    if radius_uncertainty is None or not math.isfinite(radius_uncertainty):
        raise ValueError(
            "hydrodynamic_radius_20c must provide a finite uncertainty for "
            "the declared partial uncertainty calculation"
        )
    if radius_uncertainty < 0.0:
        raise ValueError("hydrodynamic_radius_20c uncertainty must be non-negative")
    return inputs


def _conditional_transport_records(
    inputs: dict[str, PhysicalParameter],
) -> list[dict[str, Any]]:
    """Calculate nominal transport values and the strictly partial R_h error.

    The first-order propagation keeps only the reported hydrodynamic-radius
    uncertainty.  Temperature, viscosity, and model-form effects are
    intentionally excluded because the registry does
    not quantify them; this is why the result is not a complete uncertainty
    budget or a physical calibration.
    """

    temperature = float(inputs["temperature_20c"].value)
    radius = float(inputs["hydrodynamic_radius_20c"].value)
    viscosity = float(inputs["water_viscosity_20c"].value)
    radius_uncertainty = inputs["hydrodynamic_radius_20c"].uncertainty
    assert radius_uncertainty is not None  # validated by the caller

    gamma = 6.0 * math.pi * viscosity * radius
    mobility = 1.0 / gamma
    thermal_energy = BOLTZMANN_CONSTANT_J_PER_K * temperature
    diffusion = mobility * thermal_energy
    relative_radius_uncertainty = radius_uncertainty / radius

    common = {
        "verification_status": (
            VerificationStatus.CONDITIONAL_NOMINAL_DERIVED_FROM_UNVERIFIED_INPUTS.value
        ),
        "provenance_type": "DERIVED",
        "input_status": PHYSICAL_STATUS,
        "calibration_status": CALIBRATION_STATUS,
    }
    return [
        {
            "name": "stokes_drag_coefficient",
            "symbol": "gamma",
            "value": gamma,
            "unit": "kg/s",
            "formula": "gamma = 6*pi*eta*R_h",
            "uncertainty": abs(gamma) * relative_radius_uncertainty,
            "uncertainty_unit": "kg/s",
            "uncertainty_status": "PARTIAL_PROPAGATED_FROM_R_H_ONLY",
            **common,
        },
        {
            "name": "mobility",
            "symbol": "M",
            "value": mobility,
            "unit": "m/(N s)",
            "formula": "M = 1/gamma",
            "uncertainty": abs(mobility) * relative_radius_uncertainty,
            "uncertainty_unit": "m/(N s)",
            "uncertainty_status": "PARTIAL_PROPAGATED_FROM_R_H_ONLY",
            **common,
        },
        {
            "name": "thermal_energy",
            "symbol": "k_B*T",
            "value": thermal_energy,
            "unit": "J",
            "formula": "k_B*T, with exact SI k_B",
            "uncertainty": None,
            "uncertainty_status": "NOT_REPORTED_BY_PARTIAL_RH_ONLY_SCOPE",
            **common,
        },
        {
            "name": "diffusion",
            "symbol": "D",
            "value": diffusion,
            "unit": "m^2/s",
            "formula": "D = M*k_B*T",
            "uncertainty": abs(diffusion) * relative_radius_uncertainty,
            "uncertainty_unit": "m^2/s",
            "uncertainty_status": "PARTIAL_PROPAGATED_FROM_R_H_ONLY",
            **common,
        },
    ]


def _input_summary(inputs: dict[str, PhysicalParameter]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "value": parameter.value,
            "unit": parameter.unit,
            "uncertainty": parameter.uncertainty,
            "verification_status": parameter.verification_status.value,
            "source": parameter.source,
            "source_location": parameter.source_location,
        }
        for name, parameter in inputs.items()
    }


def _registry_file_identity(path: Path) -> dict[str, object]:
    """Identify the exact JSON that supplied this conditional report."""

    resolved = path.resolve()
    canonical = DEFAULT_REGISTRY.resolve()
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "canonical_repository_config_path": str(canonical),
        "uses_canonical_repository_config": resolved == canonical,
    }


def _derived_transport_payload(
    registry: MG10FParameterRegistry,
    inputs: dict[str, PhysicalParameter],
    registry_path: Path,
    command: list[str],
) -> dict[str, Any]:
    return {
        "schema_name": "mechanistic_mv.mg10f_transport_derivation",
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "MG10F_CONDITIONAL_TRANSPORT_DERIVATION",
        "model_status": MODEL_STATUS,
        "workflow_status": WORKFLOW_STATUS,
        "physical_status": PHYSICAL_STATUS,
        "calibration_status": CALIBRATION_STATUS,
        "transport_status": TRANSPORT_STATUS,
        "workflow_completed": True,
        "physical_model_ready": False,
        "registry": {
            "registry_id": registry.registry_id,
            "registry_status": registry.registry_status,
            "reference_document": registry.reference_document,
            "file_identity": _registry_file_identity(registry_path),
        },
        "inputs": _input_summary(inputs),
        "derived_transport": _conditional_transport_records(inputs),
        "uncertainty_assessment": {
            "status": "PARTIAL_R_H_ONLY",
            "propagated_input": "hydrodynamic_radius_20c",
            "included_uncertainties": ["hydrodynamic_radius_20c"],
            "excluded_uncertainties": [
                "temperature_20c",
                "water_viscosity_20c",
                "Stokes_model_form_and_soft_particle_effects",
            ],
            "drag_formula": "sigma_gamma = |gamma| * sigma_R_h / R_h",
            "mobility_formula": "sigma_M = |M| * sigma_R_h / R_h",
            "diffusion_formula": "sigma_D = |D| * sigma_R_h / R_h",
            "interpretation": (
                "Only the reported R_h uncertainty is propagated. This is not a "
                "complete uncertainty budget and must not be treated as calibration."
            ),
        },
        "source_verification": {
            "status": "ALL_USABLE_REGISTRY_INPUT_SOURCES_UNVERIFIED",
            "verified_usable_parameters": [],
            "unverified_usable_parameters": list(
                registry.unverified_usable_parameters
            ),
            "interpretation": (
                "Every non-null registry record lacks a verified primary source "
                "location. Conditional transport arithmetic does not verify it."
            ),
        },
        "blocked_calibration_parameters": list(registry.blocked_calibration_parameters),
        "downstream_status": _downstream_statuses(),
        "outputs_written": [DERIVED_TRANSPORT_FILENAME],
        "non_claims": [
            "No Hydrogel constitutive parameter set was constructed.",
            "No contact or FEM calculation was called.",
            "No F_pair or W_eff was constructed.",
            "No McKean--Vlasov solve, controller invocation, or training was run.",
            "The hydrodynamic radius is not a reference contact radius.",
        ],
        "provenance": provenance(command),
    }


def _failure_payload(error: Exception, command: list[str]) -> dict[str, Any]:
    return {
        "schema_name": "mechanistic_mv.mg10f_transport_derivation",
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "MG10F_CONDITIONAL_TRANSPORT_DERIVATION",
        "model_status": MODEL_STATUS,
        "workflow_status": BLOCKED_WORKFLOW_STATUS,
        "physical_status": "BLOCKED",
        "calibration_status": "BLOCKED",
        "transport_status": "BLOCKED",
        "workflow_completed": False,
        "physical_model_ready": False,
        "reason": f"{type(error).__name__}: {error}",
        "outputs_written": [DERIVED_TRANSPORT_FILENAME],
        "downstream_status": _downstream_statuses(),
        "provenance": provenance(command),
    }


def _load_and_validate_registry(path: Path) -> tuple[MG10FParameterRegistry, dict[str, PhysicalParameter]]:
    if path.suffix.lower() != ".json":
        raise ValueError("MG10F registry must use a .json suffix")
    # read_json_object rejects duplicate keys, NaN and Infinity before the
    # immutable registry model validates its schema and semantic constraints.
    raw_registry = read_json_object(path)
    registry = MG10FParameterRegistry.from_jsonable(raw_registry)
    return registry, _validate_conditional_nominal_registry(registry)


def run(registry_path: Path, output_directory: Path, command: list[str]) -> None:
    """Write only conditional transport evidence after the full input gate passes."""

    registry, inputs = _load_and_validate_registry(registry_path)
    write_json(
        output_directory / DERIVED_TRANSPORT_FILENAME,
        _derived_transport_payload(registry, inputs, registry_path, command),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = list(sys.argv if argv is None else ["run_mg10f_transport_derivation", *argv])
    try:
        run(args.registry, args.output_directory, command)
    except (OSError, TypeError, ValueError, KeyError) as error:
        try:
            write_json(
                args.output_directory / DERIVED_TRANSPORT_FILENAME,
                _failure_payload(error, command),
            )
        except OSError as report_error:
            print(
                "MG10F conditional transport derivation failed and could not "
                f"write its blocked report: {report_error}",
                file=sys.stderr,
            )
        else:
            print(
                "MG10F conditional transport derivation is BLOCKED: "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
            )
        return 2
    print(
        "MG10F conditional transport derivation completed; status remains "
        "unverified and blocked downstream.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
