"""Fail-closed, provenance-aware nominal MG10F parameter registry.

The registry records a user-supplied *nominal* MG10F PNIPAM/water parameter
set.  It deliberately does not configure :class:`HydrogelParameters`, contact
mechanics, or the McKean--Vlasov solver: the calibrated inputs and primary
source locations required for those steps are still absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Mapping, NoReturn, TypeAlias


BOLTZMANN_CONSTANT_J_PER_K = 1.380649e-23

# This module represents one deliberately nominal, user-supplied MG10F snapshot.
# A hand edit to its JSON must never turn that snapshot into a calibrated material
# model.  Promoting a material to a verified record needs a separately reviewed
# registry/schema, primary-source locators, and a calibration workflow.
NOMINAL_MG10F_REGISTRY_STATUS = "TEST_ONLY_NOT_CALIBRATED"
REQUIRED_CALIBRATION_PARAMETERS = (
    "network_density_times_solvent_volume",
    "initial_polymer_volume_fraction",
    "delta_chemical_potential_over_kbt",
)
REQUIRED_CALIBRATION_CONTRACT = {
    "network_density_times_solvent_volume": (
        "dimensionless",
        "REQUIRED_HYDROGEL_CALIBRATION_INPUT",
    ),
    "initial_polymer_volume_fraction": (
        "dimensionless",
        "REQUIRED_HYDROGEL_CALIBRATION_INPUT",
    ),
    "delta_chemical_potential_over_kbt": (
        "dimensionless",
        "REQUIRED_HYDROGEL_CALIBRATION_INPUT",
    ),
}
TRANSPORT_INPUT_UNITS = {
    "temperature_20c": "K",
    "hydrodynamic_radius_20c": "m",
    "water_viscosity_20c": "Pa s",
    "water_molecular_volume": "m^3/molecule",
}
CHI_COEFFICIENT_UNITS = {
    "chi_A0": "dimensionless",
    "chi_B0_per_kelvin": "K^-1",
    "chi_A1": "dimensionless",
    "chi_B1_per_kelvin": "K^-1",
}

JsonScalar: TypeAlias = str | int | float | bool | None


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate member names."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is not permitted: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(token: str) -> NoReturn:
    """Reject JSON extensions such as ``NaN`` and ``Infinity`` fail-closed."""

    raise ValueError(f"non-finite JSON constant is not permitted: {token}")


class ProvenanceType(str, Enum):
    """How a parameter entered the registry."""

    NOMINAL_SYSTEM_SPECIFICATION = "NOMINAL_SYSTEM_SPECIFICATION"
    MEASURED = "MEASURED"
    LITERATURE_CONSTITUTIVE = "LITERATURE_CONSTITUTIVE"
    CALIBRATED = "CALIBRATED"
    DERIVED = "DERIVED"


class VerificationStatus(str, Enum):
    """Verification state; statuses never promote themselves to ``VERIFIED``."""

    SOURCE_LOCATION_NEEDS_VERIFICATION = "SOURCE_LOCATION_NEEDS_VERIFICATION"
    UNVERIFIED = "UNVERIFIED"
    CONDITIONAL_NOMINAL_DERIVED_FROM_UNVERIFIED_INPUTS = (
        "CONDITIONAL_NOMINAL_DERIVED_FROM_UNVERIFIED_INPUTS"
    )
    BLOCKED = "BLOCKED"
    VERIFIED = "VERIFIED"


class MaterialNotReadyError(RuntimeError):
    """Raised when a blocked registry is requested as a material model."""


@dataclass(frozen=True, slots=True)
class PhysicalParameter:
    """Immutable provenance record for one parameter or material descriptor.

    ``value`` is intentionally JSON-scalar-only.  Missing calibrated values
    are represented by ``None`` and must use ``BLOCKED`` verification status.
    """

    name: str
    symbol: str
    value: JsonScalar
    unit: str
    uncertainty: float | None
    provenance_type: ProvenanceType
    verification_status: VerificationStatus
    source: str
    source_location: str | None
    method: str
    notes: str
    observable_role: str

    def __post_init__(self) -> None:
        for field_name in (
            "name",
            "symbol",
            "unit",
            "source",
            "method",
            "notes",
            "observable_role",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.provenance_type, ProvenanceType):
            raise TypeError("provenance_type must be a ProvenanceType")
        if not isinstance(self.verification_status, VerificationStatus):
            raise TypeError("verification_status must be a VerificationStatus")
        if self.source_location is not None and (
            not isinstance(self.source_location, str) or not self.source_location.strip()
        ):
            raise ValueError("source_location must be a non-empty string or None")
        if self.value is not None and not isinstance(self.value, (str, int, float, bool)):
            raise TypeError("value must be a JSON scalar or None")
        if isinstance(self.value, (int, float)) and not isinstance(self.value, bool):
            if not math.isfinite(float(self.value)):
                raise ValueError("numeric parameter values must be finite")
        if self.uncertainty is not None:
            if isinstance(self.uncertainty, bool) or not math.isfinite(self.uncertainty):
                raise ValueError("uncertainty must be finite or None")
            if self.uncertainty < 0.0:
                raise ValueError("uncertainty must be non-negative")
        if self.value is None and self.verification_status is not VerificationStatus.BLOCKED:
            raise ValueError("a null parameter value must be BLOCKED")
        if self.value is not None and self.verification_status is VerificationStatus.BLOCKED:
            raise ValueError("a BLOCKED parameter must have a null value")
        if (
            self.verification_status is VerificationStatus.VERIFIED
            and self.source_location is None
        ):
            raise ValueError("a VERIFIED parameter requires a source_location")

    @property
    def is_usable(self) -> bool:
        """Whether this record contains a non-null nominal value."""

        return self.value is not None

    @property
    def has_verifiable_source_location(self) -> bool:
        """Whether this value has passed the registry's source gate."""

        return (
            self.is_usable
            and self.verification_status is VerificationStatus.VERIFIED
            and self.source_location is not None
        )

    def as_jsonable(self) -> dict[str, JsonScalar]:
        """Return a standard-library JSON-compatible record."""

        return {
            "name": self.name,
            "symbol": self.symbol,
            "value": self.value,
            "unit": self.unit,
            "uncertainty": self.uncertainty,
            "provenance_type": self.provenance_type.value,
            "verification_status": self.verification_status.value,
            "source": self.source,
            "source_location": self.source_location,
            "method": self.method,
            "notes": self.notes,
            "observable_role": self.observable_role,
        }

    @classmethod
    def from_jsonable(cls, data: Mapping[str, Any]) -> PhysicalParameter:
        """Build a strictly validated record from one JSON object."""

        required = {
            "name",
            "symbol",
            "value",
            "unit",
            "uncertainty",
            "provenance_type",
            "verification_status",
            "source",
            "source_location",
            "method",
            "notes",
            "observable_role",
        }
        if not isinstance(data, Mapping):
            raise TypeError("parameter record must be a JSON object")
        missing = sorted(required.difference(data))
        if missing:
            raise ValueError(f"parameter record is missing keys: {', '.join(missing)}")
        unexpected = sorted(set(data).difference(required))
        if unexpected:
            raise ValueError(
                "parameter record has unexpected keys: " + ", ".join(unexpected)
            )
        try:
            provenance_type = ProvenanceType(data["provenance_type"])
            verification_status = VerificationStatus(data["verification_status"])
        except ValueError as error:
            raise ValueError("unknown provenance_type or verification_status") from error
        uncertainty = data["uncertainty"]
        if uncertainty is not None:
            if isinstance(uncertainty, bool) or not isinstance(uncertainty, (int, float)):
                raise TypeError("uncertainty must be numeric or null")
            uncertainty = float(uncertainty)
        return cls(
            name=data["name"],
            symbol=data["symbol"],
            value=data["value"],
            unit=data["unit"],
            uncertainty=uncertainty,
            provenance_type=provenance_type,
            verification_status=verification_status,
            source=data["source"],
            source_location=data["source_location"],
            method=data["method"],
            notes=data["notes"],
            observable_role=data["observable_role"],
        )


@dataclass(frozen=True, slots=True)
class MG10FParameterRegistry:
    """Immutable nominal registry loaded from ``physics_mg10f.json``.

    ``is_material_ready`` means that the explicitly required calibration
    fields are non-null.  It deliberately does *not* mean the material is
    verified or eligible for Hydrogel/contact/McKean--Vlasov calculations.
    """

    schema_version: int
    registry_id: str
    registry_status: str
    reference_document: str
    required_calibration_parameters: tuple[str, ...]
    parameters: tuple[PhysicalParameter, ...]

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("unsupported MG10F registry schema_version")
        for field_name in ("registry_id", "registry_status", "reference_document"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        names = [parameter.name for parameter in self.parameters]
        if not names:
            raise ValueError("registry must contain parameters")
        if len(names) != len(set(names)):
            raise ValueError("parameter names must be unique")
        if self.registry_status != NOMINAL_MG10F_REGISTRY_STATUS:
            raise ValueError(
                "this nominal MG10F registry must retain registry_status "
                f"{NOMINAL_MG10F_REGISTRY_STATUS!r}"
            )
        if self.required_calibration_parameters != REQUIRED_CALIBRATION_PARAMETERS:
            raise ValueError(
                "required_calibration_parameters must exactly match the fixed "
                "MG10F calibration contract"
            )
        parameters_by_name = {parameter.name: parameter for parameter in self.parameters}
        missing_calibration = [
            name
            for name in REQUIRED_CALIBRATION_PARAMETERS
            if name not in parameters_by_name
        ]
        if missing_calibration:
            raise ValueError(
                "registry is missing fixed calibration parameters: "
                + ", ".join(missing_calibration)
            )
        for name, (expected_unit, expected_role) in REQUIRED_CALIBRATION_CONTRACT.items():
            parameter = parameters_by_name[name]
            if parameter.unit != expected_unit:
                raise ValueError(
                    f"calibration parameter {name!r} must use unit {expected_unit!r}"
                )
            if parameter.observable_role != expected_role:
                raise ValueError(
                    f"calibration parameter {name!r} has an invalid observable_role"
                )
            if parameter.provenance_type is not ProvenanceType.CALIBRATED:
                raise ValueError(
                    f"calibration parameter {name!r} must have CALIBRATED provenance"
                )
            if parameter.value is not None:
                if isinstance(parameter.value, bool) or not isinstance(
                    parameter.value, (int, float)
                ):
                    raise ValueError(
                        f"calibration parameter {name!r} must be numeric or null"
                    )
                value = float(parameter.value)
                if not math.isfinite(value):
                    raise ValueError(
                        f"calibration parameter {name!r} must be finite"
                    )
                if name == "network_density_times_solvent_volume" and value <= 0.0:
                    raise ValueError(
                        "network_density_times_solvent_volume must be positive"
                    )
                if name == "initial_polymer_volume_fraction" and not 0.0 < value < 1.0:
                    raise ValueError(
                        "initial_polymer_volume_fraction must be in (0, 1)"
                    )
        verified_names = [
            parameter.name
            for parameter in self.parameters
            if parameter.verification_status is VerificationStatus.VERIFIED
        ]
        if verified_names:
            raise ValueError(
                "a nominal MG10F registry cannot contain VERIFIED parameter records: "
                + ", ".join(verified_names)
            )

    @classmethod
    def from_jsonable(cls, data: Mapping[str, Any]) -> MG10FParameterRegistry:
        """Create a registry from a parsed standard-library JSON document."""

        required = {
            "schema_version",
            "registry_id",
            "registry_status",
            "reference_document",
            "required_calibration_parameters",
            "parameters",
        }
        if not isinstance(data, Mapping):
            raise TypeError("MG10F registry must be a JSON object")
        missing = sorted(required.difference(data))
        if missing:
            raise ValueError(f"registry is missing keys: {', '.join(missing)}")
        unexpected = sorted(set(data).difference(required))
        if unexpected:
            raise ValueError("registry has unexpected keys: " + ", ".join(unexpected))
        raw_parameters = data["parameters"]
        if not isinstance(raw_parameters, list):
            raise TypeError("parameters must be a JSON list")
        raw_required = data["required_calibration_parameters"]
        if not isinstance(raw_required, list) or not all(
            isinstance(name, str) for name in raw_required
        ):
            raise TypeError("required_calibration_parameters must be a list of strings")
        if tuple(raw_required) != REQUIRED_CALIBRATION_PARAMETERS:
            raise ValueError(
                "required_calibration_parameters must exactly match the fixed "
                "MG10F calibration contract"
            )
        return cls(
            schema_version=data["schema_version"],
            registry_id=data["registry_id"],
            registry_status=data["registry_status"],
            reference_document=data["reference_document"],
            required_calibration_parameters=tuple(raw_required),
            parameters=tuple(
                PhysicalParameter.from_jsonable(parameter)
                for parameter in raw_parameters
            ),
        )

    def parameter(self, name: str) -> PhysicalParameter:
        """Return one named parameter or fail rather than returning a default."""

        for parameter in self.parameters:
            if parameter.name == name:
                return parameter
        raise KeyError(f"MG10F parameter is not registered: {name}")

    def _numeric_value(self, name: str) -> float:
        parameter = self.parameter(name)
        value = parameter.value
        if value is None:
            raise MaterialNotReadyError(f"MG10F parameter is BLOCKED: {name}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"MG10F parameter must be numeric: {name}")
        return float(value)

    def _numeric_value_with_unit(
        self, name: str, expected_unit: str, *, strictly_positive: bool = False
    ) -> float:
        """Read one finite SI quantity without silently accepting unit changes."""

        parameter = self.parameter(name)
        if parameter.unit != expected_unit:
            raise MaterialNotReadyError(
                f"MG10F parameter {name!r} must use SI unit {expected_unit!r}, "
                f"not {parameter.unit!r}"
            )
        value = self._numeric_value(name)
        if not math.isfinite(value):
            raise MaterialNotReadyError(f"MG10F parameter {name!r} must be finite")
        if strictly_positive and value <= 0.0:
            raise MaterialNotReadyError(
                f"MG10F parameter {name!r} must be strictly positive"
            )
        return value

    def transport_readiness(self) -> dict[str, Any]:
        """Report whether nominal Stokes inputs can be derived without raising.

        This is intentionally a structural status report: a ``BLOCKED`` result
        means a missing or malformed transport input, not a substitute value.
        """

        blockers: list[dict[str, str]] = []
        for name, expected_unit in TRANSPORT_INPUT_UNITS.items():
            try:
                parameter = self.parameter(name)
            except KeyError:
                blockers.append({"parameter": name, "reason": "MISSING_PARAMETER"})
                continue
            if parameter.unit != expected_unit:
                blockers.append(
                    {
                        "parameter": name,
                        "reason": "INVALID_SI_UNIT",
                        "expected_unit": expected_unit,
                    }
                )
                continue
            value = parameter.value
            if value is None:
                blockers.append({"parameter": name, "reason": "MISSING_VALUE"})
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                blockers.append({"parameter": name, "reason": "NON_NUMERIC_VALUE"})
                continue
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                blockers.append({"parameter": name, "reason": "NONFINITE_VALUE"})
            elif numeric_value <= 0.0:
                blockers.append({"parameter": name, "reason": "NONPOSITIVE_VALUE"})
        return {
            "status": (
                "BLOCKED"
                if blockers
                else "CONDITIONAL_NOMINAL_DERIVATION_FROM_UNVERIFIED_INPUTS"
            ),
            "required_input_units": dict(TRANSPORT_INPUT_UNITS),
            "blockers": blockers,
        }

    @property
    def blocked_calibration_parameters(self) -> tuple[str, ...]:
        """Names of calibration fields that remain null/BLOCKED."""

        return tuple(
            name
            for name in self.required_calibration_parameters
            if self.parameter(name).value is None
        )

    @property
    def is_material_ready(self) -> bool:
        """Whether this registry can be admitted as a material model.

        This particular schema is permanently a nominal, test-only snapshot.
        Completing JSON fields cannot promote it to a material model; a future
        calibrated registry needs a separately reviewed schema and workflow.
        """

        return False

    @property
    def unverified_usable_parameters(self) -> tuple[str, ...]:
        """All non-null records lacking a verified primary source location."""

        return tuple(
            parameter.name
            for parameter in self.parameters
            if parameter.is_usable and not parameter.has_verifiable_source_location
        )

    @property
    def is_verified_material_ready(self) -> bool:
        """Strict material gate: calibrated values and source locations are required."""

        return self.is_material_ready and not self.unverified_usable_parameters

    def assert_verified_material_ready(self) -> None:
        """Fail closed unless all registered usable values are source-verified."""

        blockers: list[str] = []
        blockers.append(
            "registry is a nominal TEST_ONLY_NOT_CALIBRATED snapshot and cannot "
            "be promoted by editing JSON"
        )
        if self.blocked_calibration_parameters:
            blockers.append(
                "missing calibration values: "
                + ", ".join(self.blocked_calibration_parameters)
            )
        if self.unverified_usable_parameters:
            blockers.append(
                "missing verified source location: "
                + ", ".join(self.unverified_usable_parameters)
            )
        if blockers:
            raise MaterialNotReadyError("MG10F material is not verified: " + "; ".join(blockers))

    def nominal_transport_parameters(self) -> tuple[PhysicalParameter, ...]:
        """Return conditional 20 degC Stokes--Einstein transport quantities.

        These values are calculations from nominal, unverified inputs.  They
        are not solver defaults and must not be promoted to calibrated values.
        """

        temperature = self._numeric_value_with_unit(
            "temperature_20c", TRANSPORT_INPUT_UNITS["temperature_20c"], strictly_positive=True
        )
        radius = self._numeric_value_with_unit(
            "hydrodynamic_radius_20c",
            TRANSPORT_INPUT_UNITS["hydrodynamic_radius_20c"],
            strictly_positive=True,
        )
        viscosity = self._numeric_value_with_unit(
            "water_viscosity_20c",
            TRANSPORT_INPUT_UNITS["water_viscosity_20c"],
            strictly_positive=True,
        )
        molecular_volume = self._numeric_value_with_unit(
            "water_molecular_volume",
            TRANSPORT_INPUT_UNITS["water_molecular_volume"],
            strictly_positive=True,
        )
        gamma = 6.0 * math.pi * viscosity * radius
        mobility = 1.0 / gamma
        thermal_energy = BOLTZMANN_CONSTANT_J_PER_K * temperature
        diffusion = mobility * thermal_energy
        thermal_energy_density = thermal_energy / molecular_volume
        radius_uncertainty = self.parameter("hydrodynamic_radius_20c").uncertainty
        radius_relative_uncertainty = (
            None if radius_uncertainty is None else radius_uncertainty / radius
        )
        gamma_uncertainty = (
            None
            if radius_relative_uncertainty is None
            else gamma * radius_relative_uncertainty
        )
        mobility_uncertainty = (
            None
            if radius_relative_uncertainty is None
            else mobility * radius_relative_uncertainty
        )
        diffusion_uncertainty = (
            None
            if radius_relative_uncertainty is None
            else diffusion * radius_relative_uncertainty
        )
        status = VerificationStatus.CONDITIONAL_NOMINAL_DERIVED_FROM_UNVERIFIED_INPUTS
        source = (
            "Derived from registered nominal temperature, hydrodynamic radius, "
            "water viscosity, and molecular volume"
        )
        common_notes = (
            "Conditional nominal derivation from inputs whose primary source locations "
            "still need verification; not a calibrated solver default. Uncertainty, "
            "where present, propagates only the reported hydrodynamic-radius "
            "uncertainty and is not a total uncertainty; viscosity, temperature, "
            "soft-particle, concentration, and wall corrections are not included."
        )
        return (
            PhysicalParameter(
                name="stokes_drag_coefficient",
                symbol="gamma",
                value=gamma,
                unit="kg/s",
                uncertainty=gamma_uncertainty,
                provenance_type=ProvenanceType.DERIVED,
                verification_status=status,
                source=source,
                source_location=None,
                method="gamma = 6*pi*eta*R_h",
                notes=common_notes,
                observable_role="CONDITIONAL_NOMINAL_TRANSPORT_OUTPUT",
            ),
            PhysicalParameter(
                name="mobility",
                symbol="M",
                value=mobility,
                unit="m/(N s)",
                uncertainty=mobility_uncertainty,
                provenance_type=ProvenanceType.DERIVED,
                verification_status=status,
                source=source,
                source_location=None,
                method="M = 1/gamma",
                notes=common_notes,
                observable_role="CONDITIONAL_NOMINAL_TRANSPORT_OUTPUT",
            ),
            PhysicalParameter(
                name="thermal_energy",
                symbol="k_B*T",
                value=thermal_energy,
                unit="J",
                uncertainty=None,
                provenance_type=ProvenanceType.DERIVED,
                verification_status=status,
                source=source,
                source_location=None,
                method="k_B*T with exact SI k_B",
                notes=common_notes,
                observable_role="CONDITIONAL_NOMINAL_TRANSPORT_OUTPUT",
            ),
            PhysicalParameter(
                name="diffusion",
                symbol="D",
                value=diffusion,
                unit="m^2/s",
                uncertainty=diffusion_uncertainty,
                provenance_type=ProvenanceType.DERIVED,
                verification_status=status,
                source=source,
                source_location=None,
                method="D = M*k_B*T",
                notes=common_notes,
                observable_role="CONDITIONAL_NOMINAL_TRANSPORT_OUTPUT",
            ),
            PhysicalParameter(
                name="thermal_energy_density",
                symbol="k_B*T/nu",
                value=thermal_energy_density,
                unit="Pa",
                uncertainty=None,
                provenance_type=ProvenanceType.DERIVED,
                verification_status=status,
                source=source,
                source_location=None,
                method="k_B*T/nu; nu is molecular volume in m^3/molecule",
                notes=common_notes,
                observable_role="CONDITIONAL_NOMINAL_HYDROGEL_OUTPUT",
            ),
        )

    def flory_huggins_chi(
        self, polymer_volume_fraction: float, temperature_kelvin: float
    ) -> float:
        """Evaluate the registered, still-unverified ``chi(phi, T)`` relation.

        This utility does not alter the existing constant-chi Hydrogel model.
        """

        phi = float(polymer_volume_fraction)
        temperature = float(temperature_kelvin)
        if not math.isfinite(phi) or not 0.0 <= phi <= 1.0:
            raise ValueError("polymer_volume_fraction must be finite and in [0, 1]")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature_kelvin must be finite and positive")
        a0 = self._numeric_value_with_unit("chi_A0", CHI_COEFFICIENT_UNITS["chi_A0"])
        b0 = self._numeric_value_with_unit(
            "chi_B0_per_kelvin", CHI_COEFFICIENT_UNITS["chi_B0_per_kelvin"]
        )
        a1 = self._numeric_value_with_unit("chi_A1", CHI_COEFFICIENT_UNITS["chi_A1"])
        b1 = self._numeric_value_with_unit(
            "chi_B1_per_kelvin", CHI_COEFFICIENT_UNITS["chi_B1_per_kelvin"]
        )
        return a0 + b0 * temperature + phi * (a1 + b1 * temperature)

    def reference_contact_radius_m(self) -> NoReturn:
        """Reject a forbidden reinterpretation of the hydrodynamic radius."""

        raise MaterialNotReadyError(
            "MG10F hydrodynamic_radius_20c is a swollen hydrodynamic radius, "
            "not a reference contact radius"
        )

    def hydrogel_constructor_kwargs(self) -> NoReturn:
        """Reject automatic conversion to the existing ``HydrogelParameters``.

        In addition to the missing calibrated values, the registry has a
        state-dependent ``chi(phi, T)`` relation while the current Hydrogel
        implementation accepts a constant scalar chi.  An explicit calibrated
        bridge must be designed and reviewed before any construction occurs.
        """

        missing = ", ".join(self.blocked_calibration_parameters)
        raise MaterialNotReadyError(
            "cannot construct HydrogelParameters from MG10F registry: missing "
            f"calibrated values ({missing}); chi(phi, T) is not an approved "
            "constant-chi conversion"
        )

    def to_hydrogel_parameters(self) -> NoReturn:
        """Explicitly reject a direct registry-to-Hydrogel conversion."""

        return self.hydrogel_constructor_kwargs()

    def as_jsonable(self) -> dict[str, Any]:
        """Export provenance, gates, and nominal derivations as JSON-native data."""

        transport_readiness = self.transport_readiness()
        derived_transport_parameters: list[dict[str, JsonScalar]] = []
        if transport_readiness["status"] != "BLOCKED":
            try:
                derived_transport_parameters = [
                    parameter.as_jsonable()
                    for parameter in self.nominal_transport_parameters()
                ]
            except (KeyError, MaterialNotReadyError, OverflowError, ValueError) as error:
                transport_readiness = {
                    **transport_readiness,
                    "status": "BLOCKED",
                    "blockers": [
                        *transport_readiness["blockers"],
                        {
                            "parameter": "transport_derivation",
                            "reason": "DERIVATION_FAILED",
                            "detail": str(error),
                        },
                    ],
                }
        return {
            "schema_version": self.schema_version,
            "registry_id": self.registry_id,
            "registry_status": self.registry_status,
            "reference_document": self.reference_document,
            "required_calibration_parameters": list(self.required_calibration_parameters),
            "parameters": [parameter.as_jsonable() for parameter in self.parameters],
            "derived_transport_parameters": derived_transport_parameters,
            "transport_readiness": transport_readiness,
            "material_readiness": {
                "is_material_ready": self.is_material_ready,
                "is_verified_material_ready": self.is_verified_material_ready,
                "blocked_calibration_parameters": list(self.blocked_calibration_parameters),
                "unverified_usable_parameters": list(self.unverified_usable_parameters),
                "hydrogel_conversion": "BLOCKED",
                "contact_reference_radius": "BLOCKED",
            },
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the complete registry through Python's standard ``json`` module."""

        return json.dumps(self.as_jsonable(), indent=indent, sort_keys=True)


def default_mg10f_config_path() -> Path:
    """Return the repository-local nominal MG10F JSON configuration path."""

    return Path(__file__).resolve().parents[3] / "configs" / "physics_mg10f.json"


def load_mg10f_registry(config_path: str | Path | None = None) -> MG10FParameterRegistry:
    """Load the nominal MG10F registry without changing any solver defaults."""

    path = Path(config_path) if config_path is not None else default_mg10f_config_path()
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(
            handle,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    if not isinstance(data, Mapping):
        raise TypeError("MG10F configuration root must be a JSON object")
    return MG10FParameterRegistry.from_jsonable(data)
