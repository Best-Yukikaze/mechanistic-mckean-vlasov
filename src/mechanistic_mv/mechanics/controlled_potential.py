"""Control interfaces that map an input to a conservative potential."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Protocol

import numpy as np

from .magnetic_particle_potential import (
    LinearMagneticParticle,
    MagneticParticlePotential,
    TabulatedMagnetizationLaw,
)


class ControlledPotentialBackend(Protocol):
    """Contract for the preferred control path ``u -> V(x; u)``."""

    name: str
    physical_status: str

    def potential_joule(
        self, positions_m: np.ndarray, control: np.ndarray | None
    ) -> np.ndarray: ...

    def force_newton(
        self, positions_m: np.ndarray, control: np.ndarray | None
    ) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class ZeroControlledPotential:
    name: str = field(default="zero_controlled_potential", init=False)
    physical_status: str = field(default="physical null control input", init=False)

    def potential_joule(
        self, positions_m: np.ndarray, control: np.ndarray | None = None
    ) -> np.ndarray:
        _require_null_control(control)
        points = _points(positions_m)
        return np.zeros(points.shape[:-1], dtype=np.float64)

    def force_newton(
        self, positions_m: np.ndarray, control: np.ndarray | None = None
    ) -> np.ndarray:
        _require_null_control(control)
        return np.zeros_like(_points(positions_m))


@dataclass(frozen=True, slots=True)
class TestOnlyUniformFieldPotential:
    """TEST_ONLY_NOT_FINAL_PHYSICS linear potential for direction checks.

    ``V(x;u)=-F_max u dot (x-x_ref)`` and therefore
    ``-grad(V)=F_max u``. No claim is made that this is a calibrated actuator.
    """

    maximum_force_newton: float
    reference_position_m: tuple[float, float] = (0.0, 0.0)
    name: str = field(
        default="test_only_not_final_physics_uniform_field_potential", init=False
    )
    physical_status: str = field(default="TEST_ONLY_NOT_FINAL_PHYSICS", init=False)

    def __post_init__(self) -> None:
        reference = np.asarray(self.reference_position_m, dtype=np.float64)
        if reference.shape != (2,) or not np.all(np.isfinite(reference)):
            raise ValueError("reference_position_m must be a finite two-vector")
        if not np.isfinite(self.maximum_force_newton) or self.maximum_force_newton <= 0:
            raise ValueError("maximum_force_newton must be finite and positive")
        object.__setattr__(
            self,
            "reference_position_m",
            (float(reference[0]), float(reference[1])),
        )

    def potential_joule(
        self, positions_m: np.ndarray, control: np.ndarray | None
    ) -> np.ndarray:
        points = _points(positions_m)
        force = self._force_vector(control)
        displacement = points - np.asarray(self.reference_position_m)
        return -np.sum(displacement * force, axis=-1)

    def force_newton(
        self, positions_m: np.ndarray, control: np.ndarray | None
    ) -> np.ndarray:
        points = _points(positions_m)
        return np.broadcast_to(self._force_vector(control), points.shape).copy()

    def _force_vector(self, control: np.ndarray | None) -> np.ndarray:
        if control is None:
            vector = np.zeros(2, dtype=np.float64)
        else:
            vector = np.asarray(control, dtype=np.float64)
            if vector.shape != (2,) or not np.all(np.isfinite(vector)):
                raise ValueError("control must be a finite two-vector")
            if np.linalg.norm(vector) > 1.0 + 1.0e-12:
                raise ValueError("test control norm cannot exceed one")
        return self.maximum_force_newton * vector


def _points(values: np.ndarray) -> np.ndarray:
    points = np.asarray(values, dtype=np.float64)
    if points.shape[-1:] != (2,) or not np.all(np.isfinite(points)):
        raise ValueError("positions must be finite and end in two components")
    return points


def _require_null_control(control: np.ndarray | None) -> None:
    """Reject a nonzero command when no actuator backend is installed."""

    if control is None:
        return
    vector = np.asarray(control, dtype=np.float64)
    if vector.shape != (2,) or not np.all(np.isfinite(vector)):
        raise ValueError("null-backend control must be a finite two-vector")
    if np.any(vector != 0.0):
        raise ValueError("nonzero control requires a physical potential backend")

SOURCE_PARAMETERIZED_MAGNETIC_CONTROL_STATUS = (
    "SOURCE_PARAMETERIZED_MAGNETIC_CONTROL_REQUIRES_PHYSICS_VALIDATION"
)


def _finite_scalar(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonempty_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _finite_gradient_vector2(value: object, *, name: str) -> tuple[float, float]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (2,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite two-vector")
    return (float(vector[0]), float(vector[1]))


@dataclass(frozen=True, slots=True)
class MagneticSourceReference:
    """Traceable source metadata for one required magnetic quantity.

    The Controller validates that a reference is declared, but it does not
    decide whether the cited source is scientifically sufficient.  That is a
    separate Physics/Experiment validation responsibility.
    """

    source_id: str
    locator: str
    quantity: str
    units: str
    provenance_class: str

    def __post_init__(self) -> None:
        for field_name in (
            "source_id",
            "locator",
            "quantity",
            "units",
            "provenance_class",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonempty_text(getattr(self, field_name), name=field_name),
            )

    def as_jsonable(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "locator": self.locator,
            "quantity": self.quantity,
            "units": self.units,
            "provenance_class": self.provenance_class,
        }


@dataclass(frozen=True, slots=True)
class LinearMagneticControlSourcePayload:
    """The Controller-side provenance envelope for the canonical Physics law.

    The ``LinearMagneticParticle`` is created by the Physics-owned magnetic
    law module and carries the required ``chi_v``, magnetic-particle volume,
    and source locator.  This envelope adds the independently required
    provenance of the command-to-``B`` map and its in-plane gradient.  It does
    not duplicate the magnetic energy or force formula.
    """

    particle_law: LinearMagneticParticle
    flux_density_provenance: MagneticSourceReference
    flux_density_gradient_provenance: MagneticSourceReference
    physical_status: str = field(
        default=SOURCE_PARAMETERIZED_MAGNETIC_CONTROL_STATUS, init=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.particle_law, LinearMagneticParticle):
            raise TypeError(
                "particle_law must be the Physics-owned LinearMagneticParticle"
            )
        for field_name in (
            "flux_density_provenance",
            "flux_density_gradient_provenance",
        ):
            if not isinstance(getattr(self, field_name), MagneticSourceReference):
                raise TypeError(f"{field_name} must be a MagneticSourceReference")

    def as_jsonable(self) -> dict[str, object]:
        particle_law = self.particle_law
        return {
            "magnetic_law_owner": "Physics.LinearMagneticParticle",
            "magnetic_law_status": particle_law.physical_status,
            "chi_v_dimensionless": particle_law.chi_v_dimensionless,
            "magnetic_particle_volume_m3": particle_law.particle_volume_m3,
            "particle_law_source_locator": particle_law.source_locator,
            "particle_law_provenance_class": particle_law.provenance_class,
            "chi_v_and_linear_magnetization_provenance": {
                "source_locator": particle_law.source_locator,
                "provenance_class": particle_law.provenance_class,
            },
            "magnetization_relation": "m_induced=chi_v*V_particle*B/mu0 [A m^2]",
            "flux_density_provenance": self.flux_density_provenance.as_jsonable(),
            "flux_density_gradient_provenance": (
                self.flux_density_gradient_provenance.as_jsonable()
            ),
            "physical_status": self.physical_status,
            "contains_density": False,
            "contains_phi": False,
            "contains_material_parameters": False,
            "contains_diffusion_or_mobility": False,
            "contains_empirical_correction": False,
            "contains_pair_interaction": False,
        }



    @property
    def particle_volume_m3(self) -> float:
        """Expose the sourced volume without making it a controller command."""

        return self.particle_law.particle_volume_m3

@dataclass(frozen=True, slots=True)
class TabulatedMagneticControlSourcePayload:
    """Source envelope for the Physics-owned reversible table ``M(B)`` route.

    The table itself is immutable Physics-owned data.  Unlike the linear law,
    it does not embed a particle volume, so this Controller-side envelope
    requires a separately sourced particle volume before it can bind a field
    command.  It deliberately has no density, phi, transport, or empirical
    correction input.
    """

    particle_law: TabulatedMagnetizationLaw
    particle_volume_m3: float
    particle_volume_provenance: MagneticSourceReference
    flux_density_provenance: MagneticSourceReference
    flux_density_gradient_provenance: MagneticSourceReference
    physical_status: str = field(
        default=SOURCE_PARAMETERIZED_MAGNETIC_CONTROL_STATUS, init=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.particle_law, TabulatedMagnetizationLaw):
            raise TypeError(
                "particle_law must be the Physics-owned TabulatedMagnetizationLaw"
            )
        particle_volume = _finite_scalar(
            self.particle_volume_m3, name="particle_volume_m3"
        )
        if particle_volume <= 0.0:
            raise ValueError("particle_volume_m3 must be finite and positive")
        object.__setattr__(self, "particle_volume_m3", particle_volume)
        for field_name in (
            "particle_volume_provenance",
            "flux_density_provenance",
            "flux_density_gradient_provenance",
        ):
            if not isinstance(getattr(self, field_name), MagneticSourceReference):
                raise TypeError(f"{field_name} must be a MagneticSourceReference")

    def as_jsonable(self) -> dict[str, object]:
        particle_law = self.particle_law
        return {
            "magnetic_law_owner": "Physics.TabulatedMagnetizationLaw",
            "magnetic_law_status": particle_law.physical_status,
            "magnetization_law": "SOURCE_TABULATED_M_OF_B",
            "magnetization_table_field_units": "T",
            "magnetization_table_value_units": "A/m",
            "magnetization_table_source_locator": particle_law.source_locator,
            "magnetization_table_provenance_class": particle_law.provenance_class,
            "magnetic_particle_volume_m3": self.particle_volume_m3,
            "particle_volume_provenance": self.particle_volume_provenance.as_jsonable(),
            "flux_density_provenance": self.flux_density_provenance.as_jsonable(),
            "flux_density_gradient_provenance": (
                self.flux_density_gradient_provenance.as_jsonable()
            ),
            "physical_status": self.physical_status,
            "contains_density": False,
            "contains_phi": False,
            "contains_diffusion_or_mobility": False,
            "contains_empirical_correction": False,
            "contains_pair_interaction": False,
        }


MagneticControlSourcePayload = (
    LinearMagneticControlSourcePayload | TabulatedMagneticControlSourcePayload
)
_MAGNETIC_CONTROL_SOURCE_PAYLOAD_TYPES = (
    LinearMagneticControlSourcePayload,
    TabulatedMagneticControlSourcePayload,
)
@dataclass(frozen=True, slots=True)
class CoilCurrentCommand:
    """One immutable, SI-valued field-map command ``u`` for a single coil."""

    current_ampere: float
    command_id: str = "single_coil_current"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "current_ampere", _finite_scalar(self.current_ampere, name="current_ampere")
        )
        object.__setattr__(
            self, "command_id", _nonempty_text(self.command_id, name="command_id")
        )

    def as_jsonable(self) -> dict[str, object]:
        return {
            "command_id": self.command_id,
            "current_ampere": self.current_ampere,
            "units": "A",
        }


class SourceBackedMagneticFieldSnapshot(Protocol):
    """The explicit field-only object accepted by the canonical Physics law."""

    name: str
    physical_status: str
    source_payload: MagneticControlSourcePayload

    def flux_density_tesla(self, positions_m: np.ndarray) -> np.ndarray: ...

    def gradient_flux_density_tesla_per_m(self, positions_m: np.ndarray) -> np.ndarray: ...

    def configuration_as_jsonable(self) -> dict[str, object]: ...


class SourceBackedMagneticFieldMap(Protocol):
    """Explicit source-payload map from one physical command to a field snapshot."""

    name: str
    physical_status: str
    source_payload: MagneticControlSourcePayload

    def bind_command(
        self, command: CoilCurrentCommand
    ) -> SourceBackedMagneticFieldSnapshot: ...

    def configuration_as_jsonable(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class SourceBackedAffineCoilMagneticFieldMap:
    """Bind a source-declared SI coil command to scalar ``B`` and ``grad(B)``.

    The canonical Physics law consumes a non-negative magnetic-flux-density
    magnitude in tesla and its in-plane gradient in tesla per metre.  This map
    is deliberately only the command-to-field part of the chain:

    ``B(x; I) = B0 + I * (B_per_A + grad_B_per_A dot x)``,
    ``grad_xy B(x; I) = I * grad_B_per_A``.

    It has no magnetic-energy formula and cannot move a particle or edit an
    MV state.  A real source map can replace this affine implementation while
    retaining the same bound snapshot interface.
    """

    source_payload: MagneticControlSourcePayload
    zero_current_flux_density_tesla: float
    flux_density_per_ampere_tesla: float
    flux_density_gradient_per_ampere_tesla_per_m: tuple[float, float]
    minimum_current_ampere: float
    maximum_current_ampere: float
    name: str = field(
        default="source_parameterized_affine_coil_magnetic_field_map", init=False
    )
    physical_status: str = field(
        default=SOURCE_PARAMETERIZED_MAGNETIC_CONTROL_STATUS, init=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.source_payload, _MAGNETIC_CONTROL_SOURCE_PAYLOAD_TYPES):
            raise TypeError(
                "source_payload must be a source-backed linear or tabulated magnetic payload"
            )
        zero_current = _finite_scalar(
            self.zero_current_flux_density_tesla,
            name="zero_current_flux_density_tesla",
        )
        if zero_current < 0.0:
            raise ValueError("zero_current_flux_density_tesla must be non-negative")
        object.__setattr__(
            self,
            "zero_current_flux_density_tesla",
            zero_current,
        )
        object.__setattr__(
            self,
            "flux_density_per_ampere_tesla",
            _finite_scalar(
                self.flux_density_per_ampere_tesla,
                name="flux_density_per_ampere_tesla",
            ),
        )
        object.__setattr__(
            self,
            "flux_density_gradient_per_ampere_tesla_per_m",
            _finite_gradient_vector2(
                self.flux_density_gradient_per_ampere_tesla_per_m,
                name="flux_density_gradient_per_ampere_tesla_per_m",
            ),
        )
        lower = _finite_scalar(self.minimum_current_ampere, name="minimum_current_ampere")
        upper = _finite_scalar(self.maximum_current_ampere, name="maximum_current_ampere")
        if lower >= upper:
            raise ValueError("minimum_current_ampere must be less than maximum_current_ampere")
        object.__setattr__(self, "minimum_current_ampere", lower)
        object.__setattr__(self, "maximum_current_ampere", upper)

    def bind_command(self, command: CoilCurrentCommand) -> "BoundMagneticFieldSnapshot":
        """Return the immutable field object the canonical Physics law consumes."""

        self._current_ampere(command)
        return BoundMagneticFieldSnapshot(field_map=self, command=command)

    def configuration_as_jsonable(self) -> dict[str, object]:
        return {
            "name": self.name,
            "physical_status": self.physical_status,
            "field_map_formula": "B(x;I)=B0+I*(B_per_A+grad_B_per_A dot x)",
            "gradient_formula": "grad_xy_B(x;I)=I*grad_B_per_A",
            "position_units": "m",
            "command_units": "A",
            "flux_density_units": "T",
            "flux_density_gradient_units": "T/m",
            "zero_current_flux_density_tesla": self.zero_current_flux_density_tesla,
            "flux_density_per_ampere_tesla": self.flux_density_per_ampere_tesla,
            "flux_density_gradient_per_ampere_tesla_per_m": list(
                self.flux_density_gradient_per_ampere_tesla_per_m
            ),
            "minimum_current_ampere": self.minimum_current_ampere,
            "maximum_current_ampere": self.maximum_current_ampere,
            "source_payload": self.source_payload.as_jsonable(),
            "contains_density": False,
            "contains_particle_positions": False,
            "contains_phi": False,
            "contains_diffusion_or_mobility": False,
            "contains_empirical_correction": False,
            "contains_pair_interaction": False,
        }

    def _current_ampere(self, command: object) -> float:
        if not isinstance(command, CoilCurrentCommand):
            raise TypeError("magnetic field map requires CoilCurrentCommand")
        current = command.current_ampere
        if current < self.minimum_current_ampere or current > self.maximum_current_ampere:
            raise ValueError("coil current is outside the source-declared field-map bounds")
        return current


@dataclass(frozen=True, slots=True)
class BoundMagneticFieldSnapshot:
    """Immutable command-bound field implementing the canonical Physics protocol."""

    field_map: SourceBackedAffineCoilMagneticFieldMap
    command: CoilCurrentCommand
    name: str = field(default="bound_source_parameterized_magnetic_field", init=False)
    physical_status: str = field(
        default=SOURCE_PARAMETERIZED_MAGNETIC_CONTROL_STATUS, init=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.field_map, SourceBackedAffineCoilMagneticFieldMap):
            raise TypeError("field_map must be SourceBackedAffineCoilMagneticFieldMap")
        self.field_map._current_ampere(self.command)

    @property
    def source_payload(self) -> MagneticControlSourcePayload:
        return self.field_map.source_payload

    def flux_density_tesla(self, positions_m: np.ndarray) -> np.ndarray:
        points = _points(positions_m)
        current = self.field_map._current_ampere(self.command)
        gradient = np.asarray(
            self.field_map.flux_density_gradient_per_ampere_tesla_per_m,
            dtype=np.float64,
        )
        field = self.field_map.zero_current_flux_density_tesla + current * (
            self.field_map.flux_density_per_ampere_tesla
            + np.einsum("i,...i->...", gradient, points)
        )
        if not np.all(np.isfinite(field)) or np.any(field < 0.0):
            raise ValueError(
                "source-declared field map produced an invalid negative or non-finite B magnitude"
            )
        return field

    def gradient_flux_density_tesla_per_m(self, positions_m: np.ndarray) -> np.ndarray:
        points = _points(positions_m)
        current = self.field_map._current_ampere(self.command)
        gradient = current * np.asarray(
            self.field_map.flux_density_gradient_per_ampere_tesla_per_m,
            dtype=np.float64,
        )
        result = np.broadcast_to(gradient, points.shape).copy()
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("magnetic flux-density gradient became non-finite")
        return result

    def configuration_as_jsonable(self) -> dict[str, object]:
        return {
            "name": self.name,
            "physical_status": self.physical_status,
            "command": self.command.as_jsonable(),
            "field_map": self.field_map.configuration_as_jsonable(),
            "contains_density": False,
            "contains_particle_positions": False,
            "contains_phi": False,
            "contains_diffusion_or_mobility": False,
            "contains_empirical_correction": False,
            "contains_pair_interaction": False,
        }


class MagneticFieldControlAdapter:
    """Controller-only adapter from a physical command to a Physics field snapshot.

    It performs no magnetic energy or force calculation.  Instead,
    :meth:`physics_potential_for` instantiates the canonical
    :class:`~mechanistic_mv.mechanics.magnetic_particle_potential.MagneticParticlePotential`
    with a command-bound field snapshot.  This makes the ownership boundary
    explicit: Controller owns ``u -> B, grad(B)``; Physics owns
    ``B, grad(B) -> V_mag -> F_mag``.
    """

    name = "source_parameterized_magnetic_field_control_adapter"
    physical_status = SOURCE_PARAMETERIZED_MAGNETIC_CONTROL_STATUS

    def __init__(self, *, field_map: SourceBackedMagneticFieldMap) -> None:
        source_payload = getattr(field_map, "source_payload", None)
        if not isinstance(source_payload, _MAGNETIC_CONTROL_SOURCE_PAYLOAD_TYPES):
            raise TypeError(
                "field_map must expose a complete source-backed linear or tabulated magnetic payload"
            )
        for method_name in ("bind_command", "configuration_as_jsonable"):
            if not callable(getattr(field_map, method_name, None)):
                raise TypeError(f"field_map is missing {method_name}")
        self.field_map = field_map

    @property
    def source_payload(self) -> MagneticControlSourcePayload:
        return self.field_map.source_payload

    def adapt_command(self, command: CoilCurrentCommand) -> SourceBackedMagneticFieldSnapshot:
        """Bind one checked SI command without reading or changing MV state."""

        if not isinstance(command, CoilCurrentCommand):
            raise TypeError("magnetic controller requires CoilCurrentCommand")
        field = self.field_map.bind_command(command)
        if getattr(field, "source_payload", None) is not self.source_payload:
            raise ValueError("field snapshot must retain the map's source payload")
        for method_name in (
            "flux_density_tesla",
            "gradient_flux_density_tesla_per_m",
            "configuration_as_jsonable",
        ):
            if not callable(getattr(field, method_name, None)):
                raise TypeError(f"field snapshot is missing {method_name}")
        return field

    def physics_potential_for(self, command: CoilCurrentCommand) -> MagneticParticlePotential:
        """Build the canonical Physics potential for one immutable field command."""

        field = self.adapt_command(command)
        source_payload = self.source_payload
        particle_law = source_payload.particle_law
        return MagneticParticlePotential(
            particle_law=particle_law,
            particle_volume_m3=source_payload.particle_volume_m3,
            magnetic_field=field,
        )

    def configuration_as_jsonable(self) -> dict[str, object]:
        return {
            "adapter": "MagneticFieldControlAdapter",
            "name": self.name,
            "physical_status": self.physical_status,
            "control_chain": "u[current_A] -> B(x;u)[T], grad_xy_B[T/m] -> Physics V_mag/F_mag",
            "physics_law_owner": "mechanistic_mv.mechanics.magnetic_particle_potential",
            "source_payload": self.source_payload.as_jsonable(),
            "field_map": self.field_map.configuration_as_jsonable(),
            "contains_density": False,
            "contains_particle_positions": False,
            "contains_phi": False,
            "contains_diffusion_or_mobility": False,
            "contains_empirical_correction": False,
            "contains_pair_interaction": False,
            "direct_state_mutation": False,
        }
