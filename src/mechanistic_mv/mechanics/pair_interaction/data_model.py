"""Validated external data contract for radial pair forces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..hydrogel.equilibrium import TimeScaleStatus


class PairDataValidationStatus(str, Enum):
    """Whether an external force table passed its declared upstream checks."""

    PASSED = "PASSED"
    FAILED = "FAILED"
    UNVERIFIED = "UNVERIFIED"


class PairForceScaling(str, Enum):
    """Scaling meaning of the tabulated force and its integrated potential."""

    KAC_NORMALIZED_PROBABILITY = (
        "KAC_EFFECTIVE_FOR_UNIT_MASS_RHO_AND_ONE_OVER_N_PARTICLE_FORCE"
    )
    UNSCALED_SINGLE_PAIR = "UNSCALED_PHYSICAL_SINGLE_PAIR"


@dataclass(frozen=True, slots=True)
class PairForceMetadata:
    """Required provenance and validation semantics; units are fixed to SI."""

    dataset_id: str
    source: str
    physical_status: str
    solver_status: str
    validation_status: PairDataValidationStatus
    time_scale_status: TimeScaleStatus
    scaling: PairForceScaling
    reference_distance_m: float
    reference_force_tolerance_newton: float

    def __post_init__(self) -> None:
        for name in ("dataset_id", "source", "physical_status", "solver_status"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.scaling, PairForceScaling):
            raise TypeError("scaling must be a PairForceScaling value")
        if not isinstance(self.validation_status, PairDataValidationStatus):
            raise TypeError("validation_status must be a PairDataValidationStatus")
        if not isinstance(self.time_scale_status, TimeScaleStatus):
            raise TypeError("time_scale_status must be a TimeScaleStatus")
        values = np.asarray(
            [self.reference_distance_m, self.reference_force_tolerance_newton],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)) or self.reference_distance_m <= 0.0:
            raise ValueError("reference distance must be finite and positive")
        if self.reference_force_tolerance_newton < 0.0:
            raise ValueError("reference force tolerance must be non-negative")


@dataclass(frozen=True, slots=True)
class PairForceTable:
    """Strict SI table of radial force ``F(r)`` with positive repulsion sign."""

    center_distance_m: np.ndarray
    radial_force_newton: np.ndarray
    metadata: PairForceMetadata

    def __post_init__(self) -> None:
        distance = np.asarray(self.center_distance_m, dtype=np.float64)
        force = np.asarray(self.radial_force_newton, dtype=np.float64)
        if distance.ndim != 1 or force.ndim != 1 or distance.shape != force.shape:
            raise ValueError("distance and force must be matching one-dimensional arrays")
        if distance.size < 4:
            raise ValueError("at least four force samples are required")
        if not np.all(np.isfinite(distance)) or not np.all(np.isfinite(force)):
            raise ValueError("pair-force data must be finite")
        if np.any(distance < 0.0) or np.any(np.diff(distance) <= 0.0):
            raise ValueError("center distances must be non-negative and strictly increasing")
        if np.any(force < 0.0):
            raise ValueError(
                "radial force must be non-negative for frictionless, non-adhesive contact"
            )
        if not np.isclose(
            distance[-1],
            self.metadata.reference_distance_m,
            rtol=16.0 * np.finfo(np.float64).eps,
            atol=0.0,
        ):
            raise ValueError("the final distance must equal the metadata reference distance")
        if abs(force[-1]) > self.metadata.reference_force_tolerance_newton:
            raise ValueError("force at r_ref exceeds the declared negligible-force tolerance")
        distance = distance.copy()
        force = force.copy()
        distance.setflags(write=False)
        force.setflags(write=False)
        object.__setattr__(self, "center_distance_m", distance)
        object.__setattr__(self, "radial_force_newton", force)
