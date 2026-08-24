"""Shared probability-density and number-density interaction contract."""

from __future__ import annotations

from enum import Enum
from typing import Protocol

import numpy as np


class DensityConvention(str, Enum):
    """Integral meaning of a continuum or empirical density in ``1/m^2``."""

    PROBABILITY = "PROBABILITY_DENSITY_INTEGRAL_ONE"
    NUMBER = "NUMBER_DENSITY_INTEGRAL_POPULATION"


class PairForceScaling(str, Enum):
    """Scaling meaning of a pair force and its integrated potential."""

    KAC_NORMALIZED_PROBABILITY = (
        "KAC_EFFECTIVE_FOR_UNIT_MASS_RHO_AND_ONE_OVER_N_PARTICLE_FORCE"
    )
    UNSCALED_SINGLE_PAIR = "UNSCALED_PHYSICAL_SINGLE_PAIR"


class PairScalingCarrier(Protocol):
    pair_force_scaling: PairForceScaling
    scaling_population_count: int | None
    minimum_supported_distance_m: float
    continuum_ready: bool


def validate_density_convention(value: DensityConvention) -> DensityConvention:
    if not isinstance(value, DensityConvention):
        raise TypeError("density_convention must be a DensityConvention")
    return value


def validate_population_count(
    value: int | None,
    *,
    required: bool,
) -> int | None:
    if value is None:
        if required:
            raise ValueError("number-density convention requires population_count")
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("population_count must be a positive integer")
    count = int(value)
    if count <= 0:
        raise ValueError("population_count must be a positive integer")
    return count


def expected_pair_force_scaling(
    density_convention: DensityConvention,
) -> PairForceScaling:
    convention = validate_density_convention(density_convention)
    if convention is DensityConvention.PROBABILITY:
        return PairForceScaling.KAC_NORMALIZED_PROBABILITY
    return PairForceScaling.UNSCALED_SINGLE_PAIR


def expected_density_mass(
    density_convention: DensityConvention,
    population_count: int | None,
) -> float:
    convention = validate_density_convention(density_convention)
    count = validate_population_count(
        population_count,
        required=convention is DensityConvention.NUMBER,
    )
    return 1.0 if convention is DensityConvention.PROBABILITY else float(count)


def pair_force_scaling_of(potential: object) -> PairForceScaling:
    scaling = getattr(potential, "pair_force_scaling", None)
    if isinstance(scaling, PairForceScaling):
        return scaling
    semantics = getattr(potential, "scaling_semantics", None)
    try:
        return PairForceScaling(semantics)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "pair potential must declare a recognized pair_force_scaling"
        ) from error


def validate_pair_potential_scaling(
    potential: object,
    density_convention: DensityConvention,
    *,
    population_count: int | None,
) -> PairForceScaling:
    convention = validate_density_convention(density_convention)
    runtime_count = validate_population_count(
        population_count,
        required=convention is DensityConvention.NUMBER,
    )
    actual = pair_force_scaling_of(potential)
    expected = expected_pair_force_scaling(convention)
    if actual is not expected:
        raise ValueError(
            "density/potential scaling mismatch: "
            f"{convention.name} requires {expected.name}, got {actual.name}"
        )
    declared_count = getattr(potential, "scaling_population_count", None)
    if declared_count is not None:
        declared_count = validate_population_count(declared_count, required=True)
        if runtime_count is not None and declared_count != runtime_count:
            raise ValueError(
                "Kac scaling population_count does not match the runtime population"
            )
    return actual


def require_continuum_ready(potential: object) -> None:
    """Reject a pair backend that cannot evaluate the zero-displacement kernel."""

    ready = getattr(potential, "continuum_ready", None)
    minimum = getattr(potential, "minimum_supported_distance_m", None)
    if not isinstance(ready, (bool, np.bool_)):
        raise TypeError("pair potential must explicitly declare continuum_ready")
    try:
        minimum_distance = float(minimum)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "pair potential must declare minimum_supported_distance_m"
        ) from error
    if not np.isfinite(minimum_distance) or minimum_distance < 0.0:
        raise ValueError("minimum_supported_distance_m must be finite and non-negative")
    if not bool(ready) or minimum_distance > 0.0:
        raise ValueError(
            "continuum convolution requires a physically defined pair potential "
            "at zero displacement; this particle-only backend has no short-range "
            "closure down to r=0"
        )
