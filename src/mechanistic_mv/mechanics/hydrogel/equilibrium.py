"""Auditable quasi-static time-scale decision for pair reduction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class TimeScaleStatus(str, Enum):
    UNVERIFIED = "UNVERIFIED"
    SATISFIED = "SATISFIED"
    VIOLATED = "VIOLATED"


@dataclass(frozen=True, slots=True)
class TimeScaleAssessment:
    status: TimeScaleStatus
    tau_gel_s: float | None
    tau_swarm_s: float | None
    required_max_ratio: float | None
    ratio: float | None
    reason: str

    @property
    def quasi_static_pair_reduction_verified(self) -> bool:
        return self.status is TimeScaleStatus.SATISFIED


def assess_time_scale_separation(
    *,
    tau_gel_s: float | None,
    tau_swarm_s: float | None,
    required_max_ratio: float | None,
) -> TimeScaleAssessment:
    """Assess ``tau_gel << tau_swarm`` without inventing a threshold.

    All three numerical inputs must be supplied before the result can pass.
    ``required_max_ratio`` is the caller's explicit operational definition of
    ``<<`` and must lie strictly between zero and one.
    """

    for name, value in (("tau_gel_s", tau_gel_s), ("tau_swarm_s", tau_swarm_s)):
        if value is not None and (not np.isfinite(value) or value <= 0.0):
            raise ValueError(f"{name} must be finite and positive when supplied")
    if required_max_ratio is not None and (
        not np.isfinite(required_max_ratio)
        or not 0.0 < required_max_ratio < 1.0
    ):
        raise ValueError("required_max_ratio must lie strictly between zero and one")

    ratio = None
    if tau_gel_s is not None and tau_swarm_s is not None:
        ratio = float(tau_gel_s / tau_swarm_s)
    if tau_gel_s is None or tau_swarm_s is None:
        return TimeScaleAssessment(
            TimeScaleStatus.UNVERIFIED,
            tau_gel_s,
            tau_swarm_s,
            required_max_ratio,
            ratio,
            "gel or swarm time scale is missing",
        )
    if required_max_ratio is None:
        return TimeScaleAssessment(
            TimeScaleStatus.UNVERIFIED,
            tau_gel_s,
            tau_swarm_s,
            None,
            ratio,
            "no quantitative separation criterion was supplied",
        )
    status = (
        TimeScaleStatus.SATISFIED
        if ratio <= required_max_ratio
        else TimeScaleStatus.VIOLATED
    )
    return TimeScaleAssessment(
        status,
        tau_gel_s,
        tau_swarm_s,
        required_max_ratio,
        ratio,
        "ratio satisfies the explicit criterion"
        if status is TimeScaleStatus.SATISFIED
        else "ratio exceeds the explicit criterion",
    )
