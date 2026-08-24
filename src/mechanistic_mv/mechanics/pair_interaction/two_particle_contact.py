"""Explicit stop boundary for the unavailable two-hydrogel contact FEM."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..hydrogel.equilibrium import TimeScaleAssessment
from ..hydrogel.parameters import HydrogelParameters
from .data_model import PairForceTable
from .geometry import TwoSphereGeometry


class TwoParticleContactSolver(Protocol):
    """Interface a future fully specified contact/bath solver must implement."""

    def solve_pair_force_table(
        self,
        geometry: TwoSphereGeometry,
        parameters: HydrogelParameters,
        center_distances_m: np.ndarray,
        time_scale: TimeScaleAssessment,
    ) -> PairForceTable: ...


class HydrogelContactFEMNotAvailable:
    """Scientifically honest placeholder; it never fabricates contact forces."""

    def solve_pair_force_table(
        self,
        geometry: TwoSphereGeometry,
        parameters: HydrogelParameters,
        center_distances_m: np.ndarray,
        time_scale: TimeScaleAssessment,
    ) -> PairForceTable:
        del geometry, parameters, center_distances_m, time_scale
        raise NotImplementedError(
            "two-hydrogel contact FEM is unavailable: particle geometry, mesh, "
            "frictionless-contact discretization, solvent bath conditions, and "
            "calibrated material values were not supplied"
        )
