"""Explicit stop boundary for the unavailable two-hydrogel contact FEM."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..hydrogel.equilibrium import TimeScaleAssessment
from ..hydrogel.parameters import HydrogelParameters
from .contact_problem import ContactProblemInput
from .data_model import PairContactSweep
from .geometry import TwoSphereGeometry


class TwoParticleContactSolver(Protocol):
    """Legacy incomplete solver interface retained for compatibility."""

    def solve_contact_sweep(
        self,
        geometry: TwoSphereGeometry,
        parameters: HydrogelParameters,
        center_distances_m: np.ndarray,
        time_scale: TimeScaleAssessment,
    ) -> PairContactSweep:
        """Return every requested distance, including explicit failed records."""

        ...


class ContactProblemSolver(Protocol):
    """Preferred interface for a future real, fully specified contact solver."""

    def solve_contact_problem(
        self, problem: ContactProblemInput
    ) -> PairContactSweep:
        """Consume one readiness-gated input and return every requested result."""

        ...


class HydrogelContactFEMNotAvailable:
    """Scientifically honest placeholder; it never fabricates contact forces."""

    def solve_contact_problem(
        self, problem: ContactProblemInput
    ) -> PairContactSweep:
        if not isinstance(problem, ContactProblemInput):
            raise TypeError("problem must be ContactProblemInput")
        readiness = problem.evaluate_readiness()
        if not readiness.can_submit_to_solver:
            missing = ", ".join(readiness.missing_inputs)
            unverified = ", ".join(readiness.unverified_inputs)
            details = "; ".join(
                item
                for item in (
                    f"missing={missing}" if missing else "",
                    f"unverified={unverified}" if unverified else "",
                )
                if item
            )
            raise ValueError(f"contact problem input is BLOCKED: {details}")
        raise NotImplementedError(
            "two-hydrogel contact FEM implementation is unavailable; the "
            "complete input generated no contact samples"
        )

    def solve_contact_sweep(
        self,
        geometry: TwoSphereGeometry,
        parameters: HydrogelParameters,
        center_distances_m: np.ndarray,
        time_scale: TimeScaleAssessment,
    ) -> PairContactSweep:
        del geometry, parameters, center_distances_m, time_scale
        raise NotImplementedError(
            "two-hydrogel contact FEM is unavailable: particle geometry, mesh, "
            "frictionless-contact discretization, solvent bath conditions, and "
            "calibrated material values were not supplied"
        )
