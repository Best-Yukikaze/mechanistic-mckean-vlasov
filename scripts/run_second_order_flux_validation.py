"""Validate an optional second-order, positivity-preserving continuum flux.

All forcing, parameters, and analytic comparisons in this program are
``TEST_ONLY_NOT_FINAL_PHYSICS`` numerical fixtures.  The program never
implements a flux itself: it compares only schemes exposed by the Physics
Engine through ``McKeanVlasovSolver``.  If that API is not present or cannot
unambiguously identify a first-order reference and a second-order candidate,
it records a fail-closed ``BLOCKED`` report instead of inventing a scheme.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import inspect
import json
from math import erfc, sqrt
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np

from mechanistic_mv.continuum.convolution import (
    FFTPairConvolver,
    direct_pair_convolution_joule,
)
from mechanistic_mv.continuum.diagnostics import relative_l2_error
from mechanistic_mv.continuum.initial_conditions import gaussian_density
from mechanistic_mv.continuum.mckean_vlasov import McKeanVlasovSolver
from mechanistic_mv.mechanics.controlled_potential import TestOnlyUniformFieldPotential
from mechanistic_mv.mechanics.geometry import (
    CartesianGrid,
    RectangleObstacle,
    RectangularDomain,
)
from mechanistic_mv.mechanics.pair_potential import (
    PairPotential,
    TestOnlyGaussianRepulsion,
    ZeroPairPotential,
)
from mechanistic_mv.mechanics.parameters import PhysicalParameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "validation"
    / "second_order_flux_validation.json"
)
SCHEMA_VERSION = 3
MODEL_STATUS = "TEST_ONLY_NOT_FINAL_PHYSICS"
BLOCKED_NEW_FLUX_API_UNAVAILABLE = "BLOCKED_NEW_FLUX_API_UNAVAILABLE"
SPATIAL_GRID_SIZES = (32, 64, 128)
# The candidate is advanced with SSP-RK2.  Holding the final time fixed and
# increasing this count by four on every h -> h/2 refinement makes
# dt = O(h**2), hence its global temporal truncation target is O(dt**2) =
# O(h**4), below a second-order spatial contribution.  The requested steps
# are also deliberately smaller than the observed outgoing-rate CFL bound so
# that the solver does not choose different hidden substep counts per grid.
SPATIAL_REFINEMENT_BASE_STEPS = 32
# Each value produces a distinct adaptive substep count for the TEST_ONLY
# scenario, so this is a genuine CFL-refinement sequence rather than repeated
# measurements of an identical local time grid.
ADAPTIVE_CFL_SAFETIES = (0.8, 0.15, 0.075, 0.0375)
# These deliberately smooth, non-calibrated values exercise the nonzero
# convolution path without being presented as a Hydrogel contact model.  They
# are property-gate fixtures only; no analytic Gaussian reference is used for
# either of the scenarios below.
NONANALYTIC_GRID_SIZE = 48
NONANALYTIC_OUTER_STEPS = 8
SMOOTH_TEST_PAIR_ENERGY_IN_THERMAL_UNITS = 4.0
SMOOTH_TEST_PAIR_LENGTH_SCALE_M = 1.4e-6
DIRECT_FFT_RELATIVE_TOLERANCE = 2.0e-12


@dataclass(frozen=True, slots=True)
class _SchemeBinding:
    identifier: str
    constructor_keyword: str | None
    value: object | None
    source: str

    def to_dict(self) -> dict[str, object]:
        value = self.value
        if isinstance(value, Enum):
            rendered: object = {"name": value.name, "value": value.value}
        else:
            rendered = value
        return {
            "identifier": self.identifier,
            "constructor_keyword": self.constructor_keyword,
            "value": rendered,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class _FluxApiDiscovery:
    reference: _SchemeBinding
    candidate: _SchemeBinding | None
    available_options: tuple[str, ...]
    status: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "reference": self.reference.to_dict(),
            "candidate": None if self.candidate is None else self.candidate.to_dict(),
            "available_options": list(self.available_options),
        }


@dataclass(frozen=True, slots=True)
class _TestOnlyScenario:
    domain: RectangularDomain
    initial_mean_m: tuple[float, float]
    initial_standard_deviation_m: float
    maximum_force_newton: float
    control: tuple[float, float]
    final_time_s: float
    outer_steps: int

    @property
    def requested_outer_dt_s(self) -> float:
        return self.final_time_s / self.outer_steps


@dataclass(frozen=True, slots=True)
class _SpatialTimePlan:
    """Explicit time plan used to separate spatial and SSP-RK2 errors."""

    grid_size: int
    dx_m: float
    dy_m: float
    requested_step_count: int
    requested_dt_s: float
    dt_over_h_squared_s_per_m2: float

    def to_dict(self) -> dict[str, object]:
        return {
            "grid_size": self.grid_size,
            "dx_m": self.dx_m,
            "dy_m": self.dy_m,
            "requested_step_count": self.requested_step_count,
            "requested_dt_s": self.requested_dt_s,
            "dt_over_h_squared_s_per_m2": self.dt_over_h_squared_s_per_m2,
            "scaling": "dt = O(h^2) at fixed final time",
            "SSP_RK2_temporal_error_design": (
                "with SSP-RK2, global temporal truncation is targeted as "
                "O(dt^2) = O(h^4), below the measured spatial O(h^2) term"
            ),
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reference-scheme",
        help=(
            "explicit enum name/value for the first-order reference when the "
            "Physics API exposes an ambiguous scheme enum"
        ),
    )
    parser.add_argument(
        "--candidate-scheme",
        help=(
            "explicit enum name/value for the second-order candidate when the "
            "Physics API exposes an ambiguous scheme enum"
        ),
    )
    return parser


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={PROJECT_ROOT.as_posix()}",
                "rev-parse",
                "HEAD",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _git_dirty() -> bool | None:
    try:
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={PROJECT_ROOT.as_posix()}",
                "status",
                "--porcelain",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return bool(completed.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _minimal_failed_report(reason: str) -> dict[str, object]:
    """Return a safe artifact when the detailed report cannot be serialized.

    The original payload is intentionally not embedded: it may contain NaN or
    infinity, and preserving it would turn a failed numerical validation into
    a second report-writing failure.
    """

    return {
        "schema_name": "mechanistic_mv.second_order_flux_validation",
        "schema_version": SCHEMA_VERSION,
        "model_status": MODEL_STATUS,
        "physical_status": MODEL_STATUS,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision_at_run": _git_revision(),
        "git_dirty_at_run": _git_dirty(),
        "workflow_status": "FAILED",
        "comparison_status": "FAILED_REPORT_SERIALIZATION",
        "overall_passed": False,
        "all_json_values_finite": False,
        "reason": reason,
        "interpretation": (
            "the detailed TEST_ONLY numerical report was rejected; no physical "
            "validation claim can be made"
        ),
    }


def _write_json(path: Path, payload: dict[str, object]) -> bool:
    """Write a report, falling back to a finite failed artifact when required.

    ``True`` means the supplied detailed report was written.  ``False`` means
    the artifact is an intentionally minimal failure report, which callers
    must propagate as a nonzero validation result.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    report_to_write = payload
    detailed_report_written = True
    if not _finite_json_values(payload):
        report_to_write = _minimal_failed_report(
            "detailed report contained a non-finite numeric value; raw values "
            "were omitted and validation is FAILED"
        )
        detailed_report_written = False
    try:
        serialized = json.dumps(report_to_write, indent=2, allow_nan=False)
    except (TypeError, ValueError, OverflowError):
        report_to_write = _minimal_failed_report(
            "detailed report was not safely serializable; raw values were "
            "omitted and validation is FAILED"
        )
        serialized = json.dumps(report_to_write, indent=2, allow_nan=False)
        detailed_report_written = False
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)
    return detailed_report_written


def _enum_members(parameter: inspect.Parameter) -> tuple[type[Enum], ...]:
    default = parameter.default
    if isinstance(default, Enum):
        return (type(default),)
    return ()


def _select_member(
    members: tuple[Enum, ...], requested: str | None, *, role: str
) -> Enum | None:
    if requested is not None:
        matches = [
            member
            for member in members
            if requested == member.name or requested == str(member.value)
        ]
        if len(matches) != 1:
            return None
        return matches[0]

    def matches_role(member: Enum) -> bool:
        label = f"{member.name} {member.value}".casefold()
        if role == "reference":
            return "first" in label or "upwind" in label
        return any(token in label for token in ("second", "tvd", "muscl", "high"))

    candidates = [member for member in members if matches_role(member)]
    return candidates[0] if len(candidates) == 1 else None


def discover_flux_api(
    solver_type: type[object] = McKeanVlasovSolver,
    *,
    reference_scheme: str | None = None,
    candidate_scheme: str | None = None,
) -> _FluxApiDiscovery:
    """Discover a typed, selectable Physics flux API without guessing one."""

    legacy_reference = _SchemeBinding(
        identifier="FIRST_ORDER_UPWIND_LEGACY_DEFAULT",
        constructor_keyword=None,
        value=None,
        source="current solver default documented in continuum.flux",
    )
    signature = inspect.signature(solver_type)
    parameter = next(
        (
            item
            for item in signature.parameters.values()
            if item.name
            in {"drift_flux_scheme", "flux_scheme", "flux_discretization"}
        ),
        None,
    )
    if parameter is None:
        return _FluxApiDiscovery(
            reference=legacy_reference,
            candidate=None,
            available_options=(),
            status=BLOCKED_NEW_FLUX_API_UNAVAILABLE,
            reason=(
                "McKeanVlasovSolver exposes no selectable drift_flux_scheme, "
                "flux_scheme, or flux_discretization parameter; only the "
                "documented legacy first-order upwind reference can be measured"
            ),
        )

    enum_types = _enum_members(parameter)
    if len(enum_types) != 1:
        return _FluxApiDiscovery(
            reference=legacy_reference,
            candidate=None,
            available_options=(),
            status=BLOCKED_NEW_FLUX_API_UNAVAILABLE,
            reason=(
                f"{parameter.name} is not an Enum-valued selectable scheme; "
                "provide a typed Enum Physics API before comparison"
            ),
        )
    members = tuple(enum_types[0])
    options = tuple(
        sorted({member.name for member in members} | {str(member.value) for member in members})
    )
    reference = _select_member(members, reference_scheme, role="reference")
    candidate = _select_member(members, candidate_scheme, role="candidate")
    if reference is None or candidate is None or reference is candidate:
        details = []
        if reference is None:
            details.append("first-order reference is absent or ambiguous")
        if candidate is None:
            details.append("second-order candidate is absent or ambiguous")
        if reference is candidate and reference is not None:
            details.append("reference and candidate resolve to the same scheme")
        return _FluxApiDiscovery(
            reference=legacy_reference,
            candidate=None,
            available_options=options,
            status=BLOCKED_NEW_FLUX_API_UNAVAILABLE,
            reason="; ".join(details) + "; pass explicit enum names after Physics API review",
        )
    return _FluxApiDiscovery(
        reference=_SchemeBinding(
            identifier=reference.name,
            constructor_keyword=parameter.name,
            value=reference,
            source="Physics Engine selectable Enum",
        ),
        candidate=_SchemeBinding(
            identifier=candidate.name,
            constructor_keyword=parameter.name,
            value=candidate,
            source="Physics Engine selectable Enum",
        ),
        available_options=options,
        status="READY_FOR_TEST_ONLY_COMPARISON",
        reason="typed first-order and second-order scheme selections were discovered",
    )


def _default_scenario() -> _TestOnlyScenario:
    return _TestOnlyScenario(
        domain=RectangularDomain((0.0, 20.0e-6), (0.0, 20.0e-6)),
        initial_mean_m=(9.5e-6, 10.0e-6),
        initial_standard_deviation_m=1.2e-6,
        # A nonzero, moderate TEST_ONLY drift keeps the 32-cell grid inside
        # the smooth spatial-refinement regime.  It has no physical meaning.
        maximum_force_newton=2.0e-14,
        control=(1.0, 0.0),
        final_time_s=0.08,
        outer_steps=8,
    )


def _analytic_test_only_density(
    grid: CartesianGrid,
    parameters: PhysicalParameters,
    scenario: _TestOnlyScenario,
) -> np.ndarray:
    mean, standard_deviation = _analytic_test_only_state(parameters, scenario)
    return gaussian_density(grid, mean, standard_deviation)


def _analytic_test_only_state(
    parameters: PhysicalParameters,
    scenario: _TestOnlyScenario,
) -> tuple[tuple[float, float], float]:
    """Return the free-space Gaussian state used only as a TEST_ONLY reference."""

    velocity_x = (
        parameters.mobility_m_per_newton_second * scenario.maximum_force_newton
    )
    standard_deviation = float(
        np.sqrt(
            scenario.initial_standard_deviation_m**2
            + 2.0 * parameters.diffusion_m2_per_s * scenario.final_time_s
        )
    )
    return (
        (
            scenario.initial_mean_m[0] + velocity_x * scenario.final_time_s,
            scenario.initial_mean_m[1],
        ),
        standard_deviation,
    )


def _boundary_tail(
    mean_m: tuple[float, float],
    standard_deviation_m: float,
    domain: RectangularDomain,
) -> dict[str, object]:
    """Bound the free-space Gaussian tail omitted by the closed-wall problem."""

    if standard_deviation_m <= 0.0:
        raise ValueError("Gaussian standard deviation must be positive")
    distances = {
        "left": mean_m[0] - domain.x_limits_m[0],
        "right": domain.x_limits_m[1] - mean_m[0],
        "bottom": mean_m[1] - domain.y_limits_m[0],
        "top": domain.y_limits_m[1] - mean_m[1],
    }
    sigma_distances = {
        wall: distance / standard_deviation_m
        for wall, distance in distances.items()
    }
    one_sided_tails = {
        wall: 0.5 * erfc(distance / (sqrt(2.0) * standard_deviation_m))
        for wall, distance in distances.items()
    }
    return {
        "mean_m": list(mean_m),
        "standard_deviation_m": standard_deviation_m,
        "wall_distance_in_standard_deviations": sigma_distances,
        "one_sided_free_space_tail_probability": one_sided_tails,
        "conservative_probability_mass_outside_rectangle_upper_bound": float(
            sum(one_sided_tails.values())
        ),
        "minimum_wall_distance_in_standard_deviations": float(
            min(sigma_distances.values())
        ),
    }


def _boundary_tail_report(
    parameters: PhysicalParameters,
    scenario: _TestOnlyScenario,
) -> dict[str, object]:
    final_mean, final_standard_deviation = _analytic_test_only_state(
        parameters, scenario
    )
    return {
        "finite_volume_boundary_condition": (
            "closed no-flux outer faces supplied by the continuum solver"
        ),
        "analytic_reference_limitation": (
            "the analytic TEST_ONLY comparator is a free-space drift-diffusion "
            "Gaussian.  It is not an exact reflecting-wall solution; the tail "
            "bounds below quantify why its wall mismatch is negligible here."
        ),
        "initial_time": _boundary_tail(
            scenario.initial_mean_m,
            scenario.initial_standard_deviation_m,
            scenario.domain,
        ),
        "final_time": _boundary_tail(
            final_mean,
            final_standard_deviation,
            scenario.domain,
        ),
    }


def _spatial_time_plan(
    grid: CartesianGrid,
    scenario: _TestOnlyScenario,
) -> _SpatialTimePlan:
    """Build a reproducible dt=O(h**2) schedule for a fixed three-grid study."""

    reference_dx_m = (
        scenario.domain.x_limits_m[1] - scenario.domain.x_limits_m[0]
    ) / SPATIAL_GRID_SIZES[0]
    if not np.isclose(grid.dx_m, grid.dy_m, rtol=0.0, atol=0.0):
        raise ValueError("spatial convergence fixture requires square cells")
    refinement_ratio = reference_dx_m / grid.dx_m
    requested_step_count = int(
        round(SPATIAL_REFINEMENT_BASE_STEPS * refinement_ratio**2)
    )
    if requested_step_count <= 0:
        raise ValueError("spatial time plan must contain at least one step")
    requested_dt_s = scenario.final_time_s / requested_step_count
    return _SpatialTimePlan(
        grid_size=grid.nx,
        dx_m=grid.dx_m,
        dy_m=grid.dy_m,
        requested_step_count=requested_step_count,
        requested_dt_s=requested_dt_s,
        dt_over_h_squared_s_per_m2=requested_dt_s / grid.dx_m**2,
    )


def _scheme_solver(
    grid: CartesianGrid,
    parameters: PhysicalParameters,
    controlled_potential: TestOnlyUniformFieldPotential,
    binding: _SchemeBinding,
    *,
    cfl_safety: float,
    pair_potential: PairPotential | None = None,
    obstacles: tuple[RectangleObstacle, ...] = (),
) -> McKeanVlasovSolver:
    kwargs: dict[str, object] = {
        "controlled_potential": controlled_potential,
        "cfl_safety": cfl_safety,
        "record_fixed_control_free_energy": True,
    }
    if binding.constructor_keyword is not None:
        kwargs[binding.constructor_keyword] = binding.value
    return McKeanVlasovSolver(
        grid,
        parameters,
        ZeroPairPotential() if pair_potential is None else pair_potential,
        obstacles=obstacles,
        **kwargs,
    )


def _evolve_test_only(
    solver: McKeanVlasovSolver,
    grid: CartesianGrid,
    parameters: PhysicalParameters,
    scenario: _TestOnlyScenario,
    *,
    outer_steps: int | None = None,
    initial_density_per_m2: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    if initial_density_per_m2 is None:
        initial = gaussian_density(
            grid,
            scenario.initial_mean_m,
            scenario.initial_standard_deviation_m,
        )
    else:
        initial = np.asarray(initial_density_per_m2, dtype=np.float64).copy()
    density = initial
    initial_mass = solver.mass(density)
    energies = [solver.free_energy(density, np.asarray(scenario.control)).total_joule]
    masses = [initial_mass]
    minimum_density = float(np.min(density[solver.fluid_mask]))
    clipped_negative_mass = 0.0
    fixed_control_energy_changes: list[float] = []
    adaptive_substeps = 0
    minimum_stable_dt_s = np.inf
    requested_outer_steps = scenario.outer_steps if outer_steps is None else outer_steps
    if requested_outer_steps <= 0:
        raise ValueError("outer_steps must be positive")
    requested_outer_dt_s = scenario.final_time_s / requested_outer_steps
    for _ in range(requested_outer_steps):
        density, diagnostics = solver.step(
            density,
            requested_outer_dt_s,
            control=np.asarray(scenario.control),
        )
        adaptive_substeps += diagnostics.substeps
        minimum_stable_dt_s = min(
            minimum_stable_dt_s, diagnostics.minimum_stable_dt_s
        )
        minimum_density = min(minimum_density, diagnostics.minimum_density_per_m2)
        clipped_negative_mass += diagnostics.clipped_negative_mass
        if diagnostics.fixed_control_free_energy_change_joule is None:
            raise RuntimeError(
                "Physics API did not record fixed-control free-energy diagnostics"
            )
        fixed_control_energy_changes.append(
            float(diagnostics.fixed_control_free_energy_change_joule)
        )
        masses.append(solver.mass(density))
        energies.append(
            solver.free_energy(density, np.asarray(scenario.control)).total_joule
        )
    energy_array = np.asarray(energies, dtype=np.float64)
    mass_array = np.asarray(masses, dtype=np.float64)
    return density, {
        "initial_mass": float(initial_mass),
        "final_mass": float(mass_array[-1]),
        "maximum_absolute_mass_error": float(
            np.max(np.abs(mass_array - initial_mass))
        ),
        "minimum_density_per_m2": float(minimum_density),
        "clipped_negative_mass": float(clipped_negative_mass),
        "initial_free_energy_joule": float(energy_array[0]),
        "final_free_energy_joule": float(energy_array[-1]),
        "maximum_energy_increment_joule": float(
            max(0.0, np.max(np.diff(energy_array)))
        ),
        "raw_maximum_energy_increment_joule": float(np.max(np.diff(energy_array))),
        "fixed_control_free_energy_changes_joule": fixed_control_energy_changes,
        "adaptive_substeps": adaptive_substeps,
        "minimum_stable_dt_s": float(minimum_stable_dt_s),
        "outer_steps": requested_outer_steps,
        "requested_outer_dt_s": requested_outer_dt_s,
        "exactly_one_internal_substep_per_requested_step": (
            adaptive_substeps == requested_outer_steps
        ),
        "requested_step_within_observed_CFL_bound": bool(
            requested_outer_dt_s <= minimum_stable_dt_s
        ),
    }


def _integrity_checks(record: dict[str, object]) -> dict[str, bool]:
    return {
        "mass_conservation": float(record["maximum_absolute_mass_error"]) <= 2.0e-12,
        "non_negative_density": float(record["minimum_density_per_m2"]) >= 0.0,
        "no_negative_mass_clipping": float(record["clipped_negative_mass"]) == 0.0,
    }


def _free_energy_behavior(record: dict[str, object], parameters: PhysicalParameters) -> dict[str, object]:
    initial_energy = abs(float(record["initial_free_energy_joule"]))
    energy_scale = max(initial_energy, parameters.thermal_energy_joule)
    tolerance = 2.0e-12 * energy_scale
    maximum_increment = float(record["maximum_energy_increment_joule"])
    return {
        "maximum_increment_joule": maximum_increment,
        "nonincrease_observed_within_test_tolerance": maximum_increment <= tolerance,
        "relative_tolerance": 2.0e-12,
        "absolute_tolerance_joule": tolerance,
        "interpretation": (
            "fixed-control state-functional diagnostic only; not a discrete "
            "free-energy certificate for the selected higher-order candidate"
        ),
    }


def _source_file(symbol: object) -> str:
    """Render a source location for report provenance without assuming a path."""

    source = inspect.getsourcefile(symbol)
    if source is None:
        return "unavailable"
    source_path = Path(source)
    try:
        return source_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(source_path)


def _api_provenance(symbol: object, api_symbol: str) -> dict[str, object]:
    return {
        "api_symbol": api_symbol,
        "module": getattr(symbol, "__module__", type(symbol).__module__),
        "source_file": _source_file(symbol),
        "git_revision_at_run": _git_revision(),
    }


def _is_scharfetter_gummel_binding(binding: _SchemeBinding) -> bool:
    identifier = binding.identifier.casefold()
    return "scharfetter" in identifier and "gummel" in identifier


def _test_only_smooth_pair(
    parameters: PhysicalParameters,
) -> TestOnlyGaussianRepulsion:
    """Build the explicit non-calibrated pair fixture used only below."""

    return TestOnlyGaussianRepulsion(
        SMOOTH_TEST_PAIR_ENERGY_IN_THERMAL_UNITS
        * parameters.thermal_energy_joule,
        SMOOTH_TEST_PAIR_LENGTH_SCALE_M,
    )


def _fixture_labels(
    binding: _SchemeBinding,
    pair_potential: TestOnlyGaussianRepulsion,
    controlled_potential: TestOnlyUniformFieldPotential,
    scenario: _TestOnlyScenario,
) -> dict[str, object]:
    """State explicitly why these numerical fixtures are not material inputs."""

    return {
        "fixture_status": MODEL_STATUS,
        "scheme": binding.to_dict(),
        "control": {
            "name": controlled_potential.name,
            "physical_status": controlled_potential.physical_status,
            "command": list(scenario.control),
            "maximum_force_newton": controlled_potential.maximum_force_newton,
            "contract": "u -> V_control -> -grad(V_control)",
            "interpretation": (
                "uniform-field control is TEST_ONLY_NOT_FINAL_PHYSICS; it is not "
                "a calibrated actuator model"
            ),
        },
        "interaction": {
            "name": pair_potential.name,
            "physical_status": pair_potential.physical_status,
            "energy_scale_joule": pair_potential.energy_scale_joule,
            "length_scale_m": pair_potential.length_scale_m,
            "density_convention": pair_potential.density_convention.value,
            "pair_force_scaling": pair_potential.pair_force_scaling.value,
            "interpretation": (
                "smooth Gaussian repulsion is a hand-selected numerical fixture, "
                "not a Hydrogel-derived W_eff, contact law, or material calibration"
            ),
        },
    }


def _nonanalytic_provenance(
    binding: _SchemeBinding,
) -> dict[str, object]:
    """Provide local API and Git evidence for the non-analytic scenarios."""

    return {
        "git_revision_at_run": _git_revision(),
        "git_dirty_at_run": _git_dirty(),
        "selected_flux_scheme": binding.to_dict(),
        "solver": _api_provenance(
            McKeanVlasovSolver,
            "mechanistic_mv.continuum.mckean_vlasov.McKeanVlasovSolver",
        ),
        "pair_fixture": _api_provenance(
            TestOnlyGaussianRepulsion,
            "mechanistic_mv.mechanics.pair_potential.TestOnlyGaussianRepulsion",
        ),
        "direct_convolution": _api_provenance(
            direct_pair_convolution_joule,
            "mechanistic_mv.continuum.convolution.direct_pair_convolution_joule",
        ),
        "fft_convolution": _api_provenance(
            FFTPairConvolver,
            "mechanistic_mv.continuum.convolution.FFTPairConvolver.convolve_joule",
        ),
    }


def _direct_fft_convolution_agreement(
    solver: McKeanVlasovSolver,
    density_per_m2: np.ndarray,
) -> dict[str, object]:
    """Check the current translation-invariant test fixture by two quadratures."""

    direct = direct_pair_convolution_joule(
        density_per_m2,
        solver.grid,
        solver.pair_potential,
        density_convention=solver.density_convention,
        population_count=solver.population_count,
    )
    accelerated = solver.pair_convolution_joule(density_per_m2)
    absolute_difference = float(np.max(np.abs(direct - accelerated)))
    scale = max(
        float(np.max(np.abs(direct))),
        float(np.max(np.abs(accelerated))),
        np.finfo(np.float64).tiny,
    )
    relative_difference = absolute_difference / scale
    uses_fft_pair_convolver = isinstance(solver.convolver, FFTPairConvolver)
    return {
        "applicable": True,
        "density_state": "final evolved TEST_ONLY density",
        "direct_api": (
            "mechanistic_mv.continuum.convolution.direct_pair_convolution_joule"
        ),
        "fft_api": (
            "McKeanVlasovSolver.pair_convolution_joule -> "
            "FFTPairConvolver.convolve_joule"
        ),
        "solver_uses_fft_pair_convolver": uses_fft_pair_convolver,
        "maximum_absolute_difference_joule": absolute_difference,
        "maximum_relative_difference": relative_difference,
        "relative_tolerance": DIRECT_FFT_RELATIVE_TOLERANCE,
        "passed": bool(
            uses_fft_pair_convolver
            and relative_difference <= DIRECT_FFT_RELATIVE_TOLERANCE
        ),
        "interpretation": (
            "agreement validates two numerical implementations of the same "
            "TEST_ONLY translation-invariant fixture; it does not validate W_eff"
        ),
    }


def _obstacle_no_flux_gate(
    solver: McKeanVlasovSolver,
    density_per_m2: np.ndarray,
    scenario: _TestOnlyScenario,
    binding: _SchemeBinding,
    metrics: dict[str, object],
) -> dict[str, object]:
    """Measure solid-cell and closed-face invariants without an analytic model."""

    observed_stable_dt_s = float(metrics["minimum_stable_dt_s"])
    if not np.isfinite(observed_stable_dt_s) or observed_stable_dt_s <= 0.0:
        raise ValueError("obstacle flux probe requires a positive finite CFL bound")
    probe_dt_s = 0.5 * observed_stable_dt_s
    fluxes, interaction = solver.face_fluxes(
        density_per_m2,
        control=np.asarray(scenario.control),
        dt_s=probe_dt_s,
    )
    closed_x = np.abs(fluxes.x_per_m_s[~solver.face_masks.open_x])
    closed_y = np.abs(fluxes.y_per_m_s[~solver.face_masks.open_y])
    maximum_closed_flux = max(
        float(np.max(closed_x)),
        float(np.max(closed_y)),
    )
    solid_density = np.asarray(density_per_m2)[~solver.fluid_mask]
    maximum_solid_density = float(np.max(solid_density))
    flux_scheme = fluxes.drift_flux_scheme.value
    return {
        "obstacle_cell_count": int(np.count_nonzero(~solver.fluid_mask)),
        "maximum_density_in_solid_per_m2": maximum_solid_density,
        "maximum_absolute_closed_face_flux_per_m_s": maximum_closed_flux,
        "probe_dt_s": probe_dt_s,
        "flux_scheme_reported_by_physics_engine": flux_scheme,
        "selected_scharfetter_gummel_scheme": binding.identifier,
        "maximum_absolute_pair_interaction_joule": float(
            np.max(np.abs(interaction))
        ),
        "solid_cells_remain_exactly_zero": maximum_solid_density == 0.0,
        "closed_outer_and_obstacle_faces_are_exactly_zero": (
            maximum_closed_flux == 0.0
        ),
        "physics_engine_reported_selected_scheme": flux_scheme == binding.identifier,
        "passed": bool(
            maximum_solid_density == 0.0
            and maximum_closed_flux == 0.0
            and flux_scheme == binding.identifier
        ),
        "interpretation": (
            "this is a finite-volume no-flux geometry gate. The obstacle and "
            "Gaussian pair fixture do not establish a real wall-mediated "
            "interaction law."
        ),
    }


def _nonanalytic_scenario_record(
    *,
    name: str,
    solver: McKeanVlasovSolver,
    density_per_m2: np.ndarray,
    metrics: dict[str, object],
    binding: _SchemeBinding,
    pair_potential: TestOnlyGaussianRepulsion,
    controlled_potential: TestOnlyUniformFieldPotential,
    scenario: _TestOnlyScenario,
    analytic_reference_limitation: str,
) -> dict[str, object]:
    """Record a property-only scenario that deliberately has no Gaussian truth."""

    convolution = _direct_fft_convolution_agreement(solver, density_per_m2)
    # The accelerated field itself, rather than the implementation difference,
    # establishes that this is truly a nonzero interaction scenario.
    accelerated_interaction = solver.pair_convolution_joule(density_per_m2)
    maximum_pair_interaction = float(np.max(np.abs(accelerated_interaction)))
    checks = {
        **_integrity_checks(metrics),
        "nonzero_smooth_test_pair_interaction": maximum_pair_interaction > 0.0,
        "direct_fft_convolution_agreement": bool(convolution["passed"]),
        "selected_scheme_is_scharfetter_gummel": _is_scharfetter_gummel_binding(
            binding
        ),
        "no_free_space_gaussian_reference_used": True,
        "fixture_is_explicitly_test_only": (
            pair_potential.physical_status == MODEL_STATUS
            and controlled_potential.physical_status == MODEL_STATUS
        ),
    }
    return {
        "scenario_name": name,
        "scenario_status": MODEL_STATUS,
        "acceptance_method": "property gates and direct/FFT agreement only",
        "analytic_reference": {
            "used": False,
            "limitation": analytic_reference_limitation,
        },
        "labels": _fixture_labels(
            binding, pair_potential, controlled_potential, scenario
        ),
        "provenance": _nonanalytic_provenance(binding),
        **metrics,
        "maximum_absolute_pair_interaction_joule": maximum_pair_interaction,
        "direct_fft_convolution": convolution,
        "integrity_checks": _integrity_checks(metrics),
        "scenario_checks": checks,
        "all_numerical_property_gates_pass": all(checks.values()),
        "interpretation": (
            "passing means only that this TEST_ONLY numerical scenario met its "
            "discrete gates; it is not a real physical-model validation"
        ),
    }


def _run_nonanalytic_sg_scenarios(
    binding: _SchemeBinding,
    parameters: PhysicalParameters,
    scenario: _TestOnlyScenario,
) -> dict[str, dict[str, object]]:
    """Run nonzero-pair and obstacle SG property gates without analytic misuse."""

    if not _is_scharfetter_gummel_binding(binding):
        raise ValueError(
            "non-analytic scenarios require an explicitly named "
            "SECOND_ORDER_SCHARFETTER_GUMMEL Physics Engine candidate"
        )
    grid = CartesianGrid(
        scenario.domain,
        NONANALYTIC_GRID_SIZE,
        NONANALYTIC_GRID_SIZE,
    )
    pair_potential = _test_only_smooth_pair(parameters)
    uniform_control = TestOnlyUniformFieldPotential(scenario.maximum_force_newton)

    pair_solver = _scheme_solver(
        grid,
        parameters,
        uniform_control,
        binding,
        cfl_safety=0.8,
        pair_potential=pair_potential,
    )
    pair_density, pair_metrics = _evolve_test_only(
        pair_solver,
        grid,
        parameters,
        scenario,
        outer_steps=NONANALYTIC_OUTER_STEPS,
    )
    pair_record = _nonanalytic_scenario_record(
        name="smooth_nonzero_pair_SG_property_gate",
        solver=pair_solver,
        density_per_m2=pair_density,
        metrics=pair_metrics,
        binding=binding,
        pair_potential=pair_potential,
        controlled_potential=uniform_control,
        scenario=scenario,
        analytic_reference_limitation=(
            "nonzero W makes free-space drift-diffusion Gaussian formulas "
            "inapplicable, so no analytic error or convergence order is reported"
        ),
    )

    obstacle = RectangleObstacle((9.5e-6, 11.5e-6), (7.0e-6, 13.0e-6))
    obstacle_control = TestOnlyUniformFieldPotential(8.0e-14)
    obstacle_solver = _scheme_solver(
        grid,
        parameters,
        obstacle_control,
        binding,
        cfl_safety=0.8,
        pair_potential=pair_potential,
        obstacles=(obstacle,),
    )
    obstacle_initial = gaussian_density(
        grid,
        (7.5e-6, 10.0e-6),
        1.0e-6,
        fluid_mask=obstacle_solver.fluid_mask,
    )
    obstacle_density, obstacle_metrics = _evolve_test_only(
        obstacle_solver,
        grid,
        parameters,
        scenario,
        outer_steps=NONANALYTIC_OUTER_STEPS,
        initial_density_per_m2=obstacle_initial,
    )
    obstacle_record = _nonanalytic_scenario_record(
        name="smooth_nonzero_pair_with_no_flux_obstacle_SG_property_gate",
        solver=obstacle_solver,
        density_per_m2=obstacle_density,
        metrics=obstacle_metrics,
        binding=binding,
        pair_potential=pair_potential,
        controlled_potential=obstacle_control,
        scenario=scenario,
        analytic_reference_limitation=(
            "both the no-flux obstacle and nonzero W invalidate a free-space "
            "Gaussian comparator; this scenario uses geometry and convolution "
            "property gates only"
        ),
    )
    no_flux_gate = _obstacle_no_flux_gate(
        obstacle_solver,
        obstacle_density,
        scenario,
        binding,
        obstacle_metrics,
    )
    obstacle_record["obstacle_geometry"] = {
        "kind": "RectangleObstacle",
        "x_limits_m": list(obstacle.x_limits_m),
        "y_limits_m": list(obstacle.y_limits_m),
        "physical_status": MODEL_STATUS,
        "interpretation": (
            "test-only no-flux geometry for numerical verification; it is not "
            "a measured Hydrogel or wall geometry"
        ),
    }
    obstacle_record["no_flux_obstacle_gate"] = no_flux_gate
    obstacle_checks = obstacle_record["scenario_checks"]
    if not isinstance(obstacle_checks, dict):
        raise TypeError("internal obstacle scenario checks must be a dictionary")
    obstacle_checks.update(
        {
            "obstacle_contains_solid_cells": (
                no_flux_gate["obstacle_cell_count"] > 0
            ),
            "solid_cells_remain_exactly_zero": bool(
                no_flux_gate["solid_cells_remain_exactly_zero"]
            ),
            "closed_outer_and_obstacle_faces_are_exactly_zero": bool(
                no_flux_gate["closed_outer_and_obstacle_faces_are_exactly_zero"]
            ),
            "physics_engine_reported_selected_sg_scheme": bool(
                no_flux_gate["physics_engine_reported_selected_scheme"]
            ),
        }
    )
    obstacle_record["all_numerical_property_gates_pass"] = all(
        obstacle_checks.values()
    )
    return {
        "smooth_nonzero_pair": pair_record,
        "smooth_nonzero_pair_with_no_flux_obstacle": obstacle_record,
    }


def _nonanalytic_sg_checks(
    scenarios: dict[str, dict[str, object]],
) -> dict[str, bool]:
    return {
        name: bool(record["all_numerical_property_gates_pass"])
        for name, record in scenarios.items()
    }


def _grid_study(
    binding: _SchemeBinding,
    parameters: PhysicalParameters,
    scenario: _TestOnlyScenario,
) -> list[dict[str, object]]:
    potential = TestOnlyUniformFieldPotential(scenario.maximum_force_newton)
    boundary_tail = _boundary_tail_report(parameters, scenario)
    records: list[dict[str, object]] = []
    for size in SPATIAL_GRID_SIZES:
        grid = CartesianGrid(scenario.domain, size, size)
        time_plan = _spatial_time_plan(grid, scenario)
        solver = _scheme_solver(
            grid, parameters, potential, binding, cfl_safety=0.8
        )
        advanced, metrics = _evolve_test_only(
            solver,
            grid,
            parameters,
            scenario,
            outer_steps=time_plan.requested_step_count,
        )
        metrics.update(
            {
                "grid_size": size,
                "spatial_time_refinement": time_plan.to_dict(),
                "boundary_tail_reference": boundary_tail,
                "relative_L2_to_analytic_test_only_solution": relative_l2_error(
                    advanced,
                    _analytic_test_only_density(grid, parameters, scenario),
                    grid,
                ),
                "integrity_checks": _integrity_checks(metrics),
                "free_energy_behavior": _free_energy_behavior(metrics, parameters),
            }
        )
        records.append(metrics)
    for previous, current in zip(records[:-1], records[1:], strict=True):
        previous_error = float(
            previous["relative_L2_to_analytic_test_only_solution"]
        )
        current_error = float(current["relative_L2_to_analytic_test_only_solution"])
        current["observed_order_from_previous_grid"] = float(
            np.log2(previous_error / current_error)
        )
    return records


def _cfl_study(
    binding: _SchemeBinding,
    parameters: PhysicalParameters,
    scenario: _TestOnlyScenario,
) -> list[dict[str, object]]:
    grid = CartesianGrid(scenario.domain, 64, 64)
    potential = TestOnlyUniformFieldPotential(scenario.maximum_force_newton)
    solutions: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    for cfl_safety in ADAPTIVE_CFL_SAFETIES:
        solver = _scheme_solver(
            grid, parameters, potential, binding, cfl_safety=cfl_safety
        )
        advanced, metrics = _evolve_test_only(solver, grid, parameters, scenario)
        solutions.append(advanced)
        metrics.update(
            {
                "cfl_safety": cfl_safety,
                "integrity_checks": _integrity_checks(metrics),
                "free_energy_behavior": _free_energy_behavior(metrics, parameters),
            }
        )
        records.append(metrics)
    reference = solutions[-1]
    for record, solution in zip(records, solutions, strict=True):
        record["relative_L2_to_finest_CFL"] = relative_l2_error(
            solution, reference, grid
        )
    return records


def _run_scheme(
    binding: _SchemeBinding,
    parameters: PhysicalParameters,
    scenario: _TestOnlyScenario,
) -> dict[str, object]:
    grid_records = _grid_study(binding, parameters, scenario)
    cfl_records = _cfl_study(binding, parameters, scenario)
    return {
        "binding": binding.to_dict(),
        "grid_refinement": grid_records,
        "adaptive_CFL_refinement": cfl_records,
    }


def _scheme_checks(study: dict[str, object], *, require_second_order: bool) -> dict[str, bool]:
    grid_records = study["grid_refinement"]
    cfl_records = study["adaptive_CFL_refinement"]
    if not isinstance(grid_records, list) or not isinstance(cfl_records, list):
        raise TypeError("internal validation study must contain record lists")
    grid_errors = np.asarray(
        [record["relative_L2_to_analytic_test_only_solution"] for record in grid_records],
        dtype=np.float64,
    )
    cfl_errors = np.asarray(
        [record["relative_L2_to_finest_CFL"] for record in cfl_records[:-1]],
        dtype=np.float64,
    )
    integrity = [
        passed
        for record in [*grid_records, *cfl_records]
        for passed in record["integrity_checks"].values()
    ]
    spatial_time_checks = [
        bool(record["exactly_one_internal_substep_per_requested_step"])
        and bool(record["requested_step_within_observed_CFL_bound"])
        for record in grid_records
    ]
    planned_dt_over_h_squared = np.asarray(
        [
            record["spatial_time_refinement"]["dt_over_h_squared_s_per_m2"]
            for record in grid_records
        ],
        dtype=np.float64,
    )
    checks = {
        "all_mass_and_positivity_gates_pass": all(integrity),
        "grid_error_decreases": bool(np.all(np.diff(grid_errors) < 0.0)),
        "adaptive_CFL_error_decreases": bool(np.all(np.diff(cfl_errors) < 0.0)),
        "spatial_time_schedule_is_h_squared_and_single_substep": bool(
            np.allclose(
                planned_dt_over_h_squared,
                planned_dt_over_h_squared[0],
                rtol=1.0e-14,
                atol=0.0,
            )
            and all(spatial_time_checks)
        ),
    }
    if require_second_order:
        checks["fine_grid_observed_order_at_least_1p8"] = float(
            grid_records[-1]["observed_order_from_previous_grid"]
        ) >= 1.8
    return checks


def _finite_json_values(value: object) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_json_values(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_json_values(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def _candidate_method_metadata(discovery: _FluxApiDiscovery) -> dict[str, object] | None:
    """Record the actual Physics symbol; do not restate its numerical formula."""

    if discovery.candidate is None or not isinstance(discovery.candidate.value, Enum):
        return None
    scheme = discovery.candidate.value
    enum_type = type(scheme)
    source_file = inspect.getsourcefile(enum_type)
    if source_file is not None:
        source_path = Path(source_file)
        try:
            rendered_path = source_path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            rendered_path = str(source_path)
    else:
        rendered_path = "unavailable"
    solver_source = inspect.getsourcefile(McKeanVlasovSolver)
    if solver_source is not None:
        solver_path = Path(solver_source)
        try:
            rendered_solver_path = solver_path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            rendered_solver_path = str(solver_path)
    else:
        rendered_solver_path = "unavailable"
    return {
        "scheme_name": scheme.name,
        "scheme_value": scheme.value,
        "formula_source": {
            "module": enum_type.__module__,
            "enum_symbol": f"{enum_type.__name__}.{scheme.name}",
            "flux_assembly_symbol": (
                "mechanistic_mv.continuum.flux.compute_face_fluxes"
            ),
            "solver_integration_symbol": (
                "mechanistic_mv.continuum.mckean_vlasov.McKeanVlasovSolver"
            ),
            "source_file": rendered_path,
            "source_revision_at_run": _git_revision(),
        },
        "time_step_contract_source": {
            "solver_file": rendered_solver_path,
            "flux_api_symbol": "McKeanVlasovSolver.face_fluxes",
            "CFL_api_symbol": "McKeanVlasovSolver.stable_dt_s",
            "step_api_symbol": "McKeanVlasovSolver.step",
            "flux_api_documentation": inspect.getdoc(
                McKeanVlasovSolver.face_fluxes
            ),
            "CFL_api_documentation": inspect.getdoc(
                McKeanVlasovSolver.stable_dt_s
            ),
        },
        "interpretation": (
            "The Experiment Lab records the Physics Engine scheme identifier and "
            "source location, then measures its output. It does not duplicate, "
            "infer, or rename the numerical reconstruction formula."
        ),
        "spatial_accuracy_claim": (
            "The report tests a fine-grid spatial-order gate; it makes no formal "
            "claim about global temporal order."
        ),
    }


def _base_report(
    discovery: _FluxApiDiscovery,
    parameters: PhysicalParameters,
    scenario: _TestOnlyScenario,
    command: list[str],
) -> dict[str, object]:
    return {
        "schema_name": "mechanistic_mv.second_order_flux_validation",
        "schema_version": SCHEMA_VERSION,
        "model_status": MODEL_STATUS,
        "physical_status": MODEL_STATUS,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision_at_run": _git_revision(),
        "git_dirty_at_run": _git_dirty(),
        "command": command,
        "runtime": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "python": sys.version,
            "numpy": np.__version__,
        },
        "scope": {
            "claim": (
                "numerical comparison only; not Hydrogel calibration or final "
                "physical-model validation"
            ),
            "candidate_method": _candidate_method_metadata(discovery),
            "uses_real_physical_parameters": False,
            "uses_real_pair_potential": False,
            "uses_test_only_uniform_control": True,
            "planned_nonanalytic_sg_scope": (
                "when the named Scharfetter--Gummel candidate is available, run "
                "a smooth nonzero TEST_ONLY pair fixture and a TEST_ONLY no-flux "
                "obstacle fixture using property gates only"
            ),
        },
        "scheme_api": discovery.to_dict(),
        "test_only_configuration": {
            "physical_parameters": parameters.as_dict(),
            "domain_x_limits_m": list(scenario.domain.x_limits_m),
            "domain_y_limits_m": list(scenario.domain.y_limits_m),
            "initial_mean_m": list(scenario.initial_mean_m),
            "initial_standard_deviation_m": scenario.initial_standard_deviation_m,
            "uniform_force_newton": scenario.maximum_force_newton,
            "control": list(scenario.control),
            "final_time_s": scenario.final_time_s,
            "adaptive_CFL_study_outer_steps": scenario.outer_steps,
            "spatial_refinement_time_design": {
                "grid_sizes": list(SPATIAL_GRID_SIZES),
                "step_counts": [
                    SPATIAL_REFINEMENT_BASE_STEPS * 4**index
                    for index in range(len(SPATIAL_GRID_SIZES))
                ],
                "formula": (
                    "dt(h) = dt(h_32) * (h / h_32)^2 at fixed final time; "
                    "SSP-RK2 therefore targets global temporal error O(h^4)"
                ),
                "requires_one_internal_substep": True,
            },
            "boundary_tail_reference": _boundary_tail_report(parameters, scenario),
            "adaptive_CFL_safeties": list(ADAPTIVE_CFL_SAFETIES),
            "pair_potential": "ZeroPairPotential",
            "analytic_reference": (
                "constant-force drift plus Gaussian diffusion, discretely "
                "normalized on every validation grid"
            ),
        },
        "acceptance_thresholds": {
            "maximum_absolute_mass_error": 2.0e-12,
            "minimum_density_per_m2": 0.0,
            "clipped_negative_mass": 0.0,
            "candidate_fine_grid_observed_order": 1.8,
            "candidate_fine_grid_observed_order_rationale": (
                "1.8 is a pre-set near-second-order acceptance line. It leaves "
                "headroom for finite-grid and negligible closed-boundary/reference "
                "effects, without setting the gate from this run's measured order."
            ),
        },
        "diagnostic_interpretation": {
            "free_energy_behavior": (
                "reported at fixed test-only control, but excluded from pass/fail "
                "because this experiment does not assume a discrete free-energy "
                "certificate for the selected higher-order candidate"
            ),
            "time_refinement": (
                "adaptive CFL-safety refinement, not a claim of a formal temporal "
                "convergence order"
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    parameters = PhysicalParameters()
    scenario = _default_scenario()
    discovery = discover_flux_api(
        reference_scheme=args.reference_scheme,
        candidate_scheme=args.candidate_scheme,
    )
    report = _base_report(discovery, parameters, scenario, command)

    try:
        reference_study = _run_scheme(discovery.reference, parameters, scenario)
        report["first_order_reference"] = reference_study
        report["first_order_reference_checks"] = _scheme_checks(
            reference_study, require_second_order=False
        )
        if discovery.candidate is None:
            report.update(
                {
                    "workflow_status": "BLOCKED",
                    "comparison_status": discovery.status,
                    "candidate_scheme": None,
                    "overall_passed": False,
                    "reason": discovery.reason,
                }
            )
            report["all_json_values_finite"] = _finite_json_values(report)
            _write_json(args.output, report)
            print(args.output)
            return 2

        candidate_study = _run_scheme(discovery.candidate, parameters, scenario)
        candidate_checks = _scheme_checks(candidate_study, require_second_order=True)
        nonanalytic_scenarios = _run_nonanalytic_sg_scenarios(
            discovery.candidate,
            parameters,
            scenario,
        )
        nonanalytic_checks = _nonanalytic_sg_checks(nonanalytic_scenarios)
        reference_grid = reference_study["grid_refinement"]
        candidate_grid = candidate_study["grid_refinement"]
        if not isinstance(reference_grid, list) or not isinstance(candidate_grid, list):
            raise TypeError("internal grid records must be lists")
        comparison_checks = {
            "candidate_finest_grid_error_not_larger_than_reference": float(
                candidate_grid[-1]["relative_L2_to_analytic_test_only_solution"]
            )
            <= float(reference_grid[-1]["relative_L2_to_analytic_test_only_solution"])
        }
        overall_passed = all(
            [
                *report["first_order_reference_checks"].values(),
                *candidate_checks.values(),
                *nonanalytic_checks.values(),
                *comparison_checks.values(),
            ]
        )
        report.update(
            {
                "workflow_status": (
                    "COMPLETED_TEST_ONLY"
                    if overall_passed
                    else "FAILED_TEST_ONLY_GATES"
                ),
                "comparison_status": (
                    "COMPLETED_TEST_ONLY"
                    if overall_passed
                    else "FAILED_TEST_ONLY_GATES"
                ),
                "candidate_scheme": candidate_study,
                "candidate_scheme_checks": candidate_checks,
                "nonanalytic_sg_scenarios": nonanalytic_scenarios,
                "nonanalytic_sg_scenario_checks": nonanalytic_checks,
                "comparison_checks": comparison_checks,
                "overall_passed": overall_passed,
                "reason": (
                    "comparison completed with Physics Engine selectable flux "
                    "schemes plus nonzero-pair and no-flux-obstacle numerical "
                    "property gates; TEST_ONLY result is not a physical validation"
                    if overall_passed
                    else "one or more TEST_ONLY numerical acceptance gates failed; "
                    "this is not a physical-model validation"
                ),
            }
        )
        report["all_json_values_finite"] = _finite_json_values(report)
        detailed_report_written = _write_json(args.output, report)
        print(args.output)
        return 0 if report["overall_passed"] and detailed_report_written else 2
    except (FloatingPointError, TypeError, ValueError) as error:
        report.update(
            {
                "workflow_status": "FAILED",
                "comparison_status": "FAILED_TEST_ONLY_NUMERICAL_RUN",
                "overall_passed": False,
                "reason": f"test-only flux validation failed: {error}",
            }
        )
        report["all_json_values_finite"] = _finite_json_values(report)
        _write_json(args.output, report)
        print(args.output)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
