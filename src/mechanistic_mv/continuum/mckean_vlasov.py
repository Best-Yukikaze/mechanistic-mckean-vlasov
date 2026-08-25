"""Positivity-aware conservative finite-volume McKean--Vlasov solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..mechanics.controlled_potential import (
    ControlledPotentialBackend,
    ZeroControlledPotential,
)
from ..mechanics.density_scaling import (
    DensityConvention,
    expected_density_mass,
    validate_density_convention,
    validate_population_count,
)
from ..mechanics.external_force import ExternalPotential, ZeroExternalPotential
from ..mechanics.geometry import CartesianGrid, RectangleObstacle
from ..mechanics.pair_potential import PairPotential
from ..mechanics.parameters import PhysicalParameters
from .convolution import FFTPairConvolver
from .flux import (
    DriftFluxScheme,
    FaceFluxes,
    build_face_masks,
    compute_face_fluxes,
)
from .free_energy import FreeEnergyComponents, free_energy_components


@dataclass(frozen=True, slots=True)
class StepDiagnostics:
    """Conservative-step diagnostics without a discrete energy certificate.

    ``fixed_control_free_energy_change_joule`` compares the initial and final
    state functional at the one control potential held fixed across this call.
    It is not an energy balance across a time-varying control, nor a proof that
    the high-order scheme is unconditionally free-energy dissipative.
    """

    substeps: int
    initial_mass: float
    final_mass: float
    minimum_stable_dt_s: float
    minimum_density_per_m2: float
    clipped_negative_mass: float
    maximum_abs_flux_per_m_s: float
    drift_flux_scheme: DriftFluxScheme = DriftFluxScheme.FIRST_ORDER_UPWIND
    fixed_control_free_energy_change_joule: float | None = None

    @property
    def absolute_mass_error(self) -> float:
        return abs(self.final_mass - self.initial_mass)


def conservative_update(
    density_per_m2: np.ndarray,
    fluxes: FaceFluxes,
    grid: CartesianGrid,
    dt_s: float,
) -> np.ndarray:
    """Apply one conservative cell-balance update without normalization."""

    density = np.asarray(density_per_m2, dtype=np.float64)
    divergence = (
        (fluxes.x_per_m_s[:, 1:] - fluxes.x_per_m_s[:, :-1]) / grid.dx_m
        + (fluxes.y_per_m_s[1:, :] - fluxes.y_per_m_s[:-1, :]) / grid.dy_m
    )
    return density - float(dt_s) * divergence


class McKeanVlasovSolver:
    """Reusable solver with cached geometry, potential, and FFT kernel."""

    def __init__(
        self,
        grid: CartesianGrid,
        parameters: PhysicalParameters,
        pair_potential: PairPotential,
        *,
        obstacles: Iterable[RectangleObstacle] = (),
        external: ExternalPotential | None = None,
        controlled_potential: ControlledPotentialBackend | None = None,
        cfl_safety: float = 0.9,
        drift_flux_scheme: DriftFluxScheme = DriftFluxScheme.FIRST_ORDER_UPWIND,
        record_fixed_control_free_energy: bool = False,
        density_convention: DensityConvention = DensityConvention.PROBABILITY,
        population_count: int | None = None,
    ) -> None:
        self.grid = grid
        self.parameters = parameters
        self.pair_potential = pair_potential
        self.obstacles = tuple(obstacles)
        self.external = external or ZeroExternalPotential()
        self.controlled_potential = controlled_potential or ZeroControlledPotential()
        self.density_convention = validate_density_convention(density_convention)
        declared_population = getattr(pair_potential, "scaling_population_count", None)
        if population_count is None and declared_population is not None:
            population_count = declared_population
        self.population_count = validate_population_count(
            population_count,
            required=self.density_convention is DensityConvention.NUMBER,
        )
        self.expected_density_mass = expected_density_mass(
            self.density_convention,
            self.population_count,
        )
        self.cfl_safety = float(cfl_safety)
        if not np.isfinite(self.cfl_safety) or not 0.0 < self.cfl_safety <= 1.0:
            raise ValueError("cfl_safety must lie in (0, 1]")
        if not isinstance(drift_flux_scheme, DriftFluxScheme):
            raise TypeError("drift_flux_scheme must be DriftFluxScheme")
        if not isinstance(record_fixed_control_free_energy, bool):
            raise TypeError("record_fixed_control_free_energy must be bool")
        self.drift_flux_scheme = drift_flux_scheme
        self.record_fixed_control_free_energy = record_fixed_control_free_energy

        self.fluid_mask = grid.fluid_mask(self.obstacles)
        self.face_masks = build_face_masks(self.fluid_mask)
        x, y = grid.mesh_m()
        self.cell_positions_m = np.stack((x, y), axis=-1)
        self.external_potential_joule = np.asarray(
            self.external.potential_joule(self.cell_positions_m), dtype=np.float64
        )
        if (
            self.external_potential_joule.shape != (grid.ny, grid.nx)
            or not np.all(np.isfinite(self.external_potential_joule))
        ):
            raise ValueError(
                "external potential must return one finite value per grid cell"
            )
        self.convolver = FFTPairConvolver(
            grid,
            pair_potential,
            density_convention=self.density_convention,
            population_count=self.population_count,
        )
        fluid_area = np.sum(self.fluid_mask) * grid.cell_area_m2
        self.reference_density_per_m2 = self.expected_density_mass / fluid_area

    def pair_convolution_joule(self, density_per_m2: np.ndarray) -> np.ndarray:
        density = self._validated_density(density_per_m2)
        return self.convolver.convolve_joule(density)

    def face_fluxes(
        self,
        density_per_m2: np.ndarray,
        control: np.ndarray | None = None,
        *,
        dt_s: float | None = None,
    ) -> tuple[FaceFluxes, np.ndarray]:
        """Return conservative face fluxes suitable for one explicit update.

        The default first-order scheme has no time-dependent spatial limiter.
        ``SECOND_ORDER_SCHARFETTER_GUMMEL`` instead requires ``dt_s`` so this
        method can reject an update beyond its explicit positivity CFL, rather
        than exposing a high-order flux with no usable time-step contract.
        Its full second-order temporal update is provided by :meth:`step`,
        which applies two positivity-safe stages in SSP-RK2 form.
        """

        density = self._validated_density(density_per_m2)
        controlled_potential = self._controlled_potential_grid(control)
        fluxes, stable, interaction = self._flux_stage(density, controlled_potential)
        if self.drift_flux_scheme is DriftFluxScheme.FIRST_ORDER_UPWIND:
            return fluxes, interaction
        if dt_s is None:
            raise ValueError(
                "SECOND_ORDER_SCHARFETTER_GUMMEL requires dt_s to return a "
                "flux with an explicit positivity-CFL contract"
            )
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        if dt > stable:
            raise ValueError(
                "dt_s exceeds the explicit outgoing-rate CFL bound required by "
                "SECOND_ORDER_SCHARFETTER_GUMMEL"
            )
        return fluxes, interaction

    def _effective_potential(
        self,
        density_per_m2: np.ndarray,
        controlled_potential_joule: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        interaction = self.convolver.convolve_joule(density_per_m2)
        effective_potential = (
            self.external_potential_joule
            + controlled_potential_joule
            + interaction
        )
        return effective_potential, interaction

    def _flux_stage(
        self,
        density_per_m2: np.ndarray,
        controlled_potential_joule: np.ndarray,
    ) -> tuple[FaceFluxes, float, np.ndarray]:
        """Build one spatial flux and its exact monotone explicit CFL bound."""

        effective_potential, interaction = self._effective_potential(
            density_per_m2, controlled_potential_joule
        )
        fluxes = self._compute_face_fluxes(
            density_per_m2, effective_potential, self.drift_flux_scheme
        )
        return (
            fluxes,
            self.stable_dt_s(fluxes),
            interaction,
        )

    def _compute_face_fluxes(
        self,
        density_per_m2: np.ndarray,
        effective_potential_joule: np.ndarray,
        drift_flux_scheme: DriftFluxScheme,
    ) -> FaceFluxes:
        return compute_face_fluxes(
            density_per_m2,
            effective_potential_joule,
            self.grid,
            diffusion_m2_per_s=self.parameters.diffusion_m2_per_s,
            mobility_m_per_newton_second=(
                self.parameters.mobility_m_per_newton_second
            ),
            face_masks=self.face_masks,
            drift_flux_scheme=drift_flux_scheme,
        )

    def stable_dt_s(self, fluxes: FaceFluxes) -> float:
        """Return a CFL from the actual per-cell outflow rate.

        This retains a conservative legacy compatibility path.

        Native fluxes include ``outgoing_rate_per_s`` and therefore always use
        the exact per-cell monotone row sum.  A manually constructed legacy
        ``FaceFluxes`` object has no such metadata, so it deliberately falls
        back to a global bound that permits simultaneous outflow through both
        faces of each coordinate direction, rather than guessing a local rate.
        """

        outgoing_rate = fluxes.outgoing_rate_per_s
        if outgoing_rate is None:
            return self._legacy_stable_dt_s(fluxes)
        rate = np.asarray(outgoing_rate, dtype=np.float64)
        if (
            rate.shape != (self.grid.ny, self.grid.nx)
            or not np.all(np.isfinite(rate))
            or np.any(rate < 0.0)
        ):
            raise ValueError(
                "outgoing_rate_per_s must be finite, non-negative, and match the grid"
            )
        maximum_rate = float(np.max(rate))
        if maximum_rate == 0.0:
            return np.inf
        return self.cfl_safety / maximum_rate

    def _legacy_stable_dt_s(self, fluxes: FaceFluxes) -> float:
        """Conservative CFL bound for a manually constructed legacy flux."""

        max_velocity_x = float(fluxes.max_abs_velocity_x_m_per_s)
        max_velocity_y = float(fluxes.max_abs_velocity_y_m_per_s)
        if (
            not np.isfinite(max_velocity_x)
            or not np.isfinite(max_velocity_y)
            or max_velocity_x < 0.0
            or max_velocity_y < 0.0
        ):
            raise ValueError("legacy face-flux maximum velocities must be finite and non-negative")
        diffusion_rate = 2.0 * self.parameters.diffusion_m2_per_s * (
            1.0 / self.grid.dx_m**2 + 1.0 / self.grid.dy_m**2
        )
        total_rate = (
            diffusion_rate
            + 2.0 * max_velocity_x / self.grid.dx_m
            + 2.0 * max_velocity_y / self.grid.dy_m
        )
        if total_rate <= 0.0:
            return np.inf
        return self.cfl_safety / total_rate

    def step(
        self,
        density_per_m2: np.ndarray,
        dt_s: float,
        *,
        control: np.ndarray | None = None,
    ) -> tuple[np.ndarray, StepDiagnostics]:
        """Advance a requested interval using automatically selected substeps."""

        density = self._validated_density(density_per_m2).copy()
        requested_dt = float(dt_s)
        if not np.isfinite(requested_dt) or requested_dt <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        initial_mass = self.mass(density)
        remaining = requested_dt
        substeps = 0
        minimum_stable_dt = np.inf
        clipped_mass = 0.0
        maximum_abs_flux = 0.0
        controlled_potential = self._controlled_potential_grid(control)
        initial_energy = (
            self._free_energy_with_controlled_potential(
                density, controlled_potential
            ).total_joule
            if self.record_fixed_control_free_energy
            else None
        )
        while remaining > max(np.finfo(np.float64).eps * requested_dt, 0.0):
            fluxes, stable, _ = self._flux_stage(density, controlled_potential)
            local_dt = min(remaining, stable)
            if not np.isfinite(local_dt) or local_dt <= 0.0:
                raise FloatingPointError("non-positive finite-volume substep")
            if self.drift_flux_scheme is DriftFluxScheme.FIRST_ORDER_UPWIND:
                candidate = conservative_update(
                    density, fluxes, self.grid, local_dt
                )
                self._assert_admissible_candidate(candidate)
                maximum_abs_flux = max(
                    maximum_abs_flux, self._maximum_abs_flux(fluxes)
                )
                minimum_stable_dt = min(minimum_stable_dt, stable)
            else:
                candidate, local_dt, stage_stable_dt, stage_flux = (
                    self._second_order_ssp_rk2_update(
                        density,
                        controlled_potential,
                        local_dt,
                        initial_fluxes=fluxes,
                        initial_stable_dt_s=stable,
                    )
                )
                maximum_abs_flux = max(
                    maximum_abs_flux, self._maximum_abs_flux(stage_flux[0]),
                    self._maximum_abs_flux(stage_flux[1]),
                )
                minimum_stable_dt = min(
                    minimum_stable_dt, stable, stage_stable_dt
                )
            density = candidate
            remaining -= local_dt
            substeps += 1
            if substeps > 1_000_000:
                raise RuntimeError("adaptive step exceeded one million substeps")
        final_mass = self.mass(density)
        mass_tolerance = 1.0e-11 * max(abs(initial_mass), 1.0)
        if abs(final_mass - initial_mass) > mass_tolerance:
            raise FloatingPointError(
                "finite-volume mass conservation failed without normalization"
            )
        fixed_control_energy_change = (
            self._free_energy_with_controlled_potential(
                density, controlled_potential
            ).total_joule
            - initial_energy
            if initial_energy is not None
            else None
        )
        return density, StepDiagnostics(
            substeps=substeps,
            initial_mass=initial_mass,
            final_mass=final_mass,
            minimum_stable_dt_s=float(minimum_stable_dt),
            minimum_density_per_m2=float(np.min(density[self.fluid_mask])),
            clipped_negative_mass=clipped_mass,
            maximum_abs_flux_per_m_s=maximum_abs_flux,
            drift_flux_scheme=self.drift_flux_scheme,
            fixed_control_free_energy_change_joule=fixed_control_energy_change,
        )

    def _second_order_ssp_rk2_update(
        self,
        density_per_m2: np.ndarray,
        controlled_potential_joule: np.ndarray,
        proposed_dt_s: float,
        *,
        initial_fluxes: FaceFluxes | None = None,
        initial_stable_dt_s: float | None = None,
    ) -> tuple[np.ndarray, float, float, tuple[FaceFluxes, FaceFluxes]]:
        """Apply two positivity-safe Scharfetter--Gummel stages and average.

        Each stage uses the nonlinear potential evaluated at its own density.
        The final convex average is the SSP-RK2 update, so positivity and mass
        conservation follow from the two CFL-bounded forward-Euler stages. If
        the second stage discovers a tighter row-sum CFL bound, the first
        stage is recomputed with that smaller common step before accepting
        either stage.  ``step`` supplies its already assembled initial flux
        and CFL bound so that this first SSP stage does not repeat the
        identical mean-field convolution and face-flux assembly.  The second
        stage is deliberately never reused: its density changes with the
        common substep and therefore requires a fresh mean-field potential.

        Omitting the optional pair preserves the previous private-call
        behavior.  Supplying exactly one member is rejected so the positivity
        contract cannot silently reuse a flux without its CFL bound.
        """

        local_dt = float(proposed_dt_s)
        if (initial_fluxes is None) != (initial_stable_dt_s is None):
            raise ValueError(
                "initial_fluxes and initial_stable_dt_s must be supplied together"
            )
        if initial_fluxes is None:
            flux_one, stable_one, _ = self._flux_stage(
                density_per_m2, controlled_potential_joule
            )
        else:
            flux_one = initial_fluxes
            stable_one = float(initial_stable_dt_s)
        for _ in range(32):
            local_dt = min(local_dt, stable_one)
            if not np.isfinite(local_dt) or local_dt <= 0.0:
                raise FloatingPointError("non-positive SSP-RK2 substep")
            stage_one = conservative_update(
                density_per_m2, flux_one, self.grid, local_dt
            )
            self._assert_admissible_candidate(stage_one)
            flux_two, stable_two, _ = self._flux_stage(
                stage_one, controlled_potential_joule
            )
            if local_dt > stable_two:
                local_dt = stable_two
                continue
            stage_two = conservative_update(stage_one, flux_two, self.grid, local_dt)
            self._assert_admissible_candidate(stage_two)
            candidate = 0.5 * (density_per_m2 + stage_two)
            self._assert_admissible_candidate(candidate)
            return candidate, local_dt, min(stable_one, stable_two), (
                flux_one,
                flux_two,
            )
        raise RuntimeError("SSP-RK2 CFL adaptation did not converge")

    def _assert_admissible_candidate(self, density_per_m2: np.ndarray) -> None:
        if np.any(density_per_m2[~self.fluid_mask] != 0.0):
            raise FloatingPointError("no-flux update changed a solid cell")
        if not np.all(np.isfinite(density_per_m2)):
            raise FloatingPointError("finite-volume update became non-finite")
        minimum = float(np.min(density_per_m2[self.fluid_mask]))
        if minimum < 0.0:
            raise FloatingPointError(
                "finite-volume positivity failure; reduce CFL or inspect fluxes "
                f"(minimum density={minimum:.6e})"
            )

    @staticmethod
    def _maximum_abs_flux(fluxes: FaceFluxes) -> float:
        return max(
            float(np.max(np.abs(fluxes.x_per_m_s))),
            float(np.max(np.abs(fluxes.y_per_m_s))),
        )

    def mass(self, density_per_m2: np.ndarray) -> float:
        density = self._validated_density(density_per_m2)
        return float(np.sum(density) * self.grid.cell_area_m2)

    def free_energy(
        self, density_per_m2: np.ndarray, control: np.ndarray | None = None
    ) -> FreeEnergyComponents:
        """Evaluate the state free energy at one fixed supplied control.

        Across calls with different controls this state functional excludes the
        actuator work, so its difference is diagnostic only rather than a
        controlled-system energy balance.
        """

        density = self._validated_density(density_per_m2)
        controlled = self._controlled_potential_grid(control)
        return self._free_energy_with_controlled_potential(density, controlled)

    def _free_energy_with_controlled_potential(
        self,
        density_per_m2: np.ndarray,
        controlled_potential_joule: np.ndarray,
    ) -> FreeEnergyComponents:
        interaction = self.convolver.convolve_joule(density_per_m2)
        return free_energy_components(
            density_per_m2,
            self.grid,
            self.parameters.thermal_energy_joule,
            self.reference_density_per_m2,
            self.external_potential_joule + controlled_potential_joule,
            interaction,
            fluid_mask=self.fluid_mask,
            density_convention=self.density_convention,
            population_count=self.population_count,
        )

    def _controlled_potential_grid(
        self, control: np.ndarray | None
    ) -> np.ndarray:
        flat_positions = self.cell_positions_m.reshape(-1, 2)
        values = np.asarray(
            self.controlled_potential.potential_joule(flat_positions, control),
            dtype=np.float64,
        )
        expected = (flat_positions.shape[0],)
        if values.shape != expected or not np.all(np.isfinite(values)):
            raise ValueError(
                "controlled potential must return one finite value per grid cell"
            )
        return values.reshape(self.grid.ny, self.grid.nx)

    def _validated_density(self, values: np.ndarray) -> np.ndarray:
        density = np.asarray(values, dtype=np.float64)
        if density.shape != (self.grid.ny, self.grid.nx):
            raise ValueError("density must have grid shape")
        if not np.all(np.isfinite(density)) or np.any(density < 0.0):
            raise ValueError("density must be finite and non-negative")
        solid_values = density[~self.fluid_mask]
        if np.any(solid_values != 0.0):
            raise ValueError("solid cells must contain exactly zero density")
        mass = float(np.sum(density) * self.grid.cell_area_m2)
        mass_tolerance = 1.0e-11 * max(self.expected_density_mass, 1.0)
        if abs(mass - self.expected_density_mass) > mass_tolerance:
            raise ValueError(
                "density mass does not match its convention: expected "
                f"{self.expected_density_mass:.16g}, got {mass:.16g}; "
                "the solver does not renormalize inputs"
            )
        return density
