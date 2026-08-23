"""Run the Phase 1--3 physics validation suite and save strict artifacts."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from mechanistic_mv.continuum.diagnostics import (
    density_moments,
    jensen_shannon_divergence,
    relative_l2_error,
)
from mechanistic_mv.continuum.initial_conditions import gaussian_density
from mechanistic_mv.continuum.mckean_vlasov import McKeanVlasovSolver
from mechanistic_mv.mechanics.external_force import HarmonicTestPotential
from mechanistic_mv.mechanics.geometry import (
    CartesianGrid,
    RectangleObstacle,
    RectangularDomain,
)
from mechanistic_mv.mechanics.pair_potential import (
    TestOnlyGaussianRepulsion,
    ZeroPairPotential,
)
from mechanistic_mv.mechanics.parameters import PhysicalParameters
from mechanistic_mv.particle_sim.empirical_density import empirical_density
from mechanistic_mv.particle_sim.langevin_interacting import overdamped_langevin_step


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "validation"


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


def _evolve(
    solver: McKeanVlasovSolver,
    initial_density: np.ndarray,
    *,
    number_of_steps: int,
    step_duration_s: float,
) -> tuple[np.ndarray, dict[str, list[float] | float]]:
    density = initial_density.copy()
    initial_mass = solver.mass(density)
    masses = [initial_mass]
    energies = [solver.free_energy(density).total_joule]
    minimum_densities = [float(np.min(density[solver.fluid_mask]))]
    maximum_fluxes: list[float] = []
    clipped_masses: list[float] = []
    substeps: list[float] = []
    for _ in range(number_of_steps):
        density, diagnostics = solver.step(density, step_duration_s)
        masses.append(diagnostics.final_mass)
        energies.append(solver.free_energy(density).total_joule)
        minimum_densities.append(diagnostics.minimum_density_per_m2)
        maximum_fluxes.append(diagnostics.maximum_abs_flux_per_m_s)
        clipped_masses.append(diagnostics.clipped_negative_mass)
        substeps.append(float(diagnostics.substeps))
    return density, {
        "mass": masses,
        "free_energy_joule": energies,
        "minimum_density_per_m2": minimum_densities,
        "maximum_abs_flux_per_m_s": maximum_fluxes,
        "clipped_negative_mass": clipped_masses,
        "substeps": substeps,
        "maximum_absolute_mass_error": max(abs(value - initial_mass) for value in masses),
    }


def _scenario_metrics(
    solver: McKeanVlasovSolver,
    initial: np.ndarray,
    final: np.ndarray,
    trajectory: dict[str, list[float] | float],
) -> dict[str, object]:
    first = density_moments(initial, solver.grid)
    last = density_moments(final, solver.grid)
    energies = np.asarray(trajectory["free_energy_joule"], dtype=np.float64)
    return {
        "initial_mass": solver.mass(initial),
        "final_mass": solver.mass(final),
        "maximum_absolute_mass_error": trajectory["maximum_absolute_mass_error"],
        "minimum_density_per_m2": float(
            np.min(np.asarray(trajectory["minimum_density_per_m2"]))
        ),
        "total_clipped_negative_mass": float(
            np.sum(np.asarray(trajectory["clipped_negative_mass"]))
        ),
        "maximum_abs_flux_per_m_s": float(
            np.max(np.asarray(trajectory["maximum_abs_flux_per_m_s"]))
        ),
        "initial_mean_m": first.mean_m.tolist(),
        "final_mean_m": last.mean_m.tolist(),
        "initial_covariance_m2": first.covariance_m2.tolist(),
        "final_covariance_m2": last.covariance_m2.tolist(),
        "initial_free_energy_joule": float(energies[0]),
        "final_free_energy_joule": float(energies[-1]),
        "maximum_energy_increment_joule": float(max(0.0, np.max(np.diff(energies)))),
        "total_fvm_substeps": int(np.sum(np.asarray(trajectory["substeps"]))),
    }


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    parameters = PhysicalParameters()
    domain = RectangularDomain((0.0, 20.0e-6), (0.0, 20.0e-6))
    grid = CartesianGrid(domain, 48, 48)
    thermal_energy = parameters.thermal_energy_joule

    initial = gaussian_density(grid, (8.0e-6, 10.5e-6), (1.1e-6, 1.4e-6))
    scenarios: dict[str, dict[str, object]] = {}
    fields: dict[str, np.ndarray] = {"initial": initial}
    trajectories: dict[str, dict[str, list[float] | float]] = {}

    diffusion_solver = McKeanVlasovSolver(grid, parameters, ZeroPairPotential())
    diffusion, trace = _evolve(
        diffusion_solver, initial, number_of_steps=10, step_duration_s=0.05
    )
    diffusion_metrics = _scenario_metrics(diffusion_solver, initial, diffusion, trace)
    initial_variance = density_moments(initial, grid).covariance_m2
    analytic_variance = np.diag(initial_variance) + 2.0 * parameters.diffusion_m2_per_s * 0.5
    numerical_variance = np.diag(density_moments(diffusion, grid).covariance_m2)
    diffusion_metrics["analytic_variance_m2"] = analytic_variance.tolist()
    diffusion_metrics["relative_variance_error"] = (
        np.abs(numerical_variance - analytic_variance) / analytic_variance
    ).tolist()
    scenarios["pure_diffusion_W0_V0"] = diffusion_metrics
    fields["pure diffusion"] = diffusion
    trajectories["pure diffusion"] = trace

    external = HarmonicTestPotential((10.5e-6, 9.5e-6), 7.0e-9)
    external_solver = McKeanVlasovSolver(
        grid, parameters, ZeroPairPotential(), external=external
    )
    external_density, trace = _evolve(
        external_solver, initial, number_of_steps=10, step_duration_s=0.05
    )
    scenarios["external_only_W0"] = _scenario_metrics(
        external_solver, initial, external_density, trace
    )
    fields["external only"] = external_density
    trajectories["external only"] = trace

    pair = TestOnlyGaussianRepulsion(1.5 * thermal_energy, 1.2e-6)
    pair_solver = McKeanVlasovSolver(grid, parameters, pair)
    pair_density, trace = _evolve(
        pair_solver, initial, number_of_steps=10, step_duration_s=0.05
    )
    scenarios["pair_only_nonzero_W"] = _scenario_metrics(
        pair_solver, initial, pair_density, trace
    )
    fields["pair only"] = pair_density
    trajectories["pair only"] = trace

    full_solver = McKeanVlasovSolver(grid, parameters, pair, external=external)
    full_density, trace = _evolve(
        full_solver, initial, number_of_steps=10, step_duration_s=0.05
    )
    scenarios["complete_MV_nonzero_V_nonzero_W"] = _scenario_metrics(
        full_solver, initial, full_density, trace
    )
    fields["complete MV"] = full_density
    trajectories["complete MV"] = trace

    obstacle = RectangleObstacle((9.0e-6, 11.0e-6), (6.0e-6, 14.0e-6))
    obstacle_solver = McKeanVlasovSolver(
        grid, parameters, pair, obstacles=(obstacle,), external=external
    )
    obstacle_initial = gaussian_density(
        grid,
        (6.5e-6, 10.0e-6),
        1.0e-6,
        fluid_mask=obstacle_solver.fluid_mask,
    )
    obstacle_density, trace = _evolve(
        obstacle_solver, obstacle_initial, number_of_steps=10, step_duration_s=0.05
    )
    obstacle_metrics = _scenario_metrics(
        obstacle_solver, obstacle_initial, obstacle_density, trace
    )
    obstacle_metrics["maximum_density_in_solid_per_m2"] = float(
        np.max(obstacle_density[~obstacle_solver.fluid_mask])
    )
    scenarios["no_flux_obstacle"] = obstacle_metrics
    fields["no-flux obstacle"] = obstacle_density
    trajectories["no-flux obstacle"] = trace

    comparison_grid = CartesianGrid(domain, 24, 24)
    comparison_external = HarmonicTestPotential((10.0e-6, 10.0e-6), 6.0e-9)
    comparison_pair = TestOnlyGaussianRepulsion(thermal_energy, 1.2e-6)
    comparison_solver = McKeanVlasovSolver(
        comparison_grid,
        parameters,
        comparison_pair,
        external=comparison_external,
    )
    comparison_density = gaussian_density(
        comparison_grid, (8.8e-6, 10.4e-6), 1.4e-6
    )
    rng = np.random.default_rng(20260824)
    particles = rng.normal(
        np.asarray([8.8e-6, 10.4e-6]), 1.4e-6, size=(900, 2)
    )
    mv_centroids = [density_moments(comparison_density, comparison_grid).mean_m]
    particle_centroids = [np.mean(particles, axis=0)]
    for _ in range(30):
        comparison_density, _ = comparison_solver.step(comparison_density, 0.01)
        particles, _ = overdamped_langevin_step(
            particles,
            parameters,
            comparison_pair,
            domain,
            dt_s=0.01,
            rng=rng,
            external=comparison_external,
            pair_chunk_size=128,
        )
        mv_centroids.append(density_moments(comparison_density, comparison_grid).mean_m)
        particle_centroids.append(np.mean(particles, axis=0))
    particle_density = empirical_density(particles, comparison_grid)
    mv_moments = density_moments(comparison_density, comparison_grid)
    particle_moments = density_moments(particle_density, comparison_grid)
    centroid_error = np.asarray(particle_centroids) - np.asarray(mv_centroids)
    scenarios["particle_Monte_Carlo_vs_MV"] = {
        "particle_count": int(particles.shape[0]),
        "same_external_potential": comparison_external.name,
        "same_pair_potential": comparison_pair.name,
        "same_diffusion_m2_per_s": parameters.diffusion_m2_per_s,
        "mean_error_m": (particle_moments.mean_m - mv_moments.mean_m).tolist(),
        "covariance_relative_error": (
            np.abs(
                np.diag(particle_moments.covariance_m2)
                - np.diag(mv_moments.covariance_m2)
            )
            / np.diag(mv_moments.covariance_m2)
        ).tolist(),
        "relative_L2_density_error": relative_l2_error(
            particle_density, comparison_density, comparison_grid
        ),
        "JS_divergence_nats": jensen_shannon_divergence(
            particle_density, comparison_density, comparison_grid
        ),
        "centroid_trajectory_RMSE_m": float(
            np.sqrt(np.mean(np.sum(centroid_error**2, axis=1)))
        ),
        "MV_free_energy_joule": comparison_solver.free_energy(
            comparison_density
        ).total_joule,
        "particle_empirical_free_energy_joule": comparison_solver.free_energy(
            particle_density
        ).total_joule,
    }

    continuum_names = (
        "pure_diffusion_W0_V0",
        "external_only_W0",
        "pair_only_nonzero_W",
        "complete_MV_nonzero_V_nonzero_W",
        "no_flux_obstacle",
    )
    maximum_mass_error = max(
        float(scenarios[name]["maximum_absolute_mass_error"])
        for name in continuum_names
    )
    maximum_clipped_mass = max(
        float(scenarios[name]["total_clipped_negative_mass"])
        for name in continuum_names
    )
    maximum_energy_increment = max(
        float(scenarios[name]["maximum_energy_increment_joule"])
        for name in continuum_names
    )
    comparison = scenarios["particle_Monte_Carlo_vs_MV"]
    checks = {
        "continuum_mass_error": {
            "passed": maximum_mass_error <= 1.0e-12,
            "value": maximum_mass_error,
            "upper_limit": 1.0e-12,
        },
        "no_material_negative_mass_clipping": {
            "passed": maximum_clipped_mass == 0.0,
            "value": maximum_clipped_mass,
            "upper_limit": 0.0,
        },
        "passive_free_energy_nonincrease": {
            "passed": maximum_energy_increment <= 1.0e-30,
            "value_joule": maximum_energy_increment,
            "upper_limit_joule": 1.0e-30,
        },
        "pure_diffusion_variance": {
            "passed": max(diffusion_metrics["relative_variance_error"]) <= 1.0e-4,
            "maximum_relative_error": max(
                diffusion_metrics["relative_variance_error"]
            ),
            "upper_limit": 1.0e-4,
        },
        "obstacle_solid_density": {
            "passed": obstacle_metrics["maximum_density_in_solid_per_m2"] == 0.0,
            "value_per_m2": obstacle_metrics["maximum_density_in_solid_per_m2"],
            "upper_limit_per_m2": 0.0,
        },
        "particle_MV_JS_divergence": {
            "passed": comparison["JS_divergence_nats"] <= 0.05,
            "value_nats": comparison["JS_divergence_nats"],
            "upper_limit_nats": 0.05,
        },
        "particle_MV_relative_L2": {
            "passed": comparison["relative_L2_density_error"] <= 0.25,
            "value": comparison["relative_L2_density_error"],
            "upper_limit": 0.25,
        },
        "particle_MV_centroid_trajectory": {
            "passed": comparison["centroid_trajectory_RMSE_m"] <= 2.0e-7,
            "value_m": comparison["centroid_trajectory_RMSE_m"],
            "upper_limit_m": 2.0e-7,
        },
    }
    overall_passed = all(bool(check["passed"]) for check in checks.values())
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision_at_run": _git_revision(),
        "runtime": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "model_status": (
            "Phase 1-3 numerical baseline; pair/external test potentials are "
            "TEST_ONLY_NOT_FINAL_PHYSICS"
        ),
        "primary_state": "rho(x,t)",
        "reinforcement_learning_implemented": False,
        "overall_passed": overall_passed,
        "checks": checks,
        "configuration": {
            "continuum_domain_x_limits_m": list(domain.x_limits_m),
            "continuum_domain_y_limits_m": list(domain.y_limits_m),
            "continuum_grid": [grid.ny, grid.nx],
            "continuum_number_of_steps": 10,
            "continuum_step_duration_s": 0.05,
            "obstacle_x_limits_m": list(obstacle.x_limits_m),
            "obstacle_y_limits_m": list(obstacle.y_limits_m),
            "particle_comparison_grid": [
                comparison_grid.ny,
                comparison_grid.nx,
            ],
            "particle_count": int(particles.shape[0]),
            "particle_number_of_steps": 30,
            "particle_step_duration_s": 0.01,
            "random_seed": 20260824,
        },
        "physical_parameters": parameters.as_dict(),
        "test_pair_potential": {
            "name": pair.name,
            "physical_status": pair.physical_status,
            "energy_scale_joule": pair.energy_scale_joule,
            "length_scale_m": pair.length_scale_m,
        },
        "test_external_potential": {
            "name": external.name,
            "physical_status": external.physical_status,
            "stiffness_newton_per_m": external.stiffness_newton_per_m,
        },
        "scenarios": scenarios,
        "continuum_trajectories": trajectories,
    }
    report_path = OUTPUT_DIRECTORY / "physics_validation.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    _plot_overview(
        grid,
        fields,
        comparison_grid,
        comparison_density,
        particle_density,
        obstacle_solver.fluid_mask,
    )
    _plot_diagnostics(trajectories, np.asarray(mv_centroids), np.asarray(particle_centroids))
    print(report_path)
    if not overall_passed:
        raise RuntimeError("one or more physics validation gates failed")


def _plot_overview(
    grid: CartesianGrid,
    fields: dict[str, np.ndarray],
    comparison_grid: CartesianGrid,
    comparison_density: np.ndarray,
    particle_density: np.ndarray,
    obstacle_fluid_mask: np.ndarray,
) -> None:
    entries = list(fields.items()) + [
        ("MV for MC comparison", comparison_density),
        ("particle empirical density", particle_density),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for axis, (title, density) in zip(axes.ravel(), entries, strict=True):
        selected_grid = comparison_grid if density.shape == comparison_density.shape else grid
        extent = (
            selected_grid.domain.x_limits_m[0] * 1.0e6,
            selected_grid.domain.x_limits_m[1] * 1.0e6,
            selected_grid.domain.y_limits_m[0] * 1.0e6,
            selected_grid.domain.y_limits_m[1] * 1.0e6,
        )
        image = axis.imshow(
            density * 1.0e-12,
            origin="lower",
            extent=extent,
            cmap="viridis",
            interpolation="nearest",
        )
        if title == "no-flux obstacle":
            solid = np.ma.masked_where(obstacle_fluid_mask, ~obstacle_fluid_mask)
            axis.imshow(
                solid,
                origin="lower",
                extent=extent,
                cmap="Greys",
                alpha=0.65,
                interpolation="nearest",
            )
        axis.set_title(title)
        axis.set_xlabel("x [micrometre]")
        axis.set_ylabel("y [micrometre]")
        fig.colorbar(image, ax=axis, label="density [1/micrometre^2]")
    fig.savefig(OUTPUT_DIRECTORY / "validation_overview.png", dpi=180)
    plt.close(fig)


def _plot_diagnostics(
    trajectories: dict[str, dict[str, list[float] | float]],
    mv_centroids_m: np.ndarray,
    particle_centroids_m: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for name, trace in trajectories.items():
        mass = np.asarray(trace["mass"])
        energy = np.asarray(trace["free_energy_joule"])
        axes[0, 0].plot(mass - mass[0], label=name)
        energy_scale = max(abs(energy[0]), np.finfo(float).tiny)
        axes[0, 1].plot((energy - energy[0]) / energy_scale, label=name)
        axes[1, 0].plot(np.asarray(trace["maximum_abs_flux_per_m_s"]), label=name)
    axes[0, 0].set_title("mass error without renormalization")
    axes[0, 0].set_ylabel("mass - initial mass")
    axes[0, 1].set_title("relative free-energy change")
    axes[1, 0].set_title("maximum absolute face flux")
    axes[1, 0].set_ylabel("1/(m s)")
    axes[1, 1].plot(mv_centroids_m[:, 0] * 1.0e6, label="MV x")
    axes[1, 1].plot(particle_centroids_m[:, 0] * 1.0e6, "--", label="particles x")
    axes[1, 1].plot(mv_centroids_m[:, 1] * 1.0e6, label="MV y")
    axes[1, 1].plot(particle_centroids_m[:, 1] * 1.0e6, "--", label="particles y")
    axes[1, 1].set_title("centroid trajectory")
    axes[1, 1].set_ylabel("micrometre")
    for axis in axes.ravel():
        axis.set_xlabel("recorded step")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.savefig(OUTPUT_DIRECTORY / "validation_diagnostics.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
