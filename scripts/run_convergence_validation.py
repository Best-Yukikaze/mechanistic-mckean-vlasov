"""Run grid, CFL, and multi-seed particle--continuum validation studies."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mechanistic_mv.continuum.diagnostics import (
    density_moments,
    jensen_shannon_divergence,
    relative_l2_error,
)
from mechanistic_mv.continuum.initial_conditions import gaussian_density
from mechanistic_mv.continuum.mckean_vlasov import McKeanVlasovSolver
from mechanistic_mv.mechanics.external_force import HarmonicTestPotential
from mechanistic_mv.mechanics.geometry import CartesianGrid, RectangularDomain
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


def _pure_diffusion_grid_study(
    parameters: PhysicalParameters,
    domain: RectangularDomain,
) -> list[dict[str, float | int]]:
    records = []
    final_time_s = 0.2
    initial_variance_m2 = 1.0e-12
    for size in (24, 48, 96):
        grid = CartesianGrid(domain, size, size)
        solver = McKeanVlasovSolver(grid, parameters, ZeroPairPotential())
        initial = gaussian_density(
            grid, (10.0e-6, 10.0e-6), np.sqrt(initial_variance_m2)
        )
        advanced, diagnostics = solver.step(initial, final_time_s)
        x, y = grid.mesh_m()
        variance = (
            initial_variance_m2
            + 2.0 * parameters.diffusion_m2_per_s * final_time_s
        )
        analytic = np.exp(
            -((x - 10.0e-6) ** 2 + (y - 10.0e-6) ** 2)
            / (2.0 * variance)
        )
        analytic /= np.sum(analytic) * grid.cell_area_m2
        records.append(
            {
                "grid_size": size,
                "relative_L2_density_error": relative_l2_error(
                    advanced, analytic, grid
                ),
                "mass_error": diagnostics.absolute_mass_error,
                "substeps": diagnostics.substeps,
            }
        )
    for index in range(1, len(records)):
        records[index]["observed_order_from_previous_grid"] = float(
            np.log2(
                records[index - 1]["relative_L2_density_error"]
                / records[index]["relative_L2_density_error"]
            )
        )
    return records


def _cfl_refinement_study(
    parameters: PhysicalParameters,
    domain: RectangularDomain,
) -> list[dict[str, float | int]]:
    grid = CartesianGrid(domain, 32, 32)
    initial = gaussian_density(
        grid, (8.0e-6, 11.0e-6), (1.0e-6, 1.4e-6)
    )
    pair = TestOnlyGaussianRepulsion(
        1.5 * parameters.thermal_energy_joule, 1.1e-6
    )
    external = HarmonicTestPotential((10.0e-6, 10.0e-6), 8.0e-9)
    safeties = (0.9, 0.45, 0.225, 0.1125)
    solutions = []
    records = []
    for safety in safeties:
        solver = McKeanVlasovSolver(
            grid,
            parameters,
            pair,
            external=external,
            cfl_safety=safety,
        )
        solution, diagnostics = solver.step(initial, 0.5)
        solutions.append(solution)
        records.append(
            {
                "cfl_safety": safety,
                "substeps": diagnostics.substeps,
                "mass_error": diagnostics.absolute_mass_error,
            }
        )
    reference = solutions[-1]
    for record, solution in zip(records, solutions, strict=True):
        record["relative_L2_to_finest_CFL"] = relative_l2_error(
            solution, reference, grid
        )
    return records


def _particle_multi_seed_study(
    parameters: PhysicalParameters,
    domain: RectangularDomain,
) -> list[dict[str, float | int]]:
    grid = CartesianGrid(domain, 20, 20)
    pair = TestOnlyGaussianRepulsion(
        parameters.thermal_energy_joule, 1.2e-6
    )
    external = HarmonicTestPotential((10.0e-6, 10.0e-6), 6.0e-9)
    solver = McKeanVlasovSolver(grid, parameters, pair, external=external)
    continuum = gaussian_density(grid, (8.8e-6, 10.4e-6), 1.4e-6)
    for _ in range(20):
        continuum, _ = solver.step(continuum, 0.01)
    continuum_moments = density_moments(continuum, grid)
    records = []
    for seed in range(20260824, 20260829):
        rng = np.random.default_rng(seed)
        particles = rng.normal(
            np.asarray([8.8e-6, 10.4e-6]), 1.4e-6, size=(500, 2)
        )
        for _ in range(20):
            particles, _ = overdamped_langevin_step(
                particles,
                parameters,
                pair,
                domain,
                dt_s=0.01,
                rng=rng,
                external=external,
            )
        sampled = empirical_density(particles, grid)
        sampled_moments = density_moments(sampled, grid)
        records.append(
            {
                "seed": seed,
                "particle_count": int(particles.shape[0]),
                "mean_error_norm_m": float(
                    np.linalg.norm(
                        sampled_moments.mean_m - continuum_moments.mean_m
                    )
                ),
                "maximum_diagonal_covariance_relative_error": float(
                    np.max(
                        np.abs(
                            np.diag(sampled_moments.covariance_m2)
                            - np.diag(continuum_moments.covariance_m2)
                        )
                        / np.diag(continuum_moments.covariance_m2)
                    )
                ),
                "relative_L2_density_error": relative_l2_error(
                    sampled, continuum, grid
                ),
                "JS_divergence_nats": jensen_shannon_divergence(
                    sampled, continuum, grid
                ),
            }
        )
    return records


def _summary(records: list[dict[str, float | int]], key: str) -> dict[str, float]:
    values = np.asarray([record[key] for record in records], dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "sample_standard_deviation": float(np.std(values, ddof=1)),
        "maximum": float(np.max(values)),
    }


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    parameters = PhysicalParameters()
    domain = RectangularDomain((0.0, 20.0e-6), (0.0, 20.0e-6))
    grid_records = _pure_diffusion_grid_study(parameters, domain)
    cfl_records = _cfl_refinement_study(parameters, domain)
    particle_records = _particle_multi_seed_study(parameters, domain)

    grid_errors = [
        float(record["relative_L2_density_error"]) for record in grid_records
    ]
    cfl_errors = [
        float(record["relative_L2_to_finest_CFL"])
        for record in cfl_records[:-1]
    ]
    particle_summary = {
        key: _summary(particle_records, key)
        for key in (
            "mean_error_norm_m",
            "maximum_diagonal_covariance_relative_error",
            "relative_L2_density_error",
            "JS_divergence_nats",
        )
    }
    checks = {
        "pure_diffusion_error_decreases": grid_errors[0] > grid_errors[1] > grid_errors[2],
        "pure_diffusion_finest_relative_L2_below_1e-3": grid_errors[-1] < 1.0e-3,
        "fine_grid_observed_order_above_1p5": float(
            grid_records[-1]["observed_order_from_previous_grid"]
        )
        > 1.5,
        "CFL_refinement_error_decreases": cfl_errors[0] > cfl_errors[1] > cfl_errors[2],
        "CFL_next_finest_relative_L2_below_2e-3": cfl_errors[-1] < 2.0e-3,
        "multi_seed_max_JS_below_0p08": particle_summary["JS_divergence_nats"]["maximum"] < 0.08,
        "multi_seed_max_relative_L2_below_0p35": particle_summary["relative_L2_density_error"]["maximum"] < 0.35,
        "multi_seed_max_mean_error_below_3e-7_m": particle_summary["mean_error_norm_m"]["maximum"] < 3.0e-7,
        "multi_seed_max_covariance_error_below_0p35": particle_summary[
            "maximum_diagonal_covariance_relative_error"
        ]["maximum"]
        < 0.35,
    }
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision_at_run": _git_revision(),
        "model_status": "TEST_ONLY_NOT_FINAL_PHYSICS numerical convergence study",
        "overall_passed": all(checks.values()),
        "checks": checks,
        "pure_diffusion_grid_refinement": grid_records,
        "complete_MV_CFL_refinement": cfl_records,
        "particle_MV_multi_seed": {
            "records": particle_records,
            "summary": particle_summary,
        },
    }
    output_path = OUTPUT_DIRECTORY / "convergence_validation.json"
    output_path.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    _plot(grid_records, cfl_records, particle_records)
    print(output_path)
    if not report["overall_passed"]:
        raise RuntimeError("one or more convergence validation gates failed")


def _plot(
    grid_records: list[dict[str, float | int]],
    cfl_records: list[dict[str, float | int]],
    particle_records: list[dict[str, float | int]],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    axes[0].loglog(
        [record["grid_size"] for record in grid_records],
        [record["relative_L2_density_error"] for record in grid_records],
        "o-",
    )
    axes[0].set_title("pure diffusion grid refinement")
    axes[0].set_xlabel("cells per direction")
    axes[0].set_ylabel("relative L2 density error")

    axes[1].loglog(
        [record["cfl_safety"] for record in cfl_records[:-1]],
        [record["relative_L2_to_finest_CFL"] for record in cfl_records[:-1]],
        "o-",
    )
    axes[1].invert_xaxis()
    axes[1].set_title("complete MV CFL refinement")
    axes[1].set_xlabel("CFL safety")
    axes[1].set_ylabel("relative L2 to finest CFL")

    seeds = [str(record["seed"])[-2:] for record in particle_records]
    axes[2].bar(
        np.arange(len(seeds)) - 0.18,
        [record["JS_divergence_nats"] for record in particle_records],
        width=0.36,
        label="JS [nats]",
    )
    axes[2].bar(
        np.arange(len(seeds)) + 0.18,
        [record["relative_L2_density_error"] for record in particle_records],
        width=0.36,
        label="relative L2",
    )
    axes[2].set_xticks(np.arange(len(seeds)), seeds)
    axes[2].set_xlabel("seed suffix")
    axes[2].set_title("particle--MV across seeds")
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.savefig(OUTPUT_DIRECTORY / "convergence_validation.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
