"""Measure the implemented numerical optimizations on deterministic inputs."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy

from mechanistic_mv.continuum.convolution import (
    FFTPairConvolver,
    direct_pair_convolution_joule,
)
from mechanistic_mv.mechanics.geometry import CartesianGrid, RectangularDomain
from mechanistic_mv.mechanics.pair_potential import (
    TestOnlyGaussianRepulsion,
    ZeroPairPotential,
    mean_field_pair_force_newton,
)
from mechanistic_mv.mechanics.parameters import PhysicalParameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "validation" / "optimization_benchmark.json"


def _timed(callable_object, repeats: int = 3) -> tuple[float, object]:
    durations = []
    result: object = None
    for _ in range(repeats):
        start = perf_counter()
        result = callable_object()
        durations.append(perf_counter() - start)
    return float(min(durations)), result


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


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    parameters = PhysicalParameters()
    potential = TestOnlyGaussianRepulsion(
        parameters.thermal_energy_joule, 1.2e-6
    )
    domain = RectangularDomain((0.0, 20.0e-6), (0.0, 20.0e-6))
    rng = np.random.default_rng(20260824)

    convolution_results = []
    for size in (16, 24, 32):
        grid = CartesianGrid(domain, size, size)
        density = rng.uniform(0.5, 1.5, size=(size, size))
        density /= np.sum(density) * grid.cell_area_m2
        convolver = FFTPairConvolver(grid, potential)
        direct_pair_convolution_joule(density, grid, potential)
        convolver.convolve_joule(density)
        direct_seconds, direct = _timed(
            lambda: direct_pair_convolution_joule(density, grid, potential),
            repeats=5,
        )
        fft_seconds, accelerated = _timed(
            lambda: convolver.convolve_joule(density), repeats=5
        )
        convolution_results.append(
            {
                "grid": [size, size],
                "direct_seconds": direct_seconds,
                "fft_seconds": fft_seconds,
                "speedup": direct_seconds / fft_seconds,
                "maximum_absolute_error_joule": float(
                    np.max(np.abs(np.asarray(direct) - np.asarray(accelerated)))
                ),
            }
        )

    particle_count = 1200
    positions = rng.uniform(4.0e-6, 16.0e-6, size=(particle_count, 2))
    pair_results = []
    reference = None
    for chunk_size in (64, 128, 256, 1200):
        seconds, force = _timed(
            lambda selected=chunk_size: mean_field_pair_force_newton(
                positions, potential, chunk_size=selected
            ),
            repeats=3,
        )
        if reference is None:
            reference = np.asarray(force)
        pair_results.append(
            {
                "particle_count": particle_count,
                "chunk_size": chunk_size,
                "seconds": seconds,
                "theoretical_displacement_array_bytes": min(
                    chunk_size, particle_count
                )
                * particle_count
                * 2
                * 8,
                "maximum_absolute_difference_newton": float(
                    np.max(np.abs(np.asarray(force) - reference))
                ),
            }
        )

    zero_positions = rng.uniform(size=(100_000, 2))
    zero_seconds, zero_force = _timed(
        lambda: mean_field_pair_force_newton(
            zero_positions, ZeroPairPotential(), chunk_size=256
        ),
        repeats=5,
    )
    report = {
        "benchmark_scope": "implementation optimization, not model calibration",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision_at_run": _git_revision(),
        "timing_method": (
            "one untimed warm-up for convolution, then minimum of five equal "
            "repeats; particle force uses minimum of three repeats"
        ),
        "runtime": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "pair_potential_status": potential.physical_status,
        "fft_convolution": convolution_results,
        "chunked_particle_force": pair_results,
        "zero_interaction_fast_path": {
            "particle_count": int(zero_positions.shape[0]),
            "seconds": zero_seconds,
            "maximum_absolute_force_newton": float(np.max(np.abs(zero_force))),
        },
    }
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
