"""Reference and FFT implementations of the nonlocal pair convolution."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import fftconvolve

from ..mechanics.geometry import CartesianGrid
from ..mechanics.pair_potential import PairPotential


def direct_pair_convolution_joule(
    density_per_m2: np.ndarray,
    grid: CartesianGrid,
    potential: PairPotential,
) -> np.ndarray:
    """Evaluate ``(W*rho)(x)`` by the O(number-of-cells squared) definition."""

    density = _density(density_per_m2, grid)
    x, y = grid.mesh_m()
    sources = np.stack((x.ravel(), y.ravel()), axis=-1)
    weights = density.ravel() * grid.cell_area_m2
    result = np.empty(sources.shape[0], dtype=np.float64)
    for index, target in enumerate(sources):
        displacement = target - sources
        result[index] = np.sum(potential.potential_joule(displacement) * weights)
    return result.reshape(density.shape)


@dataclass(slots=True)
class FFTPairConvolver:
    """Cached zero-padded linear convolution for a translation-invariant W."""

    grid: CartesianGrid
    potential: PairPotential
    _kernel_joule: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        x_offsets = np.arange(-(self.grid.nx - 1), self.grid.nx) * self.grid.dx_m
        y_offsets = np.arange(-(self.grid.ny - 1), self.grid.ny) * self.grid.dy_m
        offset_x, offset_y = np.meshgrid(x_offsets, y_offsets, indexing="xy")
        displacement = np.stack((offset_x, offset_y), axis=-1)
        kernel = np.asarray(
            self.potential.potential_joule(displacement), dtype=np.float64
        )
        expected = (2 * self.grid.ny - 1, 2 * self.grid.nx - 1)
        if kernel.shape != expected or not np.all(np.isfinite(kernel)):
            raise ValueError("pair potential returned an invalid convolution kernel")
        self._kernel_joule = kernel

    def convolve_joule(self, density_per_m2: np.ndarray) -> np.ndarray:
        density = _density(density_per_m2, self.grid)
        full = fftconvolve(density, self._kernel_joule, mode="full")
        y0, x0 = self.grid.ny - 1, self.grid.nx - 1
        result = full[y0 : y0 + self.grid.ny, x0 : x0 + self.grid.nx]
        result *= self.grid.cell_area_m2
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("FFT pair convolution became non-finite")
        return result


def _density(values: np.ndarray, grid: CartesianGrid) -> np.ndarray:
    density = np.asarray(values, dtype=np.float64)
    if density.shape != (grid.ny, grid.nx):
        raise ValueError("density must have grid shape")
    if not np.all(np.isfinite(density)) or np.any(density < 0.0):
        raise ValueError("density must be finite and non-negative")
    return density
