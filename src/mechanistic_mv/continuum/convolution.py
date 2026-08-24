"""Reference and FFT implementations of the nonlocal pair convolution."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.fft import irfftn, next_fast_len, rfftn

from ..mechanics.density_scaling import (
    DensityConvention,
    require_continuum_ready,
    validate_pair_potential_scaling,
)
from ..mechanics.geometry import CartesianGrid
from ..mechanics.pair_potential import PairPotential, ZeroPairPotential


def direct_pair_convolution_joule(
    density_per_m2: np.ndarray,
    grid: CartesianGrid,
    potential: PairPotential,
    *,
    density_convention: DensityConvention = DensityConvention.PROBABILITY,
    population_count: int | None = None,
) -> np.ndarray:
    """Evaluate ``W_Kac*rho`` or ``W_pair*n`` by direct quadrature."""

    validate_pair_potential_scaling(
        potential,
        density_convention,
        population_count=population_count,
    )
    require_continuum_ready(potential)
    density = _density(density_per_m2, grid)
    if isinstance(potential, ZeroPairPotential):
        return np.zeros_like(density)
    x, y = grid.mesh_m()
    sources = np.stack((x.ravel(), y.ravel()), axis=-1)
    weights = density.ravel() * grid.cell_area_m2
    result = np.empty(sources.shape[0], dtype=np.float64)
    for index, target in enumerate(sources):
        displacement = target - sources
        result[index] = np.sum(potential.potential_joule(displacement) * weights)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("direct pair convolution became non-finite")
    return result.reshape(density.shape)


@dataclass(slots=True)
class FFTPairConvolver:
    """Frequency-cached linear convolution for a translation-invariant W."""

    grid: CartesianGrid
    potential: PairPotential
    density_convention: DensityConvention = DensityConvention.PROBABILITY
    population_count: int | None = None
    _full_shape: tuple[int, int] = field(init=False, repr=False)
    _fft_shape: tuple[int, int] = field(init=False, repr=False)
    _kernel_spectrum: np.ndarray | None = field(init=False, repr=False)
    _zero_interaction: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        validate_pair_potential_scaling(
            self.potential,
            self.density_convention,
            population_count=self.population_count,
        )
        require_continuum_ready(self.potential)
        self._zero_interaction = isinstance(self.potential, ZeroPairPotential)
        if self._zero_interaction:
            self._full_shape = (self.grid.ny, self.grid.nx)
            self._fft_shape = (self.grid.ny, self.grid.nx)
            self._kernel_spectrum = None
            return
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
        self._full_shape = (
            self.grid.ny + kernel.shape[0] - 1,
            self.grid.nx + kernel.shape[1] - 1,
        )
        self._fft_shape = tuple(
            next_fast_len(length, real=True) for length in self._full_shape
        )
        self._kernel_spectrum = rfftn(kernel, self._fft_shape)

    def convolve_joule(self, density_per_m2: np.ndarray) -> np.ndarray:
        density = _density(density_per_m2, self.grid)
        if self._zero_interaction:
            return np.zeros_like(density)
        if self._kernel_spectrum is None:
            raise RuntimeError("nonzero convolution kernel spectrum is unavailable")
        density_spectrum = rfftn(density, self._fft_shape)
        padded = irfftn(
            density_spectrum * self._kernel_spectrum,
            self._fft_shape,
        )
        full = padded[: self._full_shape[0], : self._full_shape[1]]
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
