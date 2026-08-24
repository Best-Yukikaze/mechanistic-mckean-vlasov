"""Sign-explicit projection of an external solver reaction onto the pair axis."""

from __future__ import annotations

import numpy as np


def radial_reaction_force_newton(
    reaction_on_first_particle_newton: np.ndarray,
    second_to_first_center_vector_m: np.ndarray,
) -> np.ndarray:
    """Project reaction onto ``(X1-X2)/|X1-X2|``.

    Positive output is repulsive.  This function only processes an externally
    solved reaction vector; it does not supply a contact law.
    """

    reaction = np.asarray(reaction_on_first_particle_newton, dtype=np.float64)
    separation = np.asarray(second_to_first_center_vector_m, dtype=np.float64)
    if reaction.shape != separation.shape or reaction.shape[-1:] not in ((2,), (3,)):
        raise ValueError("reaction and separation must have matching (..., 2|3) shapes")
    if not np.all(np.isfinite(reaction)) or not np.all(np.isfinite(separation)):
        raise ValueError("reaction and separation must be finite")
    radius = np.linalg.norm(separation, axis=-1)
    if np.any(radius <= 0.0):
        raise ValueError("the pair center separation must be positive")
    return np.sum(reaction * separation, axis=-1) / radius
