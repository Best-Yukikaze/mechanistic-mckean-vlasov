"""Compatibility import for the canonical magnetic-particle potential API.

New code must import from :mod:`mechanistic_mv.mechanics.magnetic_particle.potential`.
This module deliberately preserves historic source and Controller imports.
"""

from .magnetic_particle.potential import *  # noqa: F401,F403
