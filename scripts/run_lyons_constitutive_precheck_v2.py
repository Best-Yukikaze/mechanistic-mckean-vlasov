"""Legacy CLI compatibility wrapper.

Use ``run_magnetic_particle_lyons_constitutive_precheck.py`` for new work.
"""

if __package__:
    from .run_magnetic_particle_lyons_constitutive_precheck import *  # noqa: F401,F403
else:  # pragma: no cover - direct legacy CLI execution
    from run_magnetic_particle_lyons_constitutive_precheck import *  # noqa: F401,F403


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
