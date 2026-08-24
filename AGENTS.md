# Collaboration rules

## Scope

This repository implements only the mechanics, interacting Langevin,
McKean-Vlasov continuum, and validation phases. Do not add RL until the user
explicitly approves it after Phase 1-4 validation.

## Physical integrity

- Every parameter has explicit SI units or is explicitly labelled dimensionless.
- A pair potential must name its physical source. The bundled Gaussian backend
  is `TEST-ONLY` and must never be described as the final material model.
- Controls act through force fields. They never directly move particles or edit
  density arrays.
- Solid obstacles are no-flux geometry, not reward penalties or artificial
  obstacle potentials.
- A solver must not hide instability through routine clipping and
  renormalization.

## User-facing update protocol

For every requirement or code update:

1. The coordinator states the goal and affected modules.
2. Each affected module window reports in Chinese:
   `file path -> changed code -> reason -> verification -> result`.
3. Unaffected modules explicitly report that no change was needed.
4. The coordinator reports integrated status, Git provenance, whether any
   simulation/training ran, and the next action.

## Shared objective

Following the management model in the read-only reference project
`D:\projects\gym_distribution\mckean_vlasov_control`, implement this mechanics-
first model as three independently reviewable modules. Keep the reference
project unchanged.

## Task ownership

### MV Physics Engine

May edit:

- `src/mechanistic_mv/mechanics/`, except `controlled_potential.py`;
- `src/mechanistic_mv/particle_sim/`;
- continuum physics and numerics in `src/mechanistic_mv/continuum/`, except
  experiment-facing `diagnostics.py`;
- physics-focused tests under `tests/`.

Must not change the control contract or validation acceptance definitions
without coordinator approval.

### MV Controller Contract

May edit:

- `src/mechanistic_mv/mechanics/controlled_potential.py`;
- `docs/CONTROL_INTERFACE.md`;
- controller-focused tests under `tests/`;
- future controller files only after the user explicitly approves the RL phase.

Must not change the physical equations or evaluation definitions. This module
is the analogue of the reference project's DQN Controller, but currently owns
only the conservative physical control contract because RL is not yet approved.

### MV Experiment Lab

May edit:

- `src/mechanistic_mv/continuum/diagnostics.py`;
- validation and benchmark programs under `scripts/`;
- evaluation-focused tests under `tests/`;
- reproducible validation reports and figures under `outputs/validation/`.

This module owns final evidence and acceptance gates but must not silently
rewrite physics or controller behaviour.

## Coordinator-owned files

Changes to these files require explicit coordination:

- `src/mechanistic_mv/__init__.py`;
- `pyproject.toml`;
- `README.md`;
- `AGENTS.md`;
- `OPTIMIZATION_LOG.md`.

The coordinator decomposes work, resolves cross-module ownership, integrates
reviewed changes, runs the final combined checks, and reports provenance. It
does not create additional projects or folders for routine module work.

## Git organization

This remains one repository. The branches `mv/physics-model`,
`mv/controller-contract`, and `mv/experiment-validation` preserve independently
reviewable module history; they are not separate projects. `main` is the
integrated version. Because the windows share one checkout, branch-changing
work must be coordinated sequentially.

## General rules

- Do not modify, rename, or delete the reference `mckean_vlasov_control`
  project.
- Keep stochastic tests reproducible with explicit seeds.
- Add tests with every behavioural change.
- Preserve probability mass and reject NaN or infinite physical states.
- Do not commit future models, replay buffers, checkpoints, or training logs.
- Training never starts merely because physics or validation code changed.
