# Collaboration rules

## Scope

This repository implements only the mechanics, interacting Langevin,
McKean-Vlasov continuum, and validation phases. Do not add RL until the user
explicitly approves it after Phase 1-4 validation.

Magnetic particles, magnetic fields, coils, magnetophoresis, and magnetic
dipole interactions are out of scope for this project. Do not select or
implement a magnetic physical backend. The eventual real non-magnetic platform
must still provide physically sourced ``V(x; u)``, nonzero ``W(x-y)``, and
diffusion parameters before any test backend is promoted.

## Physical integrity

- Every parameter has explicit SI units or is explicitly labelled dimensionless.
- McKean–Vlasov is the main control-state equation. Hydrogel variables
  `F`, `P`, `C`, and `mu_gel` remain material/contact-scale variables and must
  never become density-state channels.
- The formal material source is the supplied Hong/Guo Hydrogel Model II Gibbs
  formulation. Missing equations, parameters, boundary conditions, or closure
  assumptions must raise an explicit error or remain `BLOCKED`; do not invent
  a substitute material law.
- The final `W_eff` must be derived from validated Hydrogel two-particle
  mechanics. Gaussian, Morse, Hertz, quadratic, or other hand-selected laws
  may exist only as explicitly labelled baselines or test fixtures.
- A physical pair reduction requires an explicit `tau_gel << tau_swarm`
  assessment. Missing time scales or a missing quantitative criterion are
  `UNVERIFIED`, not passing.
- Raw contact data are unscaled single-pair mechanics. Conversion to the
  unit-mass/Kac MV convention requires an explicit population or concentration
  scale and provenance; changing an enum or label is not a conversion.
- A pair potential must name its physical source. The bundled Gaussian backend
  is `TEST-ONLY` and must never be described as the final material model.
- Do not reinterpret the generic potential interfaces as magnetic models.
- Controls act through force fields. They never directly move particles or edit
  density arrays.
- Solid obstacles are no-flux geometry, not reward penalties or artificial
  obstacle potentials.
- A solver must not hide instability through routine clipping and
  renormalization.
- Controller code may interact with physics only through
  `u -> ControlledPotential -> V(x;u)`. It may not modify `rho`, particle
  positions, Hydrogel data, or `W_eff`.
- Physics changes require physics-validation tests. Performance changes must
  preserve a slower or analytic equivalence reference.
- Experiment Lab may measure, reject, and report physics but may not rewrite
  equations or relax gates to obtain a pass.
- Every temporary model or generated fixture must be labelled
  `TEST_ONLY_NOT_FINAL_PHYSICS` or `TEST_ONLY_NOT_CALIBRATED` as appropriate.

## User-facing update protocol

For every requirement or code update:

1. The coordinator states the goal and affected modules.
2. Each affected module window reports in Chinese:
   `file path -> changed code -> reason -> verification -> result`.
3. Unaffected modules explicitly report that no change was needed.
4. The coordinator reports integrated status, Git provenance, whether any
   simulation/training ran, and the next action.

## Default complete-delivery rule

Unless the user explicitly asks for a short answer, every delivery for this
repository must be complete rather than summary-only:

- Lead with the conclusion, then state the exact affected files and the code
  or requirement changes.
- Explain the mathematical or physical reason for each substantive change.
- State the verification command or method, the result, and any numerical
  evidence that supports the claim.
- Explicitly report assumptions, failures, blockers, unverified items, and
  `TEST_ONLY_*` limitations; never omit them to make a result appear complete.
- Keep the per-module Chinese report format
  `file path -> changed code -> reason -> verification -> result`, including
  an explicit "no change needed" report for unaffected modules.

The coordinator must additionally record Git provenance, the simulation or
training status, and the next physically meaningful action. This rule applies
to new requirements and to every subsequent code update in the project.

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

Owns the corresponding mathematics: Hydrogel free/Gibbs energy, constitutive
conjugacy, stress, time-scale reduction, two-particle equilibrium/contact,
`F_pair`, `W_eff`, the particle SDE, and the MV continuum equation.

### MV Controller Contract

May edit:

- `src/mechanistic_mv/mechanics/controlled_potential.py`;
- `docs/CONTROL_INTERFACE.md`;
- controller-focused tests under `tests/`;
- future controller files only after the user explicitly approves the RL phase.

Must not change the physical equations or evaluation definitions. This module
is the analogue of the reference project's DQN Controller, but currently owns
only the conservative physical control contract because RL is not yet approved.

Owns the corresponding mathematics: `u -> V_control`,
`F_control=-grad(V_control)`, dimensional mobility, and the boundary between
control commands and the physics engine.

### MV Experiment Lab

May edit:

- `src/mechanistic_mv/continuum/diagnostics.py`;
- validation and benchmark programs under `scripts/`;
- evaluation-focused tests under `tests/`;
- reproducible validation reports and figures under `outputs/validation/`.

This module owns final evidence and acceptance gates but must not silently
rewrite physics or controller behaviour.

Owns the corresponding mathematics: finite-difference conjugacy checks,
force/potential consistency, mass and energy diagnostics, convergence orders,
and particle-versus-MV discrepancies.

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
