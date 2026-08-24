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

## Fixed project management

This is one repository with four permanent windows and three functional
branches. Do not create another project or directory for routine updates.

- `McKean–Vlasov Master Coordinator` owns `main`. It decomposes requirements,
  hands work to the responsible window, merges reviewed commits, runs the final
  integrated checks, and reports the result. It must not silently implement an
  affected module in place of that module window.
- `McKean–Vlasov Physics Engine` owns `mv/physics-model` and the mechanics,
  particle dynamics, continuum physics, geometry, flux, and energy code.
- `McKean–Vlasov Controller Contract` owns `mv/controller-contract` and only the
  physical control-potential contract. It is not a DQN training window until RL
  is explicitly approved.
- `McKean–Vlasov Experiment Lab` owns `mv/experiment-validation` and tests,
  validation scripts, figures, reports, benchmarks, and acceptance gates.

Because all windows share one checkout, branch work is handed off sequentially:

1. The coordinator starts from a clean `main` and identifies affected windows.
2. Each affected window synchronizes its branch from `main`, edits only its
   owned files, verifies them, commits them, and reports in the required Chinese
   format. The coordinator does not duplicate that implementation.
3. Each unaffected window explicitly reports that no code change was needed.
4. The coordinator merges the functional commits into `main`, runs integrated
   verification, and leaves the checkout clean on `main`.
5. A follow-up found during review returns to the owning functional window and
   branch before it is merged; it is not patched directly on `main`.

Every module report must use:

`file path -> changed code -> reason -> verification -> result`

Training never starts merely because validation code changed. The coordinator
must explicitly state whether it ran tests, short simulations, or training.
