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

