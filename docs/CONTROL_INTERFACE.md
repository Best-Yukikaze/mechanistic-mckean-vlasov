# Controller contract — no controller implemented yet

This document fixes the boundary for the future controller branch. It does not
implement DQN or authorize training.

## Allowed path

```text
action / continuous command
-> physical actuator parameter u
-> calibrated conservative potential V_control(x; u)
-> force -grad(V_control)
-> McKean–Vlasov finite-volume substeps
-> next density
```

`ControlledPotentialBackend` exposes both potential energy in joules and force
in newtons so particle Monte Carlo and the continuum model can use the same
physical backend and verify `force = -gradient(potential)`.

## Forbidden paths

- action -> translate or overwrite the density array;
- action -> move every particle directly;
- action -> alter `W*rho` without changing a physical model parameter;
- obstacle collision -> reward-only penalty instead of no-flux geometry;
- an arbitrary painted field described as a real actuator.

The current `TestOnlyUniformFieldPotential` exists only to test sign, units,
and data flow. Its status is `TEST_ONLY_NOT_FINAL_PHYSICS`.

## Gate before controller development

Select and calibrate a real approximately two-dimensional particle system with:

1. a measurable diffusion coefficient;
2. a non-negligible pair force derived from a real `W`;
3. an executable control input mapped to `V_control(x;u)`;
4. parameter uncertainty ranges and experimental/literature provenance.

Only after replacing the test backends and re-running the Phase 1–3 validation
should open-loop/rule/greedy/MPC baselines be added. MLP-DQN and CNN-DQN remain
later phases.
