# Controller contract — no controller implemented yet

This document fixes the mathematical and ownership boundary for future
controller work. It does not implement reinforcement learning, authorize
training, or select a final actuator.

Magnetic actuation is outside this project's scope. The eventual real
non-magnetic actuator is also still undecided: this contract must not be read as
an automatic choice of an electric, acoustic, optical, or other mechanism.

## Mathematical control path

At a held control time, the only allowed controller output is the command
`u_t`. It affects the state through a conservative potential:

```text
u_t
-> calibrated physical actuator
-> V_control(x; u_t)
-> F_control(x; u_t) = -grad_x V_control(x; u_t)
-> McKean–Vlasov flux and finite-volume evolution
-> rho_next
```

The SI-valued solver forms the effective potential

```text
Phi_eff = V_ext + V_control + (W_eff * rho)
```

and evolves the density through

```text
J = -D grad(rho) - M rho grad(Phi_eff),
partial_t rho + div(J) = 0.
```

Here `rho` is the two-dimensional density in `m^-2`, `D` is the diffusion
coefficient in `m^2/s`, and `M` is the overdamped mobility in `m/(N s)`.
`V_ext`, `V_control`, and `W_eff * rho` are energies in joules, so their
spatial gradients are forces in newtons. Consequently both terms in `J` have
units `1/(m s)` in two dimensions. With the Einstein relation, `D = M k_B T`.

`ControlledPotentialBackend` therefore exposes potential energy in joules and
force in newtons. Particle Monte Carlo and the continuum model can use the same
backend and verify

```text
force_newton = -gradient(potential_joule).
```

## Relation to the mobility-one notation

The implementation specification also uses the normalized form

```text
J = -sigma grad(rho) - rho grad(V + W_eff * rho).
```

That expression assumes mobility-one or corresponding nondimensional units. It
matches the SI equation only after a consistent change of variables. For
example, with energy scale `E_0`, length scale `L`, and time scale
`L^2/(M E_0)`, the dimensionless mobility is one and

```text
sigma = D/(M E_0).
```

Under the Einstein relation this becomes `sigma = k_B T/E_0`. Choosing
`E_0 = k_B T` makes the dimensionless diffusion coefficient one. Equivalently,
one may absorb `M` into rescaled potentials, but those rescaled quantities no
longer have joule units. The SI backend must not set the dimensional mobility
numerically to one or pass a rescaled potential through `potential_joule`.

## Hydrogel and pair-interaction isolation

The Hydrogel-to-pair-potential chain belongs entirely to the Physics Engine:

```text
Hydrogel state and constitutive variables (F, P, C, mu_gel)
-> calibrated two-particle mechanics / pair data
-> F_pair(r)
-> W_eff(r), with F_pair = -dW_eff/dr
-> W_eff * rho in the McKean–Vlasov effective potential
```

`V_control` and `W_eff * rho` are distinct additive contributions to
`Phi_eff`. A controller may not create, overwrite, tune, or bypass any part of
the Hydrogel or pair-reduction chain. If a future physical actuator genuinely
changes material properties, that coupling requires a separately approved and
validated Physics Engine model; it cannot be represented by direct controller
mutation.

## Current backend semantics

All current positions have shape `(..., 2)` and are measured in metres. A
controlled-potential backend returns potential shape `(...)` in joules and
force shape `(..., 2)` in newtons.

`ZeroControlledPotential` is a fail-closed null backend. For the current
two-dimensional validation interface it accepts only `None` or a finite exact
zero two-vector. It rejects a nonzero command so a missing actuator cannot
silently turn an apparent control experiment into uncontrolled evolution. Its
two-vector validation shape is not a choice of final actuator dimension.

`TestOnlyUniformFieldPotential` uses a dimensionless finite two-vector `u` with
`||u|| <= 1` and

```text
V_control(x; u) = -F_max u dot (x - x_ref),
-grad_x V_control = F_max u.
```

`F_max` is in newtons, `x - x_ref` is in metres, and their product is in
joules. `None` denotes the zero test command. This backend exists only to test
sign, units, shapes, and data flow. Its status is
`TEST_ONLY_NOT_FINAL_PHYSICS`; it is not a calibrated actuator and must not be
promoted or described as final physics.

## Forbidden paths

A future open-loop, rule-based, greedy, MPC, or RL controller must not:

- translate, overwrite, normalize, or otherwise edit the density array;
- move particle positions directly;
- modify Hydrogel `F`, `P`, `C`, or `mu_gel`;
- modify pair calibration data, `F_pair`, `W_eff`, or `W_eff * rho`;
- bypass the McKean–Vlasov flux/evolution;
- replace no-flux obstacle geometry with a reward-only collision penalty;
- describe an arbitrary painted field as a real actuator.

The controller owns only `u_t`. Density evolution remains a Physics Engine
result of the conservative flux.

## Gate before controller development

Select and calibrate a real approximately two-dimensional non-magnetic particle
system with:

1. a measurable diffusion coefficient;
2. a non-negligible pair force reduced from validated Hydrogel mechanics to a
   real `W_eff`;
3. an executable control input mapped to a calibrated
   `V_control(x; u_t)`;
4. parameter uncertainty ranges and experimental/literature provenance.

Only after replacing the test backends and re-running the required physics and
validation gates should open-loop, rule-based, greedy, or MPC baselines be
considered. MLP-DQN, CNN-DQN, and all training remain later work requiring
explicit user approval.
