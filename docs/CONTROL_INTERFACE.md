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
-> next selected density (rho_next or n_next)
```

Let `q` denote the selected density representation and let `I[q]` denote its
physically matched pair-interaction energy. The SI-valued solver forms

```text
Phi_eff[q, u_t] = V_ext + V_control(x; u_t) + I[q]
```

and evolves the density through

```text
J_q = -D grad(q) - M q grad(Phi_eff[q, u_t]),
partial_t q + div(J_q) = 0.
```

The two supported representations are:

- probability density: `q = rho`, `integral rho dx = 1`, and
  `I[rho] = W_Kac * rho`;
- number density: `q = n`, `integral n dx = N`, and
  `I[n] = W_pair * n`.

For equivalent descriptions of the same population,

```text
n = N rho,
W_Kac = N W_pair,
W_Kac * rho = W_pair * n.
```

Both `rho` and `n` are two-dimensional densities in `m^-2`. `D` is the
diffusion coefficient in `m^2/s`, and `M` is the overdamped mobility in
`m/(newton s)`. `V_ext`, `V_control`, and `I[q]` are energies in joules, so their
spatial gradients are forces in newtons. Consequently both terms in `J_q` have
units `1/(m s)` in two dimensions. With the Einstein relation, `D = M k_B T`.

`V_control(x; u_t)` is a per-particle one-body energy in joules. It is never
multiplied by `N`, divided by `N`, or otherwise recalibrated when
`density_convention`, `population_count`, or pair-force scaling changes. Thus
`F_control = -grad_x V_control` and the one-body drift
`M F_control` are identical in both density conventions. When `n = N rho` and
the pair representations are equivalent, `J_n = N J_rho`; this factor comes
from the density, not from the control backend. Likewise, a total one-body
energy acquires its population factor through integration against `n`, not by
rescaling `V_control`.

`ControlledPotentialBackend` therefore exposes potential energy in joules and
force in newtons. Particle Monte Carlo and the continuum model can use the same
backend and verify

```text
force_newton = -gradient(potential_joule).
```

The Python control interface deliberately accepts only positions and the held
command. A controlled-potential backend must not accept, query, infer, or mutate
`density_convention`, `population_count`, pair scaling, pair data, or continuum
short-range admission state.

## Relation to the mobility-one notation

The implementation specification also uses the normalized form

```text
J_q = -sigma grad(q) - q grad(V + I[q]).
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
-> calibrated two-particle mechanics / raw single-pair data
-> F_pair(r)
-> W_pair(r), with F_pair = -dW_pair/dr
-> either W_pair * n
   or an explicit, provenance-carrying W_Kac = N W_pair conversion
      followed by W_Kac * rho
```

`V_control` and `I[q]` are distinct additive contributions to `Phi_eff`. Raw
`W_pair` carries `UNSCALED_SINGLE_PAIR` semantics. It is not Kac-scaled merely
because a label or density convention changes; a
`KAC_NORMALIZED_PROBABILITY` result requires an actual multiplication by an
explicit `N` plus auditable provenance. A pair table whose calibrated domain
starts at `r_min > 0` is particle-only. It cannot enter a continuum convolution
until Physics supplies a physical short-range closure down to `r = 0` and
Experiment verifies the continuum-admission gate. The software fields
`minimum_supported_distance_m` and `continuum_ready` record this boundary;
setting `continuum_ready = True` is a necessary software condition, not evidence
that the missing material physics has been validated.

A controller may not create, overwrite, tune, relabel, extrapolate, or bypass
any part of the Hydrogel, pair-reduction, scaling, or short-range-admission
chain. These are Physics/Experiment responsibilities, not controller parameters.
If a future physical actuator genuinely changes material properties, that
coupling requires a separately approved and validated Physics Engine model; it
cannot be represented by direct controller mutation.

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
- modify pair calibration data, `F_pair`, `W_pair`, `W_Kac`, or `I[q]`;
- multiply raw single-pair data by a population, relabel its scaling, or change
  `population_count` to make a compatibility check pass;
- change `minimum_supported_distance_m`, mark an `r_min > 0` particle-only pair
  table as `continuum_ready`, invent its missing `r -> 0` values, or bypass the
  short-range admission gate;
- pass `density_convention`, `population_count`, or pair-admission state into a
  controlled-potential backend;
- bypass the McKean–Vlasov flux/evolution;
- replace no-flux obstacle geometry with a reward-only collision penalty;
- describe an arbitrary painted field as a real actuator.

The controller owns only `u_t`. Density evolution remains a Physics Engine
result of the conservative flux.

## Gate before controller development

Select and calibrate a real approximately two-dimensional non-magnetic particle
system with:

1. a measurable diffusion coefficient;
2. a non-negligible physical `W_pair` reduced from validated Hydrogel mechanics,
   or a numerically converted `W_Kac = N W_pair` carrying the population count
   and provenance;
3. a physically justified short-range closure to `r = 0` before any pair backend
   is admitted to continuum convolution;
4. an executable control input mapped to a calibrated
   `V_control(x; u_t)`;
5. parameter uncertainty ranges and experimental/literature provenance.

Real-actuator calibration is independent of the chosen density representation:
the same per-particle `V_control` and `F_control` must be used with probability
or number density. Choosing a density convention, supplying `population_count`,
proving pair scaling, and passing the `r = 0` continuum gate remain separate
Physics/Experiment prerequisites; they are not controller calibration knobs.

Only after replacing the test backends and re-running the required physics and
validation gates should open-loop, rule-based, greedy, or MPC baselines be
considered. MLP-DQN, CNN-DQN, and all training remain later work requiring
explicit user approval.
