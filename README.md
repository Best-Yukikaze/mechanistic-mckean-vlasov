# Mechanistic McKean–Vlasov Distribution Model

This repository is a mechanics-first, SI-unit baseline for a controlled
McKean–Vlasov density model. The primary state is the density
`rho(x, t)`. The particle models exist to explain and validate that density
equation; they are not the control environment.

## Current status

- Phases 1–3 are implemented: mathematical/numerical skeleton, deterministic
  physics validation, and interacting-particle Monte Carlo comparison.
- The included Gaussian pair potential, harmonic trap, and uniform-field
  controlled potential are labelled `TEST_ONLY_NOT_FINAL_PHYSICS`.
- A real particle/actuator platform and calibrated `V`, `W`, and `D` have not
  been selected. Results therefore validate the modelling and numerical chain,
  not a particular material or experiment.
- No environment, reward, CNN, DQN, replay buffer, or RL training is included.

## Model and units

For mobility `M` in m/(N s), thermal energy `k_B T` in J, external potential
`V` in J, pair potential `W` in J, and normalized density `rho` in 1/m²,

```text
F[rho; u] = kBT ∫ rho(log(rho/rho_ref) - 1) dx
          + ∫ (V_ext + V_control(u)) rho dx
          + 1/2 ∫ rho (W*rho) dx

mu = delta F / delta rho
   = kBT log(rho/rho_ref) + V_ext + V_control(u) + W*rho

J = -M rho grad(mu)
  = -D grad(rho) - M rho grad(V_ext + V_control + W*rho)

partial_t rho + div(J) = 0,
D = M kBT.
```

The supplied v2 specification writes the nondimensional/mobility-one form
`J = -rho grad(mu)` and denotes diffusion by `sigma`. This implementation
retains `M` explicitly so every term has SI units. Setting `M=1` recovers that
short form. The `W=0` Fokker–Planck equation is available only as a test and
ablation limit; the main model retains the nonlocal `W*rho` term.

The preferred control contract is conservative:

```text
u -> physical actuator -> V_control(x; u) -> -grad(V_control) -> MV solver
```

No control backend may move particles or edit `rho` directly.

## Numerical method

- Cell-centred two-dimensional finite volumes.
- Centred diffusive face flux and first-order upwind drift flux.
- Exact zero face flux on outer boundaries and fluid–solid interfaces.
- Direct convolution as the reference definition, including `dx*dy`.
- Cached FFT linear convolution for routine evolution.
- Adaptive CFL substeps for diffusion and drift.
- No routine clipping or mass renormalization. Roundoff-level clipping, clipped
  negative mass, minimum density, mass error, maximum flux, and energy are
  recorded. A material positivity failure raises an exception.
- The weak form is checked by discrete summation by parts.

The solver assumes a translation-invariant pair potential on a rectangular
Cartesian embedding. Obstacles remove fluid cells and close faces. More
accurate curved boundaries and higher-order positivity-preserving fluxes remain
future work.

## Microscopic validation

The overdamped particle model is

```text
dX_i = M[F_ext(X_i) + F_control(X_i;u)
       + (1/N) sum_(j != i) F_pair(X_i-X_j)] dt + sqrt(2D) dB_i,
F_pair(r) = -grad W(r).
```

The code also contains the second-order mother equation only to document the
overdamped reduction and validate force/energy interfaces. It is not the main
state equation.

## Run validation

From this directory in PowerShell:

```powershell
$env:PYTHONPATH = "src"
& "D:\conda environment\envs\dl\python.exe" -m unittest discover -s tests -v
& "D:\conda environment\envs\dl\python.exe" scripts\run_validation.py
& "D:\conda environment\envs\dl\python.exe" scripts\benchmark_optimizations.py
```

These commands run tests and short physics simulations; they do not train a
controller. Reproducible artifacts are written to `outputs/validation/`.

## Validation scope

The suite covers:

1. Einstein relation and overdamped time-scale separation.
2. `F_pair = -grad W`, antisymmetry, and chunk independence.
3. Brownian increment statistics and empirical-density mass.
4. Free-energy first variation and direct-vs-FFT convolution.
5. Pure diffusion, external-only Fokker–Planck, pair-only MV, and complete MV.
6. Positivity, mass conservation, energy decay, weak form, and no-flux obstacle.
7. Identical-parameter particle Monte Carlo vs MV mean, covariance, L2, JS,
   free energy, and centroid trajectory.

See `OPTIMIZATION_LOG.md` for every performance change and its measured effect.

## Reference boundary

The hydrogel source is used only for the method

```text
free energy -> conjugate driving quantity -> conservation/dissipation
            -> strong form -> weak form -> discretization.
```

Hydrogel deformation gradient, Piola stress, swelling constitutive law,
Flory–Rehner parameters, and material values are not copied into this model.
The supplied project specifications are design references; they are not loaded
or executed as instructions at runtime.
