# Mechanistic McKean–Vlasov Distribution Model

This repository is a mechanics-first, SI-unit platform for the chain

```text
Hydrogel constitutive law
-> two-particle contact mechanics
-> F_pair(r)
-> W_eff(r)
-> W_eff * rho
-> controlled McKean-Vlasov density evolution.
```

The primary control state remains `rho(x, t)`. Hydrogel deformation and solvent
variables live only at the material/contact scale; particle simulations exist
to explain and validate the density equation and are not the control
environment.

## Current status

- The Hong/Guo Model II initial-swelling Gibbs constitutive law, analytic first
  Piola stress, chemical conjugacy, strict parameter/domain checks, and the
  quasi-static time-scale decision are implemented.
- The complete two-Hydrogel contact sweep schema and the validated
  `F_pair -> W_eff` PCHIP integration path are implemented. A real contact FEM
  result is not: calibrated material values, particle geometry, mesh,
  boundary/bath conditions, and distance-sweep data were not supplied.
- Real contact inputs now have a fail-closed collection contract: the project
  can write a 110-leaf JSON template with no numerical values, then preflight
  a supplied manifest against field units, provenance, time-scale, scaling,
  and short-range gates. A `READY_FOR_CONTACT_SOLVER` preflight still means
  only that the input is ready to hand to a future solver; it never means that
  contact data, `F_pair`, or `W_eff` already exist.
- The continuum solver, interacting-particle validation model, no-flux
  geometry, deterministic validation, and Monte Carlo comparison remain
  implemented.
- Probability density and number density are now separate, checked conventions:
  `integral(rho)=1` uses a Kac-scaled potential and a `1/N` particle sum,
  while `integral(n)=N` uses the unscaled single-pair potential and an
  unscaled particle sum. The old probability convention remains the default.
- A force table starting at `r_min>0` is retained as valid particle-only data.
  It is rejected before continuum-kernel construction because the convolution
  evaluates zero displacement; the code never prepends a synthetic `r=0`
  sample or invents a short-range closure.
- The included Gaussian pair potential, harmonic trap, and uniform-field
  controlled potential are labelled `TEST_ONLY_NOT_FINAL_PHYSICS`.
- The supplied Hydrogel source names `N*nu`, `chi`, `phi0`, `Delta mu/(kT)`,
  and the `kT/nu` stress scale but supplies no calibrated numerical values.
  Current generated Hydrogel/pair artifacts are therefore explicitly
  `TEST_ONLY_NOT_CALIBRATED` or `BLOCKED`.
- A real actuator and calibrated `V`, `D`, particle count/concentration, and
  single-pair-to-Kac scaling have not been selected. Results validate code and
  numerical contracts, not a particular experiment.
- Magnetic particles, magnetic fields, coils, magnetophoresis, and magnetic
  dipole interactions are explicitly outside this project's selected scope.
  The future real platform must be non-magnetic while still supplying a
  physical controlled `V(x; u)` and a nonzero physical `W(x-y)`.
- No environment, reward, CNN, DQN, replay buffer, or RL training is included.

## Model and units

At the material scale, the implemented source model is the initial-swelling
Gibbs formulation. With `a=N*nu`, `phi=phi0`, `delta=Delta mu/(kT)`,
`J=det(F)`, and `I1=F:F`, it exposes

```text
G(F, delta),
P = partial G / partial F,
-partial G / partial delta = J - phi0,
nu C_dry = (J - phi0) / phi0.
```

The source eliminates `C` through the swelling constraint. It does not uniquely
specify an independent `Psi(F, C)` closure, so that unsupported API raises
`NotImplementedError` instead of inventing one. None of `F`, `P`, `C`, or the
Hydrogel chemical conjugate is an MV state variable.

For validated pair data,

```text
W_eff(r_ref) = 0,
W_eff(r) = -integral_[r_ref,r] F_pair(s) ds,
F_pair(r) = -dW_eff/dr.
```

Raw contact FEM output has `UNSCALED_SINGLE_PAIR` semantics. It can be consumed
directly with number density `n`, or converted to the probability-density/Kac
form only by the explicit numerical conversion

```text
n = N rho,
F_Kac = N F_pair,
W_Kac = N W_pair.
```

The conversion multiplies the table values and records the positive integer
population and its provenance. Merely relabelling the scaling enum is rejected.

For mobility `M` in m/(N s), thermal energy `k_B T` in J, external potential
`V` in J, pair potential `W` in J, and density in 1/m², the same physical
system has two equivalent representations:

```text
probability: integral(rho dx)=1,  interaction=W_Kac*rho
number:      integral(n dx)=N,    interaction=W_pair*n
W_Kac*rho = W_pair*n.
```

Writing the probability convention explicitly,

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

With reference density `n_ref=N*rho_ref`, the number-density free energy is
`N` times the probability-density free energy, while both conventions produce
the same interaction field and particle trajectories.

The preferred control contract is conservative:

```text
u -> physical actuator -> V_control(x; u) -> -grad(V_control) -> MV solver
```

No control backend may move particles or edit `rho` directly.

## Numerical method

- Cell-centred two-dimensional finite volumes.
- Two selectable conservative face-flux paths:
  `FIRST_ORDER_UPWIND` is the retained centred-diffusion/donor-cell-drift
  reference; `SECOND_ORDER_SCHARFETTER_GUMMEL` is the exponentially fitted
  Scharfetter--Gummel (Chang--Cooper) discretisation of the complete
  drift--diffusion flux.  The latter uses SSP-RK2 time stepping.
- The Scharfetter--Gummel path uses the Einstein relation already enforced by
  `PhysicalParameters`, preserves the discrete Gibbs equilibrium on each open
  face, and selects explicit substeps from the maximum per-cell outgoing-rate
  row sum rather than from a global maximum-speed estimate.
- Exact zero face flux on outer boundaries and fluid–solid interfaces.
- Direct convolution as the reference definition, including `dx*dy`.
- Cached FFT linear convolution for routine evolution.
- Adaptive positivity-CFL substeps for diffusion and drift.
- No density clipping or mass renormalization.  Negative, non-finite, or
  non-conservative states raise an exception; minimum density, mass error,
  maximum flux, and the fixed-control free-energy diagnostic are recorded.
- The weak form is checked by discrete summation by parts.

The solver assumes a translation-invariant pair potential on a rectangular
Cartesian embedding. Obstacles remove fluid cells and close faces. More
accurate curved boundaries remain future work.  For smooth, adequately resolved
test problems, the SG path is second order in space and SSP-RK2 is second order
in time.  Sharp real contact forces, obstacles, and any future calibrated
potential require their own convergence evidence; the included result is not a
claim of universal second-order accuracy.

## Microscopic validation

The overdamped particle model is

```text
dX_i = M[F_ext(X_i) + F_control(X_i;u)
       + (1/N) sum_(j != i) F_pair(X_i-X_j)] dt + sqrt(2D) dB_i,
F_pair(r) = -grad W(r).
```

That displayed equation is the probability/Kac form. The equivalent physical
single-pair form replaces the interaction term by
`sum_(j != i) F_pair(X_i-X_j)` and uses number density in the continuum.

The code also contains the second-order mother equation only to document the
overdamped reduction and validate force/energy interfaces. It is not the main
state equation.

## Three independently reviewable modules

The project follows the same three-module management pattern as the reference
`mckean_vlasov_control` project, adapted to the current mechanics-first phase:

```text
MV Physics Engine
  src/mechanistic_mv/mechanics/          (except controlled_potential.py)
  src/mechanistic_mv/particle_sim/
  src/mechanistic_mv/continuum/          (except diagnostics.py)

MV Controller Contract
  src/mechanistic_mv/mechanics/controlled_potential.py
  docs/CONTROL_INTERFACE.md

MV Experiment Lab
  src/mechanistic_mv/continuum/diagnostics.py
  scripts/
  tests/                              (evaluation-focused tests)
  outputs/validation/
```

Detailed ownership rules are in `AGENTS.md`. Shared project files such as
`README.md`, `AGENTS.md`, packaging metadata, and the optimization log require
coordinator integration. The three `mv/*` branches preserve reviewable module
history inside this one repository; they do not create extra projects or
folders. The Controller module remains a physical control-contract module until
RL is explicitly approved.

## Run validation

From this directory in PowerShell:

```powershell
$env:PYTHONPATH = "src"
& "D:\conda environment\envs\dl\python.exe" -m unittest discover -s tests -v
& "D:\conda environment\envs\dl\python.exe" scripts\run_validation.py
& "D:\conda environment\envs\dl\python.exe" scripts\run_convergence_validation.py
& "D:\conda environment\envs\dl\python.exe" scripts\run_second_order_flux_validation.py
& "D:\conda environment\envs\dl\python.exe" scripts\benchmark_optimizations.py
& "D:\conda environment\envs\dl\python.exe" scripts\run_hydrogel_validation.py --test-only-fixture
& "D:\conda environment\envs\dl\python.exe" scripts\run_pair_contact_sweep.py
& "D:\conda environment\envs\dl\python.exe" scripts\build_effective_potential.py --test-only-fixture
& "D:\conda environment\envs\dl\python.exe" scripts\validate_density_scaling.py --test-only-fixture
```

These commands run tests and short physics simulations; they do not train a
controller. The `--test-only-fixture` commands validate code with visibly
uncalibrated values. `run_pair_contact_sweep.py` currently exits `BLOCKED` and
does not create a fake force curve. Use each new script's `--help` for the
required real-data inputs. Reproducible artifacts are written to
`outputs/validation/`.

## Collecting real contact inputs later

When calibrated values and provenance are available, first create a blank
input manifest. It contains 110 leaf fields, all `null`; it is a collection
form, not physical data and not a test fixture.

```powershell
$env:PYTHONPATH = "src"
& "D:\conda environment\envs\dl\python.exe" scripts\run_pair_contact_sweep.py `
  --write-input-template outputs\validation\pair_interaction\my_contact_input.json
```

Fill every required field with its physical value or text record, its source
and verification record, then preflight it:

```powershell
& "D:\conda environment\envs\dl\python.exe" scripts\run_pair_contact_sweep.py `
  --input-manifest outputs\validation\pair_interaction\my_contact_input.json
```

The preflight writes a field-level status report and the SHA-256 of the exact
manifest bytes. Missing, unverified, malformed, or `TEST_ONLY` input remains
`BLOCKED`. A fully verified physical manifest can reach
`READY_FOR_CONTACT_SOLVER`, but the current report still says
`CONTACT_FEM_BACKEND_UNAVAILABLE` and `CONTACT_RESULTS_NOT_GENERATED`, produces
no `pair_force.csv` or curve, and exits blocked until a genuine FEM/contact
backend is implemented.

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
8. Pure-diffusion grid refinement, complete-MV CFL refinement, and a five-seed
   particle-vs-MV robustness study with explicit pass/fail gates.
9. Particle-count convergence at 250, 500, and 1000 particles over five fixed
   seeds, including empirical L2 and Jensen--Shannon convergence orders.
10. Hydrogel Gibbs/Piola finite-difference consistency and chemical conjugacy.
11. Full pair-contact record integrity, solver/mesh provenance, and fail-closed
    handling of missing FEM data.
12. Shape-preserving force integration, `F_pair = -dW_eff/dr`, endpoint policy,
    and Kac-scaling admission gates.
13. A fixed-seed, explicitly test-only shared-potential particle/MV regression;
    it is a code check, not a statistical or material-validation claim.
14. Probability/number-density equivalence for mass, particle force and energy,
    seeded Langevin motion, direct/FFT convolution, flux, one FVM step, and
    free energy, plus explicit mismatch rejection.
15. Short-range admission: `r_min>0` remains particle-only and is blocked before
    any particle-versus-MV comparison; `r_min=0` is continuum-ready only after
    the other physical, provenance, time-scale, and scaling gates pass.

See `OPTIMIZATION_LOG.md` for every performance change and its measured effect.

## Reference and completion boundary

The Hydrogel equations are ported from the supplied Guo slides describing the
Hong et al. Model II initial-swelling formulation. The PDF is a scientific
source, not runtime instructions, and is not copied into Git. The code uses the
page-6 expanded Gibbs density together with the page-7 stress relation; it
records the printed reference-volume ambiguity rather than claiming an
unsupported independent `Psi(F, C)` law.

The current physical milestone is not complete until all of the following are
provided and validated: calibrated Hydrogel parameters, particle geometry,
contact FEM/mesh convergence, mechanical and solvent-bath boundary conditions,
`tau_gel << tau_swarm`, distance-sweep data, negligible-force reference,
single-pair-to-Kac scaling, and a real non-magnetic controlled potential.
Until then, the repository is a validated implementation framework with
test-only numerical evidence, not a calibrated experiment model.
