# Optimization log

Every optimization below preserves a slower or analytical reference check. No
optimization changes the physical model or promotes the test potentials to
real constitutive laws.

## O1 — Chunked mean-field particle force

- **Changed:** evaluate the `N x N x 2` displacement tensor in row chunks.
- **Reason:** bound peak temporary memory while retaining the exact `1/N` sum.
- **Verification:** chunk sizes 64, 128, 256, and 1200 gave zero numerical
  difference for 1200 particles.
- **Measured result:** the displacement array bound falls from 23.04 MB at a
  full 1200-row block to 1.23 MB at 64 rows. All measured chunk runtimes are
  close on the recorded 1200-particle case; 64 rows is the lowest-memory tested
  setting and is used as the default after this measurement.

## O2 — Zero-interaction fast path

- **Changed:** `W=0` returns a zero force array without constructing pair
  displacements.
- **Reason:** diffusion/Fokker–Planck ablations should not pay `O(N²)` cost.
- **Verification:** 100,000 particles returned exactly zero force.
- **Measured result:** sub-millisecond in the saved benchmark. Exact timing is
  kept only in the JSON because it varies between runs.

## O3 — Frequency-cached FFT physical convolution

- **Changed:** build the translation-invariant `W` offset kernel once, choose a
  zero-padded fast transform shape, and cache the kernel's real-valued frequency
  transform. Each call now transforms only the changing density. The direct
  `O(cells²)` definition and spatial-kernel-cached `fftconvolve` benchmark remain
  as independent references.
- **Reason:** `W*rho` is recomputed at every MV substep.
- **Verification:** direct and FFT outputs agree within about `1.6e-37 J`, and
  tests explicitly check the required `dx*dy` factor and kernel alignment.
- **Measured result:** the saved benchmark reports both direct-to-cached and
  `fftconvolve`-to-frequency-cache speedups. Exact timings are intentionally not
  duplicated here; the versioned JSON is the authoritative record.

## O3b — Zero-interaction continuum fast path

- **Changed:** a `ZeroPairPotential` returns an exact zero cell array without
  building a kernel or running any transform.
- **Reason:** pure diffusion and Fokker–Planck ablations must not pay for a
  nonlocal interaction that is mathematically absent.
- **Verification:** the direct and optimized zero-potential paths return exact
  zeros; the unit and benchmark cases cover repeated calls.
- **Measured result:** the 128² per-call timing is stored in the benchmark JSON.

## O4 — Cached geometry and face masks

- **Changed:** fluid cells and open x/y faces are computed once per solver.
- **Reason:** obstacle topology is static during a rollout.
- **Verification:** all closed faces are exactly zero in the obstacle test;
  solid density remains exactly zero and mass error stays at roundoff.
- **Measured result:** removes repeated boolean geometry construction from every
  flux evaluation. This micro-optimization was not timed separately because it
  is dominated by convolution on the current grids.

## O5 — Vectorized face fluxes

- **Changed:** gradients, upwind selection, face closing, divergence, and
  diagnostics use NumPy array operations rather than Python cell loops.
- **Reason:** flux construction runs every substep and should scale with grids.
- **Verification:** weak-form residual is roundoff-level; pure-diffusion
  variance matches the analytic law to roughly `1.7e-9` relative error.
- **Measured result:** included in end-to-end validation; no separate baseline
  loop is kept because the vectorized implementation is the tested reference.

## O6 — Adaptive diffusion-and-drift CFL substeps

- **Changed:** each requested interval is split using the current maximum face
  drift and the explicit diffusion stability rate.
- **Reason:** external/pair forces change with density, so one fixed step can be
  safe in one scenario and negative in another.
- **Verification:** five continuum scenarios finish with no clipped negative
  mass; maximum absolute mass error is `1.11e-16`.
- **Measured result:** the 0.5 s validation horizon uses 10 substeps for
  diffusion/pair-only cases and 30 for stronger external/full/obstacle cases.

## O7 — Static controlled-potential reuse

- **Changed:** evaluate `V_control(x;u)` once per requested solver step and reuse
  it across all adaptive MV substeps while the held input `u` is unchanged.
- **Reason:** the RL/control time-scale contract holds one command across those
  substeps, so recomputing the same field has no physical or numerical benefit.
- **Verification:** a counting test forces multiple FVM substeps and observes
  exactly one controlled-potential evaluation. Shape and finite-value checks
  reject malformed backends before flux construction.
- **Measured result:** benefit scales with the cost of the future real actuator
  backend and the number of adaptive substeps; no fake actuator is benchmarked.

## O8 — Complete-displacement specular no-flux reflection

- **Changed:** particle boundary handling now advances to the earliest outer or
  rectangular-solid face, reflects the untravelled displacement, and continues
  until the complete integration-step displacement has been resolved. Corner
  contacts reflect both face-normal components. A vectorized zero-collision
  fast path remains active when there are no obstacles and all proposals stay
  inside the outer domain.
- **Reason:** stopping at first obstacle contact discarded tangential and
  post-impact motion, creating an artificial residence-time bias next to solid
  boundaries. Reflecting the remaining displacement is the specular discrete
  realization of particle no penetration used by this backend.
- **Verification:** analytical oblique and obstacle-then-wall trajectories have
  exact expected endpoints; 200 randomized large displacements and 300
  Brownian particles over 12 steps remain in the fluid and preserve unit
  empirical mass.
- **Measured result:** no separate timing claim is made. The common
  obstacle-free, no-collision path returns before the per-particle swept-path
  solver, while colliding paths now pay only for the reflections they require.

## Reproduce measurements

Run `scripts/benchmark_optimizations.py`. Exact wall-clock values vary by
machine; `outputs/validation/optimization_benchmark.json` is the authoritative
record and includes hardware/software metadata, timing policy, and Git revision.
