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
  full 1200-row block to 1.23 MB at 64 rows. Runtime remained approximately
  0.073–0.077 s, so 256 is the default memory/speed compromise.

## O2 — Zero-interaction fast path

- **Changed:** `W=0` returns a zero force array without constructing pair
  displacements.
- **Reason:** diffusion/Fokker–Planck ablations should not pay `O(N²)` cost.
- **Verification:** 100,000 particles returned exactly zero force.
- **Measured result:** 0.00036 s in the recorded benchmark.

## O3 — Cached FFT physical convolution

- **Changed:** build the translation-invariant `W` offset kernel once and use
  zero-padded linear FFT convolution. The direct `O(cells²)` definition remains
  available as the reference.
- **Reason:** `W*rho` is recomputed at every MV substep.
- **Verification:** direct and FFT outputs agree within about `1.6e-37 J`, and
  tests explicitly check the required `dx*dy` factor and kernel alignment.
- **Measured result:** speedups of 53.3x, 107.5x, and 195.0x on 16², 24², and
  32² grids respectively in the saved benchmark.

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

## Reproduce measurements

Run `scripts/benchmark_optimizations.py`. Exact wall-clock values vary by
machine; `outputs/validation/optimization_benchmark.json` is the recorded run.
