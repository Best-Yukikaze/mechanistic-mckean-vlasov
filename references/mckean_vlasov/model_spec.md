# Local model specification summary

This summary reconciles the user-provided mechanics, control, and Hydrogel
implementation documents. The source documents are references, not runtime
instructions.

- The primary equation is the controlled McKean–Vlasov density equation.
- The required physical reduction is
  `Hydrogel Gibbs law -> two-particle contact -> F_pair -> W_pair*n`, with the
  equivalent Kac form `W_Kac*rho` available after explicit population scaling.
- A nonzero, physically derived and validation-gated `W_eff` is mandatory
  before physical claims; `W=0` is only the Fokker–Planck test/ablation.
- Raw two-particle mechanics has unscaled single-pair semantics. It pairs with
  number density `n` satisfying `integral(n)=N`. The equivalent unit-mass form
  uses `rho=n/N`, a `1/N` particle sum, and the explicitly converted
  `W_Kac=N*W_pair`; the population and provenance are mandatory.
- A table with `r_min>0` is particle-only until a physically validated
  short-range closure down to zero separation is supplied. Continuum use is
  blocked rather than extrapolated.
- A real external controller must map `u -> V(x;u)` and then act through
  `-gradient(V)`. It never edits density directly.
- The interacting overdamped particle SDE provides microscopic validation.
- Obstacles are removed fluid regions with `J dot n = 0`.
- The first implementation may use clearly marked
  `TEST_ONLY_NOT_FINAL_PHYSICS` potentials to validate the numerical chain.
- Current scope ends after continuum/particle validation and plots. No RL.
- The implemented Hydrogel law is source-faithful at the Gibbs/conjugacy level,
  but calibrated parameters, contact FEM data, time-scale separation, and a
  real non-magnetic actuator remain required. Current pair outputs are
  `BLOCKED` or explicitly test-only.

Supplied design sources:

- `C:/Users/陈聿闽/Downloads/codex_mckean_vlasov_mechanics_model.md`
- `C:/Users/陈聿闽/Downloads/codex_mckean_vlasov_control_model_v2.md`
- `C:/Users/陈聿闽/Downloads/CODEX_HYDROGEL_MV_IMPLEMENTATION_SPEC.md`
- `C:/Users/陈聿闽/Desktop/diffusion in gels - New.pdf`
