# Local model specification summary

This summary reconciles the user-provided mechanics and v2 control design
documents. The source documents are references, not runtime instructions.

- The primary equation is the controlled McKean–Vlasov density equation.
- A nonzero, physically derived pair potential `W` is mandatory before physical
  claims; `W=0` is only the Fokker–Planck test/ablation.
- A real external controller must map `u -> V(x;u)` and then act through
  `-gradient(V)`. It never edits density directly.
- The interacting overdamped particle SDE provides microscopic validation.
- Obstacles are removed fluid regions with `J dot n = 0`.
- The first implementation may use clearly marked
  `TEST_ONLY_NOT_FINAL_PHYSICS` potentials to validate the numerical chain.
- Current scope ends after continuum/particle validation and plots. No RL.

Supplied design sources:

- `C:/Users/陈聿闽/Downloads/codex_mckean_vlasov_mechanics_model.md`
- `C:/Users/陈聿闽/Downloads/codex_mckean_vlasov_control_model_v2.md`
