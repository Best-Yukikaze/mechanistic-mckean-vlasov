# MV physical validation source provenance v1

Status: **SOURCE_AUDIT_COMPLETE_REQUIRED_SAME_SYSTEM_INPUTS_MISSING**

This register is source-first and fail-closed. A recorded value is not a
calibrated project parameter unless its `admissibility` and transfer evidence
say so. `source_sha256` is optional archival metadata and is not an admission
gate.

## Source roles

| ID | Source | Admitted role | Important limit |
|---|---|---|---|
| S1 | Lyons et al. 2026, DOI 10.1039/D6NA00088F | Same-system drift observations | Table 1 LG/IG/HG velocities lack condition- and batch-matched magnetic-force tuples in the accessible article. |
| S1b | Stephen Lyons thesis, 2020 | Magnetic-parameter candidate provenance | Batch/coating/magnet compatibility with S1 2026 is not documented; its values are not transferred. |
| S2 | Fatin-Rouge et al. 2004, DOI 10.1016/S0006-3495(04)74325-8 | Independent agarose hindrance/regime evidence | Different tracers and 1.5% agarose; not a calibration for S1. |
| S2b | Moncure et al. 2022, DOI 10.1021/acs.jpcb.2c00771 | Independent particle/mesh sanity check | Au/PAAm system; no pointwise numeric SI table was acquired. |
| S3 | Myrovali et al. 2016, DOI 10.1038/srep37934 | Dipolar-chain regime and ordering evidence | Alignment occurs during agarose cool-down; not a fixed-gel quantitative transport benchmark. |
| S4 | Basak et al. 2009, DOI 10.2147/IJN.S4114 | Joint semiquantitative optical comparator candidate | Figure 7 is normalized optical intensity and the paper calls gel transport qualitative. It is not calibrated number density. |

## Fail-closed transfer findings

- The S1b values `chi_v=0.281`, `V_m=3.69e-25 m^3`, `B=0.55 T` and
  `grad(B)=45 T/m` remain valid descriptions of the thesis calculation. They
  are **not** paired with the S1 2026 Table 1 observations.
- S2 and S2b constrain plausible hindrance regimes only. Their parameters are
  not copied into S1.
- S3 supplies qualitative size/concentration/field ordering, but not the
  moment/contact/slab inputs needed for a dimensionally closed 2-D `W_dd`.
- S4 Figure 7 can in principle be manually digitized as normalized optical
  intensity. It cannot be admitted as `rho` without a calibrated observation
  model and pointwise uncertainty, and digitization cannot repair missing
  upstream `D/M`, `V_mag` and `W_dd` closure.

## Missing inputs that block physical execution

1. Same-batch magnetic moment/susceptibility and material volume for both S1
   particle classes.
2. Condition-specific `B`, `grad(B)` or `B grad(B)` for every S1 LG/IG/HG row,
   with uncertainty.
3. A same-system passive diffusivity or a predeclared, source-compatible
   hindrance transfer model.
4. Source-backed slab thickness, vertical distribution and physical
   zero-separation/contact closure for the current 2-D density model.
5. S3 moment/contact/density inputs sufficient to compute `lambda_dd` without
   a free multiplier.
6. S4 particle magnetic properties and a calibrated optical observation model
   or raw data adequate for a parameter-free joint test.

No magnetic PDE, interaction simulation, fit, Gym evaluation or training is
represented by this register.
