# Lyons constitutive precheck v2

Status: **LYONS_CONSTITUTIVE_DATA_INSUFFICIENT**

Project status preserved: **CURRENT_2D_MV_PHYSICAL_REDUCTION_INVALID** (not recomputed by this source audit).

Lineage: `LYONS_BROUGHAM_2020_PEG1000_CORE8P9_AGAROSE0P3`. No 2026 PEG2000 data, no PDE/Gym/RL, no phi fit.

## Source-specific conditional arithmetic (not admitted physical parameters)

| Recipe | F nominal [N] | M_front apparent [m/(N s)] | velocity-SD-only | apparent D/D0 |
|---|---:|---:|---:|---:|
| chi_manuscript | 2.10033973e-18 | 4.89338825e+10 | 2.64507473e+09 | 9.851039 |
| chi_thesis | 2.04219884e-18 | 5.03270180e+10 | 2.72037935e+09 | 10.131496 |

The force uncertainty is unknown. The displayed SD propagates only the reported
front-velocity SD and is not full uncertainty or a confidence interval. D/T is
reported in JSON, but D and D0 at transport temperature remain null. The apparent
ratios exceed 1.25 only under unverified front=drift and Einstein assumptions.
They expose a consistency problem to investigate, not a measured passive diffusivity.

## Source and observation checks

- chi=0.281 (thesis) versus 0.289 (manuscript/ESI) remain separate.
- 0.23 T susceptibility reference is not the 0.55 T nominal magnet field.
- The 2021 correction requires v*d, in mm^2/h, not v/d.
- Single-force repeated gels cannot determine mobility CV across fields.
- Figure 1: 16 digitized manuscript markers give front speed 0.37316638 mm/h, R2=0.99666175, residual RMSE=0.04987747 mm.
- These are rendered-figure coordinates, not raw replicate CSV. Localization bounds
  and axis calibration are stored; no replicate uncertainty is reconstructed.

- The 6 mm gel height does not establish a depth-integrated 2D closure.

## Missing inputs

- Reconcile 0.281 vs 0.289 using same-batch magnetometry, units, temperature and uncertainty.
- Map B(x) and gradB(x), including covariance/position uncertainty; distinguish 0.23 T reference from 0.55 T nominal.
- Obtain raw per-gel front times/positions and front threshold or calibrated concentration/centroid profiles.
- Obtain multiple independent magnetic-force conditions for the same PEG1000 batch and 0.3% agarose.
- Confirm transport temperature, passive diffusion and Einstein-relation applicability.

## Inspected sources

- [RSC2020](https://pubs.rsc.org/en/content/articlelanding/2020/nr/d0nr01602k): Nanoscale 12 (2020), 10550-10558; author manuscript and official ESI used for accessible details; PUBLISHER_METADATA_CHECKED_FULL_PUBLISHER_TEXT_NOT_RETRIEVED.
- [MANUSCRIPT2020](https://doras.dcu.ie/25059/1/DORAS%20submission.pdf): DORAS item 25059; 8-page manuscript; PDF pp.3-5, Eq.1, Fig.1, Table1; PRIMARY_AUTHOR_MANUSCRIPT_TEXT_AND_PDF_PAGES3_4_VISUALLY_CHECKED.
- [ESI2020](https://www.rsc.org/suppdata/d0/nr/d0nr01602k/d0nr01602k1.pdf): p.2 Figs.S1-S2; p.7 Figs.S7-S8; p.8 Figs.S9-S10; p.9 Figs.S11-S12; p.10 Fig.S14; PRIMARY_INDEXED_PDF_TEXT_CHECKED_ALL_11_PAGES_LIVE_DOWNLOAD_RETURNED_404.
- [THESIS2020](https://doras.dcu.ie/25019/2/Stephen%20Lyons%20Thesis%20Sept%2015th.pdf): PDF p.17 publication list; p.71 core size; p.78 Fig.2.13; p.79 Fig.2.14; pp.95-96 Eq.3.1 and comparison; p.98 Table3.1. PDF page is 1-based, distinct from printed page.; PRIMARY_THESIS_TEXT_CHECKED.
- [CORRECTION2021](https://pubs.rsc.org/en/content/articlehtml/2021/nr/d0nr90262d): Nanoscale 13 (2021), 1365-1366; corrections to Fig.3/Fig.4 and p.10555; PRIMARY_CORRECTION_INDEXED_TEXT_CHECKED.

## Author data request draft - NOT SENT

Subject: PEG1000 8.9 nm-core agarose magnetophoresis: source data and force/front calibration

Dear Dr Lyons, Prof Brougham and Prof Morrin,

We are auditing the constitutive interpretation of the PEG1000, 8.9 nm-core,
0.3% w/v low-EEO agarose/DI-water system in Nanoscale 2020,
DOI 10.1039/D0NR01602K (and correction 10.1039/D0NR90262D), together with
chapters 2-3 of the Lyons thesis. We are not pooling the 2026 PEG2000 batches.

Could you share, or point us to an existing public deposit containing:

1. Figure 1 and ESI S8-S10/S12-S13 per-gel time/front-position CSV files,
   replicate IDs, initial-point exclusions and slope-fit uncertainties; the
   original images/profiles and ImageJ front threshold/scale procedure.
2. The relevant synthesis/measurement batch map, TEM diameter distribution,
   DLS uncertainty and magnetometry M(H)/M(B) data with unit and normalization
   conventions. The thesis quotes chi=0.281, whereas manuscript/ESI gives
   0.289. ESI S2 refers to B=0.23 T at 6 mm, but the manuscript/thesis equation
   also quotes the magnet's nominal 0.55 T and gradB=45 T/m. Which values and
   magnet geometry apply to each measured trajectory, and with what errors?
3. B(x) and gradB(x) or an original field map/model and uncertainty over the
   front's full path; results at independently varied force levels for the
   same particle batch/gel, if available. Corner-repeatability data alone
   cannot establish the CV of mobility across force strengths.
4. Transport-run temperature, medium viscosity, no-field front-width or
   independently measured diffusivity, and centroid/mean-drift information
   that could test whether optical-front velocity equals mean particle drift.
5. Clarification of the reported no-phi theoretical velocity and its exact
   force/viscosity inputs; we will not fit a tortuosity phi to close a mismatch.

We can use anonymized replicate IDs and will preserve the distinction between
published summaries, digitized points and raw measurements, with your preferred
data citation and reuse terms. This request is a draft only and has not been sent.
