# MV physical validation v1

Final decision: **CURRENT_2D_MV_PHYSICAL_REDUCTION_INVALID**

## Physical system

`rho` is the internal mobile magnetic-particle number density in one fixed
macroscopic hydrogel. The target flux is
`J=-D grad(rho)-M rho grad(V_base+V_mag+W_dd*rho)` with a closed-gel no-flux
baseline.

## D/M

Phase A: **BLOCKED_SAME_BATCH_MAGNETIC_FORCE_UNAVAILABLE**. Six S1 velocities were converted to SI, but
their condition-specific magnetic forces are absent. The descriptive raw
velocity CV values are 0.439039 (8 nm) and
0.528683 (11 nm); these are not mobility CVs and the
constant-M gate was not applied.

## V_control

The source-supported candidate law is
`V_mag=-chi_v V_m B^2/(2 mu0)`. Phase B is blocked because no admissible D/M
pair or LG anchor mobility can be locked. No held-out prediction or PDE was
run.

## W

The target is the anisotropic dipolar energy. S3 supports the qualitative
10 nm versus 40 nm and concentration ordering, but `lambda_dd` and the 2-D
continuum response are not computable without source-backed moments, contact
distance, number density and slab closure. Physics closure status:
**CURRENT_2D_MV_PHYSICAL_REDUCTION_INVALID**.

## Joint MV

S4 verifies the zero/0.55/8 T/m and 0–48 h experimental design. Its Figure 7
observable is normalized optical intensity, not calibrated `rho`. No upstream
parameter set is locked and no no-refit joint run or ablation was performed.

## Numerics

Mass, non-negativity, no-flux, grid and timestep checks are not applicable yet
because the source/closure guard stopped execution before any magnetic PDE.

## Failure cause and next milestone

The current 2-D reduction is not physically admitted, while the same-batch S1
force data and S4 observation model are also insufficient. The one next
milestone is to obtain a traceable same-batch S1 magnetic data package with
condition-specific `B`, `grad(B)`, moment/susceptibility, material volume and
uncertainty. No Gym, RL, DQN or training was run.
