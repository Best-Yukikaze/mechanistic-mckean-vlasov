# Hydrogel constitutive source and limits

The formal material source is the user-supplied PDF at
`C:/Users/陈聿闽/Desktop/diffusion in gels - New.pdf`. The PDF itself is not
copied into Git.

The implemented law is the Hong/Guo Model II initial-swelling Gibbs
formulation shown on pages 6-8. Its named inputs are:

```text
N*nu                         dimensionless network parameter
chi                          dimensionless Flory-Huggins parameter
phi0                         initial polymer volume fraction
Delta mu/(kT)                dimensionless bath-potential increment
kT/nu                        energy-density/stress scale in Pa
```

The source supplies parameter names and equations but no calibrated numerical
values for this project. Code therefore requires all values explicitly; test
values are labelled `TEST_ONLY_NOT_CALIBRATED`.

Implemented source relations include the page-6 expanded Gibbs increment, the
page-7 first Piola stress, and the page-8 concentration/reference conversion:

```text
P = partial G / partial F
-partial G / partial(Delta mu/kT) = J - phi0
nu C_dry = (J - phi0) / phi0
```

The source eliminates concentration through the swelling constraint and does
not uniquely specify an independent `Psi(F,C)` closure for this repository.
That unsupported operation raises `NotImplementedError` rather than silently
substituting another law. The page-6 printed reference-volume normalization is
also recorded as an ambiguity; the implementation uses the expanded expression
that differentiates consistently to the page-7 stress.

Hydrogel `F`, `P`, `C`, and its chemical conjugate remain below the group
scale. They reach McKean-Vlasov only through validated two-particle mechanics,
`F_pair(r)`, and `W_eff(r)`; they are never inserted into `rho` directly.

The source does not supply project-specific radii, contact geometry, FEM mesh,
mechanical constraints, solvent-bath boundary conditions, time scales, or a
distance sweep. Those missing inputs keep the real contact workflow `BLOCKED`.
