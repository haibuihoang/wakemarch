# Scientific scope and evidence

WakeMarch 0.3 implements the single-turbine model of the accompanying paper: thrust
exponent 2, force-based circulation, projected initial deficit, momentum fraction
0.95, eddy viscosity delayed over the near-wake length with A = 2, vortex diffusivity
factor 0.25, and the near field of the rotor displayed as a superposition of
actuator-disk vortex cylinders. Optional inputs switch the hub term, the
near-field display, the wake rotation and the blockage of a confined domain;
the package does not expose the options tested and not adopted during the
development.

The intended first audience is researchers. The model has known near-wake,
high-turbulence and high-yaw/far-wake limitations. It does not resolve wake
pressure gradients or ground-boundary-layer dynamics. Input guards are not an
experimental validation envelope. Engineering and operational use requires
application-specific assessment; no farm solver or active downstream-turbine
model is provided.

Advection is approximately conservative; variable-coefficient diffusion does not
establish exact circulation conservation. Force-based initialization does not
guarantee exact equality of the discrete vortex integral and its target. Better
empirical predictions are not proof of exact conservation or force consistency.

## What the software checks establish

- API tests: input conventions, sampling, progress, independent results and
  diffusion residual checks.
- Research-solver comparison: the march agrees with the authors' research solver
  on eight cases to a relative difference of 1.5e-9. This is implementation
  agreement, not an observational error bar.
- Grid notebook: differences between numerical resolutions for a specified case.
  Its finest grid is a reference, not a converged solution or observation.
- Historical numerical tests: manufactured operators, symmetry and approximate
  invariants, subject to their stated tolerances.
- Historical measurement comparisons: scientific evidence with calibration and
  preprocessing choices described in the research reports, not recreated by a
  successful package installation.

## Data and validation boundary

The accompanying paper compares the model with PALM LES, hub-height PIV,
Oldenburg lidar and NTNU/ForWind LDA data. The LES data used in the paper are not
distributed with this package. Other datasets have their own access and license
terms; a code license cannot grant redistribution rights to them. No measurement
files are included in the wheel or source distribution.

The optional [PIV measurement notebook](../examples/wakemarch_piv_validation.ipynb)
uses [Bastankhah and Porté-Agel's Zenodo v1 dataset](https://zenodo.org/records/3607598),
DOI 10.5281/zenodo.3607598. Obtain the HDF5 file separately and inspect its terms;
the notebook checks the record's MD5 and records SHA-256. No data are bundled.
It inherits the historical unyawed CT=0.9 and k=0.024 calibration, converts k to
equivalent TI and performs no new fit. These cases were previously examined
during model development: this is not a new blind test. Its same-point RMS
comparison differs from historical scoreboard preprocessing and metrics.
Do not label the introductory or grid notebooks as measurement validation.
Record geometry, inflow/CT values, yaw sign and coordinate mapping, uncertainty
and calibration exposure. Evaluate model and data on the same measurement support.
For tower/floor-contaminated laboratory planes, document the rotor-band mask.

Do not silently substitute synthetic data when measurements are missing. Farm
validation additionally requires turbine interaction, local inflow and thrust
definitions; the single-wake downstream rotor u³ proxy is insufficient.

## Paper linkage

`release-metadata.json` records publication metadata. The results of the paper
were computed with the kernel of this package (release candidate 0.3.0rc2,
numerically identical to 0.3.0); the ablation variants of the paper use the
research solver. Until the paper reference is available, see
[citation](../CITATION.md).

## Reference turbine performance inputs

The package now includes reference power/CT tables and a rotor-equivalent
operating model; see [TURBINES.md](TURBINES.md). This adds no farm validation or
new wake physics. CP converted from the power tables is electrical-equivalent.
User CP inputs require an explicit efficiency convention. Effective-speed yaw
power mapping is an approximation, not a controller simulation.
