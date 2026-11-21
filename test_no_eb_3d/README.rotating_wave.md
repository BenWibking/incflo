# Axisymmetric base flow and upper-half-space rotating waves

## Equilibrate the axisymmetric flow

The checked-in `axisymmetric_base.npz` contains the 2,804 velocity coefficients of the reference-torque equilibrium and its independent verification measurements. From the repository root, render it with

```bash
python3 test_no_eb_3d/plot_axisymmetric.py test_no_eb_3d/axisymmetric_base.npz --out axisymmetric.png
```

`solve_axisymmetric.py` solves the steady, forced similarity equations in full space at fixed `g=0.502` and unit viscosity. The radial velocity is even across the equator and the axial velocity and swirl are odd. All meridional motion develops through the equations. The prescribed body force has only an azimuthal component, with cylindrical coordinates `r,z` and amplitude `a`.

```text
F_phi = a (r/R) (z/Z) exp[1 - ((r/R)^2 + (z/Z)^2)/2]
```

The default widths are `R=19` and `Z=21` in similarity units. This Gaussian torque is a modeling choice. Each steady solve keeps its coefficient fixed. The reference coefficient for a sweep gives the digitized mean swirl in the *linearized* steady response, using an average over the observed area weighted by the torque profile. The actual nonlinear swirl is a prediction. The figure initialization uses all three components of the reconstructed axisymmetric mean. The Stokes initialization uses the linear forced solution. Increasing and decreasing torque sequences let the two starts meet at identical physical controls.

The divergence-free meridional splines extend to infinity. Damped Gauss–Newton steps minimize the full-space momentum residual after a least-squares pressure fit. The azimuthal integration is analytic for an axisymmetric field. Radial and polar quadrature each use at least 128 nodes. Each stage is checked on independent finer quadratures and with two higher pressure degrees. The default numerical balance criterion requires the defect on every finer check to be at most 3% of the applied force, with agreement to 0.3 percentage points between those checks. The report separately records the magnitude of the force relative to the pressure-projected linear and nonlinear terms, the unforced defect, plane inflow and axial jet velocity.

```bash
driver=src/test_no_eb_3d/setonix/rotating-wave.sbatch
sbatch -A YOUR_PROJECT-gpu -p gpu-highmem "$driver" axisymmetric \
    --seed figure --fractions 1 .5 .25 .1 .04 --iterations 48
sbatch -A YOUR_PROJECT-gpu -p gpu-highmem "$driver" axisymmetric \
    --seed stokes --fractions .04 .1 .25 .5 1 --iterations 48
coarse=axisymmetric-STOKES_JOB_ID-0
sbatch -A YOUR_PROJECT-gpu -p gpu-highmem "$driver" axisymmetric \
    --restart "$coarse/stage-00.npz" --degree 5 --fractions .04 .1 .25 .5 --iterations 85
fine=axisymmetric-REFINEMENT_JOB_ID-0
sbatch -A YOUR_PROJECT-gpu -p gpu-highmem "$driver" axisymmetric \
    --audit-only "$fine/stage-03.npz" --compare "$coarse/stage-03.npz" --grid 192 --audit-grids 224 256
python3 src/test_no_eb_3d/plot_axisymmetric.py "$fine/stage-03.npz" \
    --history "$coarse/axisymmetric.json" --history "$fine/axisymmetric.json" --out axisymmetric.png
```

Restarts inherit the checkpoint's geometry, polynomial degree, knots and physical force. Supply `--fractions` for a new torque sequence or specify higher degree and additional knots for refinement. The comparison command checks that both checkpoints have the same force, widths, viscosity and dilation. It measures the relative velocity difference both in the observed region and with a core weight proportional to `(1 + (|y|/50)^2)^-2`.

Starting at the fifth-degree `$fine/stage-03.npz` from the example, the following two refinements reproduce the reference-torque field checked on Setonix.

```bash
knots=(--sigma-breaks 0 .18 .36 .45 .54 .63 .73 .88 1 --eta-breaks 0 .06 .12 .2 .3 .53 .76 .88 1)
sbatch -A YOUR_PROJECT-gpu -p gpu-highmem "$driver" axisymmetric --restart "$fine/stage-03.npz" \
    --degree 5 "${knots[@]}" --fractions .5 1 2 --iterations 85
hp5=axisymmetric-MESH_JOB_ID-0
sbatch -A YOUR_PROJECT-gpu -p gpu-highmem "$driver" axisymmetric --restart "$hp5/stage-01.npz" \
    --degree 6 "${knots[@]}" --fractions 1 .5 1.5 2 --iterations 80
hp6=axisymmetric-DEGREE_JOB_ID-0
sbatch -A YOUR_PROJECT-gpu -p gpu-highmem "$driver" axisymmetric --audit-only "$hp6/stage-00.npz" \
    --compare "$hp5/stage-01.npz" --grid 192 --audit-grids 224 256
```

At physical torque amplitude 118.45044, the degree-six field had inward radial velocity −51.33 at `(r,z)=(30,0)` and axial velocity +140.44 at `(0,24)`. Its independent 256-by-256 meridional check with degree-nine pressure measured an imbalance of 1.215% of the applied force. Relative velocity changes from the degree-five field were 3.71% over the observed region and 4.27% in the core-weighted norm. The drive was 44.1% of the sum of the pressure-projected linear and nonlinear term norms. This supplies a forced equilibrium for the modeled torque. The authors’ intermediate axisymmetric field and its force are unavailable in the public release.

At half that torque, the lower-swirl state had weighted mean swirl 43.92 and a 0.171% momentum imbalance. Its velocity changed by 1.13% in the observed region from degree five to six. On the same degree-six mesh a downward continuation also produced a larger-swirl trial, with mean 66.18 and 2.33% imbalance. At degree five both directions returned to mean 43.78. The larger-swirl branch remains unconfirmed.

## Single rotating wave

`solve_rotating_wave.py` searches the auxiliary problem described in Appendix C.3. In the upper half-space it uses an axisymmetric mean and harmonics of one traveling phase `kappa*phi - omega*s`. The radial basis includes the winding of this phase at infinity. Both the dilation exponent `g` and the phase frequency `omega` are unknown, with viscosity fixed at one. The fundamental has prescribed positive RMS speed in a fixed core weight. A phase condition sets its azimuthal orientation. A real wave number defines a periodic sector of width `2*pi/kappa`.

The initial mean and fundamental are fitted to the public meridional profiles and the horizontal vorticity slice in Figures 2 and 3. The included `rotating_figure_seed.npz` records the reconstruction assumptions. The unobserved vertical wave polarization and the temporary equatorial condition are choices in this implementation. At the plane the normal velocity and mean swirl vanish, with radial slip for the mean and free slip for the rotating disturbance. The force is the fitted seed's weak momentum imbalance, multiplied by a scalar. Pseudo-arclength continuation follows equilibria through folds of that scalar. A separate trust-region solve sets the force to zero and enforces the fundamental amplitude at every trial. [Appendix C of the authors’ paper](https://github.com/tristanbuckmaster/navier-stokes/blob/main/paper/finite-time-blow-up-navier-stokes.pdf)

The following commands use the same project scratch layout and PETSc environment as the full-space solver. After each job, substitute its actual output directory for the shell variable.

```bash
driver=src/test_no_eb_3d/setonix/rotating-wave.sbatch
sbatch -A YOUR_PROJECT-gpu -p gpu-highmem "$driver" search --kappa 18.5 \
    --sigma-breaks 0 .58 1 --eta-breaks 0 .30 1 --harmonics 3 --stages 44
forced=wave-FIRST_JOB_ID-0
sbatch -A YOUR_PROJECT-gpu -p gpu-highmem "$driver" polish "$forced/closest.npz" --iterations 100
unforced=wave-SECOND_JOB_ID-0
sbatch -A YOUR_PROJECT-gpu -p gpu-highmem "$driver" polish "$unforced/trial.npz" --harmonics 6 --iterations 130
```

The search uses at least 128 radial, 128 polar and 128 phase quadrature nodes within the sector. Phase sampling resolves every quadratic product. The number of velocity harmonics and the spline degrees and knots set the independent field resolution. During the zero-force solve, each iterate receives a pressure-fitted physical momentum residual in the half-space version of the paper's H norm, including harmonics through twice the cutoff. `best-h.npz` stores the minimum of this absolute norm and `trial.npz` stores the last iterate. Both receive checks on 160³ and 192³ quadrature grids with a higher pressure degree at 192³. `polish.json` reports these measurements and whether the candidate passed. Acceptance also requires a small weak defect, the prescribed amplitude, a nonstationary phase, and a stable relative physical defect. To plot consecutive runs with Matplotlib or trace a wave that passed these checks, run

```bash
refined=wave-THIRD_JOB_ID-0
python3 src/test_no_eb_3d/plot_rotating_wave.py "$unforced/best-h.npz" \
    --history "$unforced/polish.json" --history "$refined/polish.json" --out rotating-wave.png
sbatch -A YOUR_PROJECT-gpu -p gpu-highmem "$driver" trace "$unforced/unforced.npz" \
    --coordinate kappa --values 18.75 19
```

The trace rechecks the starting wave, continues in wave number or amplitude, and refines sign changes of `g - 0.5` between accepted fields. A full-space profile would also require the counter-rotating reflected wave, the full-space symmetries and an integer wave number. The Setonix runs at 18.5 and 16.5 have yet to produce a verified unforced half-space wave.
