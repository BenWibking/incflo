# Upper-half-space rotating-wave search

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
