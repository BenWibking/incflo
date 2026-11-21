# Searching for a periodic orbit in similarity coordinates

`solve_similarity_orbit.py` solves for a three-dimensional velocity field over a full similarity period and for the period itself. It uses a space-time Galerkin discretization on all of space. PyTorch evaluates the field, nonlinear terms and linearized operator in double precision on one GPU. Arclength continuation defaults to SciPy Newton–GMRES. Implicit pseudo-time marching defaults to PETSc SNES. The PETSc backend supplies line-search and trust-region Newton methods. The existing incflo executable can replay the resulting initial field independently in physical time.

## Run on Setonix

From the project scratch directory containing the fork in `src/` and the Enzyme-enabled GPU executable in `build/incflo.ex`, submit

```bash
sbatch -A YOUR_PROJECT-gpu src/test_no_eb_3d/setonix/similarity.sbatch \
    --replay build/incflo.ex --step 0.15 --max-stages 20 --krylov 72
```

The job uses one GPU die and the Pawsey ROCm PyTorch module. It searches on a 128-point radial and polar quadrature, audits the result at 128, 160 and 192, and tests additional harmonics at 192. The optional first argument `--replay PATH` runs a separate 128³ incflo simulation at the fitted period. Set `DSS_REPLAY_GRID` and `DSS_REPLAY_STEPS` in the job environment to change the replay sampling. The launcher writes `similarity-JOBID-0/` under the submission directory.

With Python 3.10 or newer, NumPy, SciPy and a GPU build of PyTorch, the equivalent standalone commands are

```bash
driver=test_no_eb_3d/solve_similarity_orbit.py
python3 "$driver" search --device gpu --out orbit --step 0.15 --max-stages 20
python3 "$driver" audit orbit/orbit.npz --device gpu --grids 128 160 192 --out orbit/audit.json
```

`search.json` records the forcing, period, phase error, spiral amplitude, Krylov iterations and unforced equation defect. Each accepted continuation point has a restart file in `stages/`. `orbit.npz` holds the latest field, model parameters, original forcing reference and continuation history needed to resume. Search and export commands take a new output directory.

```bash
# Resume the same branch, including its direction through folds.
python3 "$driver" search --restart orbit/orbit.npz --device gpu --out longer --max-stages 40

# Keep the field, introduce new harmonics, and solve the unforced equations immediately.
python3 "$driver" search --lift orbit/stages/stage-0001.npz --direct --device gpu \
    --grid 160 --radial 9 --degrees 9 --azimuthal 4 --temporal 4 \
    --newton 14 --krylov 80 --out finer
```

A lift embeds the original radial and angular coefficients exactly and carries over the fitted period. A lift without `--direct` starts a new continuation whose forcing is the residual of that field in the larger space. For distinct modeled starts, vary `--amplitude` and `--wave` or set a positive `--jitter` and vary `--seed`.

## PETSc SNES

Use a real double-precision PETSc and petsc4py in the same Python environment as PyTorch. On Setonix, run the following once inside a CPU allocation. The build uses a single-rank PETSc and the ROCm container's OpenBLAS. An optional second argument to the installer specifies a wheelhouse with the pinned source archives and build dependencies for an offline build.

```bash
module load pytorch/2.13.0-rocm7.2.4
pytorch-exec /bin/bash src/test_no_eb_3d/setonix/install_similarity_petsc.sh "$PWD/snes-env"
```

The batch launcher discovers `snes-env` in its submission directory. Submit one search per GPU die, for example

```bash
sbatch -A YOUR_PROJECT-gpu src/test_no_eb_3d/setonix/similarity.sbatch \
    --replay build/incflo.ex --solver petsc --snes-type newtonls --pc bordered \
    --direct --grid 160 --radial 9 --degrees 9 --azimuthal 4 --temporal 4 \
    --seed 19002 --jitter 0.05 --newton 200 --krylov 80
```

`--snes-type newtontr` selects trust-region Newton. Both methods use the same augmented residual, Jacobian products and checkpoint format, so `--restart` works across backends. The coefficient vectors live on one host rank. Field evaluations, linearized advection and the transpose action run on the selected device. The transpose differentiates the linear field map in reverse mode when the nonlinear method requests it.

The residual always carries the reference radial Stokes–dilation inverse. PETSc's `--pc bordered` updates each radial inverse to the current period and solves a three-variable Schur complement for the period, forcing and inverse-square deflation derivative. It includes both scalar constraints. Velocity advection remains in the exact Jacobian action. A singular small border falls back to radial scaling for that linearization. `--pc none` gives a baseline with the same residual and the same right-GMRES stopping norm.

Standard PETSc options use the prefix `orbit_`. For example, set `PETSC_OPTIONS='-orbit_ksp_rtol 0.01 -orbit_ksp_monitor_true_residual'` in the run environment to change and monitor the inner solve. The default inner relative tolerance is 0.03. `--newton` and `--krylov` set the default outer and inner iteration limits. Each SNES iteration prints the nonlinear residual and cumulative Krylov count. In `search.json`, `petsc_correctors` records each attempt's SNES and KSP termination reasons, residual history and border fallback count. The search also checks the actual augmented residual against `--tol` before accepting a corrector. [PETSc SNES manual](https://petsc.org/release/manual/snes/)

## Implicit pseudo-transient continuation

`search --pseudo-transient` evolves the entire space-time coefficient vector in an artificial time $\tau$. The Fourier phase $s/S$ still spans one similarity period at every $\tau$. Write $x$ for the normalized field coefficients, $S$ for the free period, and $\lambda$ for the scalar homotopy strength. The clock starts at $\lambda(0)=1$ and obeys $T\dot\lambda=-\lambda$, so the zero-forcing limit is at $\tau\to\infty$. The backward Euler equations for a step of length $h$ are

$$
\lambda_{n+1}=\frac{\lambda_n}{1+h/T},\qquad
\frac{x_{n+1}-x_n}{h}+\widehat R(x_{n+1},S_{n+1};\lambda_{n+1})=0,\qquad
p^\mathsf{T}(x_{n+1}-x_*)=0.
$$

Here $\widehat R$ is the forced weak similarity residual after the reference radial-block scaling. The artificial mass is the identity on those normalized coefficients. The phase condition is algebraic and determines the period. The driver solves the scalar backward Euler row analytically, evaluates all family parameters at the new strength, and solves the implicit field step with SNES. Its matrix-free Jacobian and radial preconditioner both include the $I/h$ field shift. The pseudo-time search uses the ordinary weak residual. Inverse-square deflation is specific to the direct Newton search.

The modeled path uses the equation family in Appendix C.3 and the fixed 19-fold, full-space Galerkin basis. It varies $g=1/2+(g_0-1/2)\lambda$ and $\nu=1+(\nu_0-1)\lambda$, with damping $\gamma_0\lambda$ on non-axisymmetric velocity modes. Defaults are $T=1$, $g_0=0.502$, $\nu_0=1$, and $\gamma_0=0.1$. In weak coefficient space the forcing is $\lambda f_{\rm ax}+\lambda^2 f_{\rm w}$. The residual of the steady axisymmetric part of the seed defines $f_{\rm ax}$. The additional force $f_{\rm w}$ balances the finite-amplitude wave seed at $\lambda=1$ and decays faster. This gives a reproducible forced starting point. The authors also followed half-space rotating waves with a continuously variable azimuthal wave number and identified a second family that reached $g=1/2$. Their intermediate states and numerical forcing are unavailable in the public materials. [Paper, Appendix C.3](https://github.com/tristanbuckmaster/navier-stokes/blob/main/paper/finite-time-blow-up-navier-stokes.pdf)

With the PETSc environment installed, the one-GPU Setonix command is

```bash
sbatch -A YOUR_PROJECT-gpu src/test_no_eb_3d/setonix/similarity.sbatch \
    --replay build/incflo.ex --pseudo-transient \
    --grid 160 --radial 9 --degrees 9 --azimuthal 4 --temporal 4 \
    --seed 19002 --jitter 0.05 --step 0.1 --newton 10 --krylov 80 --max-stages 40
```

`--step` sets the initial pseudo-time step. A successful step using at most half the Newton budget (rounded up, minimum two iterations) grows the next step by 1.5, up to `--ptc-max-step`. A failed implicit solve halves the step and retries from the last accepted field and clock. `--restart` restores the adaptive step, accumulated pseudo-time, log forcing, seed and family parameters. `--lift` starts a new clock from an existing field.

`stages[].implicit_residual` measures the backward Euler equation. `projected_residual` measures the instantaneous forced equation with the time derivative removed. `unforced_relative_defect` always evaluates the physical target at $g=1/2,\nu=1,F=0$ and divides its scaled residual by the current field norm. The march records the first crossing below `--min-spiral` and continues. A nonzero projected root requires sufficiently small forcing, absolute target residual and phase error, relative target defect, and sufficient field and spiral amplitude. A small absolute residual and a field or spiral below the amplitude floor at the clock tolerance are labeled `limit_below_amplitude_floor`. The independent audit and incflo replay also use the physical target equations.

## Equation and field representation

With unit viscosity, similarity position $y$, similarity time $s$, and modified pressure $H$, the solver imposes

$$
\partial_s W + \tfrac12(W+y\cdot\nabla W)
+ (\nabla\times W)\times W + \nabla\times\nabla\times W + \nabla H=0,
\qquad \nabla\cdot W=0,\qquad W(s+S)=W(s).
$$

The field is a sum of poloidal and toroidal vector spherical harmonics. These enforce incompressibility analytically. The test fields are also divergence-free, so the pressure drops out of the weak equations. Viscosity contributes the inner product of the trial and test curls. The radial mapping covers zero to infinity. Trial velocities admit a $1/r$ tail and test velocities decay as $1/r^3$.

For azimuthal order $m=19j$, the temporal harmonic $n$ has the same parity as $j$. This enforces a rotation by π/19 in half a period. Real spherical harmonics also impose the half turn about the horizontal axis. The radial phase is

$$
\chi = \frac{2\pi}{S}\left[s-\log\!\left(1+(r/r_c)^2\right)\right].
$$

Spatial derivatives are taken at fixed $s$ and include derivatives of this phase. Its logarithmic winding allows the time-dependent far field to follow the similarity dilation. The radial functions use Jacobi polynomials. The default radial map scale of 10 and winding radius of 60 are numerical choices. Appendix C of Alpöge and Buckmaster describes piecewise polynomial elements and a continuation from a forced axisymmetric state through rotating waves. The public materials supply the equations and symmetries. The starting coefficients and forcing here are modeled. [Paper, Appendix C.1–C.3](https://github.com/tristanbuckmaster/navier-stokes/blob/main/paper/finite-time-blow-up-navier-stokes.pdf)

The default expansion has eight radial coefficients and eight polar degrees per allowed azimuthal order, $j\le3$ and $n\le3$, for **1,792 real field coefficients**. The period is a further unknown. At `--grid 128`, the integrals use 128 radial nodes, 128 polar nodes and ten azimuthal samples per 19-fold sector, equivalent to 190 over the full circle. Ten time samples integrate products through the retained temporal bandwidth. The azimuthal and time rules use more than three times the highest retained index so quadratic products can be projected without aliasing. Increasing `--grid` refines quadrature. Increasing `--radial`, `--degrees`, `--azimuthal` or `--temporal` gives the field more degrees of freedom.

### Recovering physical space and time

For collapse at the origin at time $T$, use $s=\log[T/(T-t)]$ and $y=x/\sqrt{T-t}$. A periodic similarity velocity $W$ gives

$$
u(x,t)=(T-t)^{-1/2}W(s,y),\qquad
\omega(x,t)=(T-t)^{-1}(\nabla_y\times W)(s,y),\qquad t<T.
$$

One full period gives every pre-collapse time by setting the phase fraction in `sample_cartesian` to $(s/S)\bmod 1$. The checkpoint contains the spatial and temporal mode coefficients and the fitted period. The evaluator includes the radius-dependent phase winding. These formulas reconstruct the exactly discretely self-similar flow when the profile solves the unforced equations. The paper's compactly supported initial data lead to a separate finite-energy flow that approaches the profile in similarity coordinates near blow-up. [Paper, equations (1.2)–(1.6) and Theorem 1.1](https://github.com/tristanbuckmaster/navier-stokes/blob/main/paper/finite-time-blow-up-navier-stokes.pdf)

## Continuation and interpretation

Write $R(c,S)$ for the weak similarity-equation residual. The initial coefficients $c_*$ specify an axisymmetric background with time-dependent, oppositely rotating 19-fold waves. The solver follows

$$
R(c,S)-\lambda R(c_*,S_*)=0
$$

from λ = 1 toward λ = 0, starting at $S_*=0.36$ by default. The initial period is adjustable. A condition on the temporal phase fixes time-translation freedom. Pseudo-arclength continuation allows the forcing parameter to turn at a fold. The Jacobian action differentiates the quadratic advection exactly in the velocity coefficients and centers a difference in log period. Radial blocks of the linear Stokes–dilation operator at the reference period precondition the residual.

`--direct` starts Newton at zero forcing. Zero-forcing corrections use inverse-square deflation in the time-dependent mode-19 amplitude. A minimum retained spiral and field norm guards against convergence to the trivial branch. `--min-spiral` sets this floor relative to the reference coefficients. A direct attempt that exhausts its Newton budget saves the latest iterate with status `direct_unconverged`. A nonzero field satisfying the retained, unforced equations to the requested tolerance is labeled `unforced_projected_root`.

The value `unforced_relative_defect` is a preconditioned coefficient norm divided by the current coefficient norm. `projected_residual` includes the current artificial forcing. A tiny forced residual means the Newton corrector solved that continuation step. The independent `audit` also measures `normalized_weak_defect`, the spatial $L^2$ norm of the unforced residual projected on the solenoidal test fields, multiplied by $S$ and divided by the velocity norm with radial weight `(1+(r/50)^2)^-2`. This measure is independent of the Newton preconditioner. The final audit adds one radial degree, one polar degree per order, one azimuthal order and one temporal frequency while preserving the candidate exactly. Compare quadrature changes and this enriched defect separately.

## Independent physical-time replay

```bash
exe="$PWD/build/enzyme/incflo.ex"
python3 "$driver" export finer/orbit.npz --phase 0 --grid 128 --steps 144 --out replay
(cd replay && "$exe" inputs)
cat replay/dss_result.txt
```

The export samples the similarity field onto native Cartesian cell centers and writes a float64 `velocity.bin` and an incflo input file. The physical run advances a full cycle to `1-exp(-S)`. The native output gives half the squared relative velocity return error, the relative error, the final physical time and the step count. This is an unforced run even when the exported continuation point has nonzero forcing. `--phase` is a fraction of the similarity period. `--half-box` changes the periodic replay box. The Galerkin solution extends to infinity, so box size, mesh and timestep all matter when interpreting the replay.

## Numerical checks

Run the independent Cartesian derivative, symmetry, far-field, weak-equation, Jacobian, Newton and replay-format checks with

```bash
uv run --no-project --with numpy --with scipy --with torch python test_no_eb_3d/test_similarity_orbit.py -v
```

Pass `--gpu -v` to the test script inside a GPU allocation to run the same checks against the accelerated operator. With petsc4py installed, the suite also recovers the known forced solution with both SNES methods, exercises restart and deflation, and checks the actual residual after an early PETSc stopping reason.

### Setonix results

Job `53513055` used one MI250X GPU die, the default 1,792 coefficients, a maximum arclength step of 0.15 and 72 GMRES directions. Eight continuation steps reduced the forcing from 1 to 0.111971 at a period of 0.385601. The next correction reduced the spiral to 1.17% of its reference coefficient norm and triggered the 5% rejection threshold. The checkpoint retained the preceding field, with 12.07% of the initial spiral norm. Its forced Newton residual was 2.18e-8. Its unforced weak defect was 3.36093 at all three quadratures and 3.36120 with 3,726 test coefficients. An independent 128³ incflo run over 144 steps measured a relative velocity return error of 44.41%.

Job `53513447` lifted the fourth continuation point into 3,726 coefficients and ran 14 deflated Newton steps at exactly zero forcing on the 160-point mesh. The period moved from 0.375987 to 0.395549. The normalized weak defect was 4.19573 on all three quadratures and 4.19592 with 6,600 test coefficients. The separate 128³ incflo replay over 144 steps measured a **55.21% velocity return error**. The search recorded `direct_unconverged`. All eight implementation tests passed on both CPU and MI250X, including the Cartesian weak-equation check and recovery of the period of a manufactured forced solution.

Job `53517682` ran the SciPy direct search from seed 19002 with jitter 0.05, 3,726 coefficients and 160 radial and polar nodes. It used all 200 Newton iterations and 15,233 GMRES iterations. GMRES reached the limit of 80 on 184 solves. The final preconditioned unforced relative defect was 0.07915 at period 2.03901. The 128³ physical-time return error was 100.13%.

Job `53520094` compared PETSc 3.26.1 line-search Newton on the same MI250X die, using right GMRES with relative tolerance 0.03 and a limit of 80. Each pair started from identical coefficients, period and phase reference. Both preconditioner choices retained reference-period residual scaling. The numbers in each cell are GMRES iterations for successive Newton solves.

| Starting field | No additional preconditioner | Bordered radial preconditioner |
| --- | --- | --- |
| Fresh seed 19002 | 55, 31, 51 | 33, 35, 75 |
| Checkpoint after 200 SciPy iterations | 80, linear solve failed | 6, 6, 8 |

The bordered restart reduced the augmented residual from 0.120827 to 0.120735 in three Newton steps. Its independent 128³ incflo replay over 144 steps gave a relative velocity return error of 100.12%. The checkpoint is an unconverged candidate. All eleven similarity tests passed on the MI250X and on CPU.

Job `53521653` ran the pseudo-time command above on one MI250X die with 3,726 field coefficients, 160 radial and polar nodes, 247 effective azimuthal nodes, and 13 phase samples. It accepted 15 backward Euler steps with zero rejections and 123 total GMRES iterations. The largest implicit step residual was 7.50e-8. At pseudo-time 87.38 the forcing was 3.23e-9, the field coefficient norm was 2.32e-8 of its starting value, and the spiral norm was 4.11e-9 of its starting value. The absolute unforced projected residual was 2.32e-8 and the relative unforced defect was 1.00000075. The run reported `limit_below_amplitude_floor`. A 128³ incflo replay took 144 physical steps and measured 10.99% relative velocity return error on this nearly zero field. All twelve similarity tests passed on CPU and on the MI250X.
