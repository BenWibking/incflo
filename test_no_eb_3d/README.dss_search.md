# Searching for a discretely self-similar velocity field

The harness minimizes the one-period return error of a nonzero initial velocity field in the 3D incompressible solver. A run uses the native incflo MOL predictor and corrector, constant unit density and viscosity, periodic boundaries, a single fixed grid and fixed time steps. Each independent search uses one CPU rank or one AMD GPU die. This checkout generates a modeled 19-fold seed with a compact envelope. The authors describe a velocity tail proportional to `1/r`. Their public release omits the numerical profile coefficients. [Authors' public materials](https://github.com/tristanbuckmaster/navier-stokes)

The GPU space-time Galerkin solver searches for a periodic orbit directly in similarity coordinates and exports fields for this harness to replay. [Direct orbit solver](README.similarity_orbit.md)

The upper-half-space solver searches for one rotating wave at finite amplitude, with free similarity exponent and phase frequency. [Rotating-wave search and Setonix commands](README.rotating_wave.md)

## Build with Enzyme on a CPU

The commands below run from a working directory containing this checkout in `src/`. Use upstream LLVM Clang and a ClangEnzyme plugin compiled for the same major LLVM version. The tested generator used LLVM 23.1.6 and Enzyme 0.0.311. On a Mac, Homebrew supplies both with `brew install llvm enzyme cmake ninja`. Use Python 3.10 or later with NumPy and SciPy, for example in a virtual environment with `python3 -m pip install numpy scipy`. The optional `pack` command also uses yt.

```sh
git clone --branch enzyme-dss-search https://github.com/BenWibking/incflo.git src
git clone https://github.com/AMReX-Codes/amrex.git deps/amrex
git -C deps/amrex checkout --detach b034843fdc6a7ff6153cdbc3b5b9e1eb939f8449
git clone https://github.com/AMReX-Fluids/AMReX-Hydro.git deps/AMReX-Hydro
git -C deps/AMReX-Hydro checkout --detach 374f98a15775c3bd3ec7918921a2d5018e18bb3c

llvm_prefix=$(brew --prefix llvm)
enzyme_prefix=$(brew --prefix enzyme)
major=$("$llvm_prefix/bin/llvm-config" --version | cut -d. -f1)
cmake -S src -B cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_C_COMPILER="$llvm_prefix/bin/clang" -DCMAKE_CXX_COMPILER="$llvm_prefix/bin/clang++" \
  -DINCFLO_DIM=3 -DINCFLO_EB=OFF -DINCFLO_MPI=OFF \
  -DAMREX_HOME="$PWD/deps/amrex" -DAMREX_HYDRO_HOME="$PWD/deps/AMReX-Hydro" \
  -DINCFLO_ENZYME_PLUGIN="$enzyme_prefix/lib/ClangEnzyme-$major.dylib"
cmake --build cpu --target incflo -j 8
```

On Linux set the corresponding Clang paths and use the `.so` plugin. The gradient check takes four steps spanning the full physical period on a 16³ mesh. It perturbs a dense velocity field in three independent directions and compares the reverse-mode directional derivatives with centered differences at two step sizes.

```sh
driver=src/test_no_eb_3d/optimize_dss.py
python3 "$driver" check --exe cpu/incflo.ex --out check
python3 "$driver" search --exe cpu/incflo.ex --random-seed 19001 \
  --n 128 --steps 96 --iterations 3 --out search-19001 --plots
```

Use a fresh output directory for each run. The optimizer generates the modeled seed directly on the 128³ control grid. By default each gradient uses 128³, 160³ and 192³ with 96, 120 and 144 steps. Use `--single-grid` for a single-resolution comparison. The smallest optimization grid must have at least 128 cells per dimension. To start from a saved velocity, pass `--seed field.npy`. A Quokka velocity plotfile can be converted with `python3 "$driver" pack /path/to/plt_dss00000 field.npy`.

`evaluations.csv` records the objective, return error, gradient norm and execution time. `search.json` records the seed, initial-field hash, accepted result and termination reason. The accepted field is replayed in a fresh forward run. Its input, before incflo's initial projection, is saved in `candidate/velocity.bin` as native-endian float64 in C-order `[component,z,y,x]`. `--plots` also writes ordinary AMReX initial and final plotfiles for the accepted field.

## One GPU node on Setonix

Use the three repository checkouts and pinned revisions above on project scratch. From their parent directory, submit to your GPU allocation.

```sh
sbatch --account=YOUR_PROJECT-gpu src/test_no_eb_3d/setonix/smoke.sh
```

The job builds for `gfx90a` with ROCm 7.0.1 on its allocated GPU node, runs the gradient check on one GPU, then launches eight different seeds across the node's eight GPU dies. Each search uses one die, the three default grids and three L-BFGS iterations. The launcher checks that all eight initial-field hashes differ and that every accepted candidate reduces its return error on an independent forward replay. Per-seed histories and the aggregate `summary.json` are in `smoke-$SLURM_JOB_ID/`.

The field evolution, checkpointed stages, local reverse kernels, viscous operators and both projection solves run on the GPU. The endpoint loss and its trilinear interpolation run on the host, as does SciPy's optimizer. The LLVM IR file `src/dss/incflo_dss_faces.ll` contains the two scalar pullbacks generated by Enzyme. The ROCm frontend links these into the device kernels. This permits a GPU build on systems that ship ROCm without its LLVM development headers. To regenerate the IR after changing the reconstructed face states, run

```sh
python3 src/src/dss/export_enzyme_faces.py cpu rebuilt-faces.ll
diff -u src/src/dss/incflo_dss_faces.ll rebuilt-faces.ll
```

### Earlier 96³ Setonix benchmark

Job `53508614` built the executable and completed in 4 min 05 s on one MI250X node. The eight searches ran concurrently for 62 s. The three GPU directional derivatives on the 16³ check agreed with centered differences to a maximum relative error of `7.87e-9` at perturbation `1e-6`.

All eight 96³ searches completed three L-BFGS iterations. The following errors come from a separate forward run of each accepted candidate.

| Seed | Initial return error | Final return error |
| ---: | ---: | ---: |
| 19001 | 56.53% | 37.87% |
| 19002 | 55.82% | 38.11% |
| 19003 | 58.23% | 38.28% |
| 19004 | 55.63% | 37.89% |
| 19005 | 54.94% | 37.85% |
| 19006 | 57.94% | 39.27% |
| 19007 | 57.47% | 38.19% |
| 19008 | 57.27% | 38.30% |

Seed 19005 used four objective-and-gradient evaluations with a median of 13.50 s each. Its final core velocity norm was 1.093 times the modeled seed's norm. Replaying this field on the CPU gave a return error of `0.378467027609623`, within `1.8e-13` of the GPU result. The smoke run used the packed seed. At 96³ the standalone generator agrees with that seed to a relative velocity L2 difference of `6.6e-16` or less.

## Objective and controls

Let `S=0.36`, `t=1-exp(-S)`, and `a=exp(-S/2)`. Write `u0` for the projected initial velocity and `u1` for the independently evolved final velocity. On each grid the native objective is half the squared norm of `a*u1(a*y)-u0(y)`, divided by the squared norm of `u0`. Both norms cover cell centers in `|y| <= 50`. The reported return error is the square root of twice this objective. A full period needs no rotation. A half-period comparison of the paper's profile would require a rotation by π/19. Appendix C describes the periodic orbit and the continuation procedure used to find its approximate profile. [Alpöge and Buckmaster](https://github.com/tristanbuckmaster/navier-stokes)

The reverse pass uses the velocity at the start and predictor stage of every native step. Enzyme differentiates the periodic slope limiter, Riemann selection, and velocity flux contraction on the executed branches. Constant-density periodic MAC and nodal projections use their symmetric transpose solves, and the constant-coefficient viscous operator is also symmetric. The endpoint derivative includes the normalization and the dependence of the reference field on the initial velocity. Grid, viscosity, time step and period are held fixed. At limiter switches a classical derivative need not exist; the step-size sweep tests for a range over which the numerical directional derivative agrees.

The executable returns a gradient for every component in every cell. The example driver searches in a smaller space. It adds perturbations with cylindrical velocity components having azimuthal wave numbers 0 and 19, with bilinear coefficients in radius and vertical distance, and imposes the half turn about the horizontal axis. The default 9 × 7 basis has 567 scalar coefficients before two linear constraints. Additional multiples of 19 can be selected with `--harmonics 19 38`. Separate core overlaps with the even and odd azimuthal parts of the input seed are held fixed. This keeps the spiral nonzero and fixes its orientation relative to the seed. These amplitude constraints and the taper outside the modeled core are choices for this experiment.

A lower return error measures recurrence on this grid and box. Identifying a candidate with the paper's orbit would also require independent resolution and box-size tests, other phases, the expected far-field decay and quantitative comparison with the published velocity and vorticity diagnostics.

## Parallel searches with Optimas

From the Setonix working directory described above, reuse the HIP executable in `build/incflo.ex`. Install the optimizer on a login node. Cray Python supplies MPI, NumPy and SciPy. PyTorch fits the small response model on the CPU.

```sh
module load cray-python/3.12.12
python3 -m venv --system-site-packages optimas-env
optimas-env/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.15.0+cpu'
optimas-env/bin/python -m pip install 'optimas==0.10.1' 'ax-platform==1.3.1' 'yt==4.4.4'
job=$(sbatch --parsable -A YOUR_PROJECT-gpu src/test_no_eb_3d/setonix/optimas.sbatch)
squeue -j "$job"
```

Each MPI rank controls one AMD GPU die. Optimas proposes the amplitude of the component even under a half turn about the vertical axis in [0.85, 1.15] and the amplitude of the odd, spiral component in [0.75, 1.35]. Each trial has the same inner random seed. The initial batch uses eight Sobol proposals. Subsequent batches use the Optimas `AxSingleFidelityGenerator` with the completed three-grid scores. Every proposal is measured on all three grids. [Optimas WarpX example](https://optimas.readthedocs.io/en/latest/examples/bo_with_warpx.html)

### Three-grid Richardson optimization

Every L-BFGS evaluation varies one vector of 567 coefficients. It generates the velocity on the 128³ mesh and samples its periodic Fourier interpolant at the 160³ and 192³ cell centers. Each native incflo solve projects and evolves this common control. The default grids use 96, 120 and 144 timesteps over the same physical cycle, keeping the ratio of timestep to cell width constant. The three discrete adjoints are pulled back with the transpose of the Fourier interpolation and combined before the next L-BFGS step. A run with `--check-direction` tests the first trial's combined coefficient gradient against centered differences on all three grids.

For a normalized return error `E`, assume a leading discretization error proportional to `h^p`. The default trial order is `p=2`. Write `c, m, f` for the three resolutions and `Rcm=(Nm/Nc)^p`, `Rmf=(Nf/Nm)^p`. Two estimates of the correction from the finest grid to zero spacing are

```text
a = (Ef - Em) / (Rmf - 1)
b = (Em - Ec) / ((Rcm - 1) * Rmf)
```

When the corrections have the same sign, the search uses their harmonic mean `C=2ab/(a+b)`. At a reversal it sets `C=0`. The score is `hypot(Ef+C, a-b)`. This limiter and disagreement penalty are choices for the optimizer. They recover the usual Richardson limit for a single leading power and penalize inconsistent trends. The inner loss is half the squared score for velocity. The outer Optimas score is the largest of the guided velocity, vorticity and complex mode-19 errors. Velocity and vorticity use the sphere of radius 50. The mode uses axial vorticity at height 6.5 with radial weight `r dr`, retaining the azimuthal phase. Mode-19 final-to-initial amplitude is also recorded separately.

The report estimates the observed order using the actual unequal grid ratios. It flags consistency with the assumed order when the trend yields a positive observed order within 0.5 of that order and a nonnegative extrapolate. A reversal, a flat difference or an unresolved order is recorded explicitly. The score remains an optimization criterion. A physical convergence claim requires an independent finer grid. Because the mesh and timestep are refined together, the observed order describes their combined error. [NASA grid-convergence tutorial](https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html)

A separate one-GPU forward run can check the final winner at a resolution withheld from both optimizers. For a completed or running default campaign, submit

```sh
sbatch -A YOUR_PROJECT-gpu --dependency=afterok:"$job" src/test_no_eb_3d/setonix/holdout.sbatch "$job" 224 168
```

The holdout compares each observed error with the prediction `Ef+C*(1-(Nf/Nholdout)^p)` and with the uncorrected finest-grid value. The measurements, prediction errors and SHA-256 of the selected control are in `holdout-*/richardson-holdout.json`. The same comparison is written by the local `profile` command when its finer replay preserves the timestep-to-mesh ratio.

`optimas-$job/trial-NNNNN/coarse/search.json` contains all three velocity returns and the native adjoint weights for each evaluation. `richardson-score.json` contains the extrapolation diagnostics for each of the three observables. The native scores and endpoint plotfiles are retained separately for each grid. `summary.json` ranks the completed candidates by the guided score after each batch. The winner's `fine/` directory also contains its vorticity PNG, PDF, radial CSV and 3D vector arrays.

Use `--batches` and `--iterations` to set the search budget. To change resolution, specify `--coarse-grid`, `--middle-grid`, `--fine-grid` and the corresponding step counts, keeping steps per cell along one dimension constant. Set `--order` to change the assumed truncation order. For an independent-start campaign the standalone driver uses the same three-grid objective, with `--richardson-grids MIDDLE FINE` and `--richardson-steps MIDDLE FINE` controlling refinement. The previous two-resolution Optimas workflow is available with `--mode screen`. It optimizes on the coarse grid, replays all initial proposals on the fine grid, then promotes the best three and worst one in subsequent batches of eight.

### Measured three-grid search and independent holdout

Setonix job `53510544` completed in 25 min 38 s on one MI250X node. Eight Sobol and eight BoTorch trials each completed two L-BFGS iterations. The 48 combined objective-and-gradient evaluations used 144 native GPU adjoints. The median three-grid evaluation took 179.7 s, excluding the evaluation instrumented for the finite-difference check. At coefficient perturbation `1e-5`, the combined adjoint directional derivative was 0.08827392 and the centered difference was 0.08829033, a relative difference of 0.0186%.

Trial 10 ranked first by both the guided score (0.97821) and the measured 192³ score (0.94743). Job `53511041` replayed its accepted control on one GPU die at 224³ in 1 min 59 s. The model and control were fixed before the holdout.

| Evaluation | Steps | Velocity return error | Vorticity return error | Complex mode-19 return error |
| --- | ---: | ---: | ---: | ---: |
| 128³ | 96 | 42.59% | 90.99% | 91.08% |
| 160³ | 120 | 42.74% | 93.07% | 91.41% |
| 192³ | 144 | 42.88% | 94.74% | 92.27% |
| 224³ predicted | 168 | 42.94% | 95.56% | 92.45% |
| 224³ measured | 168 | 42.98% | 95.98% | 92.66% |

The three-grid prediction was closer to the independent 224³ measurement than the uncorrected 192³ value for each observable. Its absolute prediction errors were 0.04, 0.43 and 0.20 percentage points for velocity, vorticity and mode 19. Across all 16 trials, the assumed second-order trend passed the diagnostic for zero velocity errors, zero vorticity errors and five mode-19 errors. Trial 10 retained 41.72% of its initial mode-19 amplitude at 224³ and had a vorticity return error of 95.98%. The holdout tests one refinement of one candidate. A continuum recurrence remains unresolved.

### Measured 128³ optimization and 192³ replay

Setonix job `53509793` completed in 8 min 02 s on one MI250X node in `gpu-highmem`. All 16 searches used 128³ cells and 96 timesteps per cycle. Each took four objective-and-gradient evaluations for three accepted L-BFGS iterations. The median of the per-trial median evaluation times was 34.3 s. The eight GPU workers overlapped for 156 s in the Sobol batch and 154 s in the BoTorch batch. Twelve matched controls completed a 192³ replay with 144 timesteps. BoTorch added `1e-8` diagonal jitter during the surrogate fit and produced eight distinct proposals.

Trial 5 had the lowest measured fine score. Its velocity return error at 128³ fell from 57.16% to 40.59% during optimization. The table compares the same accepted field on both grids.

| Grid and steps | Velocity return error | Vorticity return error | Complex mode-19 return error | Mode-19 amplitude retained |
| --- | ---: | ---: | ---: | ---: |
| 128³, 96 | 40.59% | 86.77% | 97.49% | 13.77% |
| 192³, 144 | 40.86% | 90.96% | 95.90% | 27.00% |

For the eight initial proposals evaluated on both grids, the rank correlation was 0.976. Trial 5 ranked second at 128³ and first at 192³. Across all twelve paired evaluations, which include four deliberately selected from the second batch, the correlation was 0.839. The spiral weakened substantially over one cycle. These measurements establish the cost and resolution sensitivity of this short optimization. A converged recurrence would require smaller return errors and further resolution tests.

### Earlier 96³ to 128³ Optimas pilot

Setonix job `53509475` completed on one MI250X node in 3 min 16 s. Eight GPU workers overlapped in each coarse batch. The first eight proposals used Sobol sampling and the next eight used BoTorch. All 16 produced distinct accepted controls. Twelve were replayed at fine resolution and their input hashes matched the coarse candidates. BoTorch emitted numerical warnings during acquisition optimization, including a retry, and returned eight distinct model-guided proposals.

Trial 12 had the lowest fine score, 0.97313. Its fine errors were 0.38253 for velocity, 0.83660 for vorticity and 0.97313 for the complex mode 19. It retained 16.18% of the initial mode amplitude. Among the eight initial pairs, coarse/fine rank correlation was 0.83 and the fine winner ranked fourth on the coarse grid. Pooling all 12 pairs, including the four chosen by screening in the second batch, gave 0.20.

For the original fixed-seed campaign without Bayesian proposals, the following runs 32 independent searches for up to 80 iterations, using at most two nodes concurrently.

```sh
sbatch -A YOUR_PROJECT-gpu --array=0-3%2 src/test_no_eb_3d/setonix/search-array.sbatch 19001 80
```

## Download and plot the best field

The Bash helper takes either a local directory or an SSH source. In the commands below set `campaign` to a single completed or ongoing search. Richardson campaigns rank completed three-grid scores. Screened campaigns rank measured fine scores. Independent searches rank their saved guided or single-grid velocity objective.

```sh
campaign=setonix:/scratch/YOUR_PROJECT/YOUR_USER/YOUR_RUN/optimas-JOBID
profile=src/test_no_eb_3d/best_dss_profile.sh
"$profile" rank "$campaign" | head -5
"$profile" figure "$campaign" best-vorticity
"$profile" convergence "$campaign" grid-convergence
```

`figure` downloads the completed Optimas winner's PNG, PDF, radial CSV, fine score and the Richardson diagnostics when available. It needs rsync and Python's standard library. The figure shows signed axial-vorticity maps on a common color scale and the ring-averaged RMS magnitude.

`convergence` plots the leading completed trial's measured velocity, vorticity and complex mode-19 return errors on all three grids. It needs NumPy and Matplotlib. A local or SSH path to `richardson-holdout.json` can be passed as a final argument to overlay the prediction and independent measurement. It can be run while the campaign continues.

To reconstruct the profile locally from the winning endpoint plotfiles, run `"$profile" plot "$campaign" best-from-plots`. Use this for ongoing and legacy campaigns whose winning endpoints have been written. Provide NumPy, SciPy, yt and Matplotlib in your local Python environment and use `PYTHON=/path/to/python` to select it. The output directory must be new. Besides the figure and radial data, `vorticity-3d.npz` stores both vector fields with axes `[phase, component, y1, y2, y3]`.

For runs without plotfiles or for independent resolution checks, use `profile`. It downloads the accepted velocity, runs the local Enzyme-enabled executable, and makes the same products.

```sh
"$profile" profile "$campaign" cpu/incflo.ex best-replay
"$profile" profile "$campaign" cpu/incflo.ex best-refined --grid 224 --steps 168
```

Time and mesh refinement can affect the diagnostics differently. For the earlier seed 19005, independent CPU replays of the same accepted field gave the following results.

| Grid | Steps | Velocity return error | Vorticity return error | Mode-19 amplitude retained |
| ---: | ---: | ---: | ---: | ---: |
| 96³ | 48 | 37.85% | 80.14% | 4.54% |
| 96³ | 96 | 37.84% | 80.12% | 4.83% |
| 128³ | 96 | 38.06% | 83.00% | 14.00% |

The larger retained amplitude at the finer resolution coexists with a large phase-sensitive return error. Two spatial resolutions provide a sensitivity check. Establishing convergence requires further refinement and an independent fine-grid search.

## Proposed Frontier search

With a ceiling of 2,048 nodes and 50,000 node-hours, the initial unit of work would be one independent search per GPU die, or eight per node while the search fits in memory. Use bounded batches and promote distinct candidates only after an independent forward replay. The GPU smoke test covers a single die per search. Spatial decomposition and multiple shooting would need implementation and scaling tests before the later allocations. [Frontier compute nodes](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#frontier-compute-nodes)

| Stage | Node-hours | Work and decision |
| --- | ---: | --- |
| Qualification | 2,000 | Measure cost and memory at each resolution, repeat gradient checks on representative candidates, and measure how the residual changes with step size, interpolation and box size. |
| Independent starts | 16,000 | Vary the profiles, amplitudes and phases of the axisymmetric and spiral components. Add a parametrized far field with the reported decay. Periodically cluster candidates after allowing time shifts and azimuthal rotations. |
| Continuation | 14,000 | Follow promising families from simpler forced or axisymmetric solutions towards the unforced 19-fold case. Continue the period and fix a time-phase condition. Promote families with a persistent spiral and improving residual under refinement. |
| Orbit solves | 12,000 | Introduce multiple shooting in similarity time with phase and normalization conditions. Use Newton–Krylov corrections and parallel segment solves once their scaling has been measured. Track both full-period closure and the half-period rotation by π/19. |
| Independent verification | 6,000 | Repeat selected orbits with a second discretization, finer grids and larger boxes. Measure the full-field residual, divergence, vorticity and far-field behavior at multiple phases. |

The budget totals 400,000 GPU-die hours, or 24.4 wall-clock hours if all 2,048 nodes run continuously. The allocation for later stages depends on finding a branch that survives refinement. The 128³ gradient timings above provide a baseline for qualification.
