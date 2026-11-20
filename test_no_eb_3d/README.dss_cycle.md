# One similarity period of the modeled spiral seed

This case advances the unforced, incompressible Navier–Stokes equations on a
periodic 96³ grid covering [−96,96]³. The density and dynamic viscosity are both
one. The numerical method is incflo's method-of-lines predictor/corrector with
explicit viscosity, MAC projection for transport and nodal velocity projection.
The input is the phase-zero, cell-centered velocity of the Quokka 19-fold spiral
experiment. Its amplitudes and envelope are modeled. Reproducing the
Alpöge–Buckmaster orbit requires the authors' numerical profile.
[Authors' public materials](https://github.com/tristanbuckmaster/navier-stokes)

For a paired comparison both codes evolve in physical time. The half and full
similarity periods at s = 0.18 and 0.36 correspond to t = 1 − exp(−s). The deck
writes the initial state, the exact half-period, and the final state.

## Build and run

Use a 3D double-precision MPI build with embedded boundaries disabled. To
populate `build/deps`, run these commands from the incflo root.

```sh
git clone https://github.com/AMReX-Codes/amrex.git build/deps/amrex
git -C build/deps/amrex checkout --detach b034843fdc6a7ff6153cdbc3b5b9e1eb939f8449
git clone https://github.com/AMReX-Fluids/AMReX-Hydro.git build/deps/AMReX-Hydro
git -C build/deps/AMReX-Hydro checkout --detach 374f98a15775c3bd3ec7918921a2d5018e18bb3c
```

Configure and build from the same directory.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DINCFLO_DIM=3 \
  -DINCFLO_MPI=ON -DINCFLO_EB=OFF \
  -DAMREX_HOME="$PWD/build/deps/amrex" -DAMREX_HYDRO_HOME="$PWD/build/deps/AMReX-Hydro"
cmake --build build --target incflo -j 8
```

The upstream revisions used for this run were incflo `0539037a2549`,
AMReX `b034843fdc6a`, and AMReX-Hydro `374f98a15775`. The local incflo
change adds the optional `incflo.initial_velocity_plotfile` input. In a
fresh run directory, use the original Quokka phase-zero plotfile as follows.

```sh
mkdir -p build/dss-cycle
cp -R ../quokka/tests/plt_dss00000 build/dss-cycle/seed
cd build/dss-cycle
mpiexec -n 4 ../incflo.ex ../../test_no_eb_3d/inputs.dss_cycle > run.log 2>&1
```

The reader requires the named cell-centered fields `velocity_x`, `velocity_y` and
`velocity_z` on a single level with matching domain bounds and cell counts.
Box layouts may differ. The usual incflo initial projection is enabled.
For a standalone generator of the same modeled velocity and the instructions
for advancing and optimizing it from this checkout, see the
[one-cycle search guide](README.dss_search.md).

## Compare the runs

From the incflo root, use Python with NumPy, yt, and Matplotlib.

```sh
python3 test_no_eb_3d/compare_dss_cycle.py build/dss-cycle ../quokka/tests \
  --figure build/dss-cycle/comparison.png
```

The script samples W(s,y) = a u(t,a y), with a = exp(−s/2), by trilinear
interpolation at the same similarity-grid cell centers in both runs. Each
return error is the L2 norm of W(s) − W(0), divided by the L2 norm of that
solver's W(0), within |y| ≤ 50. The solver-to-solver error uses Quokka's W(s)
as its denominator in the same sphere. The figure uses the same centered
derivative on the physical velocity fields in both runs to compute the axial
vorticity before rescaling and interpolating to the similarity plane y₃ = 6.5.

## Results on the 96³ grid

The initial velocity components agreed in every cell, after incflo's initial
projection. Four CPU MPI ranks advanced the incflo solution to t = 0.3023236739
in 45 steps.

| Similarity time | incflo change from initial | Quokka change from initial | incflo vs. Quokka |
| --- | ---: | ---: | ---: |
| 0.18 | 39.90% | 41.01% | 5.13% |
| 0.36 | 56.37% | 57.30% | 7.48% |

Both solvers show that this modeled velocity field fails to recur on the
tested grid. The comparison command prints the errors and writes the vorticity
slices to `build/dss-cycle/comparison.png`.
