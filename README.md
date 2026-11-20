# Directory overview

| Directory     | Description                                         |
| --------------| --------------------------------------------------- |
| test_2d       | Directory for building 2D EB executable             |
| test_3d       | Directory for building 3D EB executable             |
| test_no_eb_2d | Directory for building 2D non-EB executable         |
| test_no_eb_3d | Directory for building 3D non-EB executable         |
| src           | Source files                                        |


# Using incflo

To build and run incflo, you will need to first clone the AMReX and AMReX-Hydro repositories.

We suggest using the _development_ branch of both repositories.

## Check out AMReX 

Clone AMReX from the official Git repository
```shell
> git clone git@github.com:AMReX-Codes/amrex.git
```

Clone AMReX-Hydro from the official Git repository
```shell
> git clone git@github.com:AMReX-Fluids/AMReX-Hydro.git
```

## Build and run an example incflo problem
Clone and build incflo
```shell
> git clone git@github.com:AMReX-Fluids/incflo.git
> cd incflo/test_3d
> make -j4
> mpirun -np 4 incflo3d.gnu.MPI.EB.ex benchmark.channel_cylinder-x
```

Note that, depending on where you have put the amrex and AMReX-Hydro repos
in your local working space, you may need to set AMREX_HOME and AMREX_HYDRO_HOME
either as environment variables or in incflo/test*/GNUmakefile.

## Search for a self-similar flow

The [one-cycle search guide](test_no_eb_3d/README.dss_search.md) covers the Enzyme
adjoint, the L-BFGS driver, and an eight-seed smoke test on one Setonix GPU node.

# Contributing

We welcome contributions in the form of pull-requests from anyone. For more
details on how to contribute to incflo, please see
[CONTRIBUTING.md](CONTRIBUTING.md).
