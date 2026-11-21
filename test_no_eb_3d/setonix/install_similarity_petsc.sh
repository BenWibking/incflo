#!/bin/bash
# Run inside the same pytorch-exec container used by similarity.sbatch.
set -euo pipefail
prefix=${1:?Supply a virtual environment path}
if [[ -n ${2:-} ]]; then
    export PIP_NO_INDEX=1 PIP_FIND_LINKS
    PIP_FIND_LINKS=$(realpath "$2")
fi
python3 -m venv --system-site-packages --without-pip "$prefix"
py="$prefix/bin/python"
# The ROCm image supplies torch, numpy, scipy, pip, Python headers and OpenBLAS.
"$py" -m pip install 'setuptools==80.9.0' 'wheel==0.45.1' 'packaging==25.0' 'cython==3.1.8'
export PETSC_CONFIGURE_OPTIONS="--with-mpi=0 --with-cxx=0 --with-fc=0 --with-blaslapack-lib=-lopenblas --with-make-np=${SLURM_CPUS_PER_TASK:-4}"
"$py" -m pip install --no-cache-dir --no-build-isolation --no-deps 'petsc==3.26.1'
PETSC_DIR=$("$py" -c 'import petsc; print(petsc.get_petsc_dir())')
export PETSC_DIR
"$py" -m pip install --no-cache-dir --no-build-isolation --no-deps 'petsc4py==3.26.1'
"$py" -c 'from petsc4py import PETSc; import torch; print("PETSc", PETSc.Sys.getVersion(), "ROCm", torch.version.hip)'
