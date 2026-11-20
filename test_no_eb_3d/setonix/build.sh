#!/bin/bash
set -euo pipefail
module load cmake/3.31.9 ninja/1.13.0
export ROCM_PATH=/opt/rocm-7.0.1
export HIP_PATH="$ROCM_PATH"
export PATH="$ROCM_PATH/bin:$PATH"
"$ROCM_PATH/lib/llvm/bin/llvm-link" "$PWD/src/src/dss/incflo_dss_faces.ll" -o "$PWD/device-faces.bc"
cmake -S src -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DINCFLO_DIM=3 -DINCFLO_EB=OFF -DINCFLO_MPI=OFF -DINCFLO_HIP=ON \
    -DINCFLO_ENZYME_DEVICE_BC="$PWD/device-faces.bc" \
    -DAMReX_AMD_ARCH=gfx90a -DCMAKE_HIP_ARCHITECTURES=gfx90a \
    -DCMAKE_CXX_COMPILER="$ROCM_PATH/bin/hipcc" -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_FLAGS=--gcc-install-dir=/usr/lib64/gcc/x86_64-suse-linux/14 \
    -DAMREX_HOME="$PWD/deps/amrex" -DAMREX_HYDRO_HOME="$PWD/deps/AMReX-Hydro" "$@"
cmake --build build --target incflo -j "${SLURM_CPUS_PER_TASK:-12}"
