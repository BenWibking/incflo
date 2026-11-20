#!/bin/bash
#SBATCH --job-name=dss-optimas
#SBATCH --partition=gpu-dev
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=optimas-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module load cray-python/3.12.12
export ROCM_PATH=/opt/rocm-7.0.1
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export DSS_EXPECT_HIP=1
python="${DSS_OPTIMAS_PYTHON:-$PWD/optimas-env/bin/python}"
driver="$PWD/src/test_no_eb_3d/optimas_dss.py"
output="$PWD/optimas-$SLURM_JOB_ID"
if [[ ${SLURM_RESTART_COUNT:-0} != 0 ]]; then
    output="$output-retry-$SLURM_RESTART_COUNT"
fi
# Resolve the device on each MPI rank after Slurm sets its local rank.
# shellcheck disable=SC2016
srun --kill-on-bad-exit=1 -N "$SLURM_JOB_NUM_NODES" -n "$((8*SLURM_JOB_NUM_NODES))" \
    --ntasks-per-node=8 -c8 --gres=gpu:8 --cpu-bind=cores bash -c '
        export ROCR_VISIBLE_DEVICES="$SLURM_LOCALID"
        exec "$@"
    ' bash "$python" -m mpi4py "$driver" --exe "$PWD/build/incflo.ex" \
       --out "$output" "$@"
