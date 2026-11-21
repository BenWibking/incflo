#!/bin/bash
#SBATCH --job-name=dss-search
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=search-%A_%a.log
set -euo pipefail

if (( $# < 2 )) || [[ ! $1 =~ ^(0|[1-9][0-9]*)$ || ! $2 =~ ^[1-9][0-9]*$ ]]; then
    echo 'Usage: sbatch --array=0-3%2 search-array.sbatch FIRST_SEED ITERATIONS [search options]' >&2
    exit 2
fi
export DSS_FIRST_SEED="$1" DSS_ITERATIONS="$2"
shift 2
cd "${SLURM_SUBMIT_DIR:?}"
export DSS_EXE="$PWD/build/incflo.ex"
export DSS_DRIVER="$PWD/src/test_no_eb_3d/optimize_dss.py"
[[ -x "$DSS_EXE" && -r "$DSS_DRIVER" ]] || {
    echo "Expected the existing incflo build and Python driver in $PWD" >&2
    exit 1
}

module load cray-python/3.12.12
export ROCM_PATH=/opt/rocm-7.0.1
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

campaign="$PWD/search-${SLURM_ARRAY_JOB_ID:?submit with --array}"
export DSS_OUTPUT="$campaign/batch-${SLURM_ARRAY_TASK_ID:?}-${SLURM_RESTART_COUNT:-0}"
mkdir -p "$campaign"
mkdir "$DSS_OUTPUT"

# Each worker resolves its own local device and assigned seed.
# shellcheck disable=SC2016
srun --kill-on-bad-exit=0 -N1 -n8 -c8 --gres=gpu:8 --cpu-bind=cores bash -c '
    set -euo pipefail
    export ROCR_VISIBLE_DEVICES="$SLURM_LOCALID"
    seed=$(( DSS_FIRST_SEED + 8 * SLURM_ARRAY_TASK_ID + SLURM_LOCALID ))
    out="$DSS_OUTPUT/seed-$seed"
    exec python3 "$DSS_DRIVER" search --exe "$DSS_EXE" \
        --random-seed "$seed" --iterations "$DSS_ITERATIONS" --out "$out" "$@" > "$out.log" 2>&1
' bash "$@"
