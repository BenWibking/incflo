#!/bin/bash
#SBATCH --job-name=dss-holdout
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=holdout-%j.log
set -euo pipefail
if (( $# != 3 )) || [[ ! $1 =~ ^[0-9]+$ || ! $2 =~ ^[1-9][0-9]*$ || ! $3 =~ ^[1-9][0-9]*$ ]]; then
    echo 'Usage: sbatch holdout.sbatch OPTIMAS_JOBID GRID STEPS' >&2
    exit 2
fi
cd "${SLURM_SUBMIT_DIR:?}"
module load cray-python/3.12.12
export ROCM_PATH=/opt/rocm-7.0.1
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
export PYTHON="${DSS_OPTIMAS_PYTHON:-$PWD/optimas-env/bin/python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 DSS_EXPECT_HIP=1
campaign="$PWD/optimas-$1"
out="$PWD/holdout-$1-$2-$SLURM_JOB_ID-${SLURM_RESTART_COUNT:-0}"
"$PYTHON" - "$campaign/config.json" "$2" "$3" <<'PY'
import json
import sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text())
grid, steps = map(int, sys.argv[2:])
if (config.get("mode") != "richardson" or grid <= config["fine_grid"]
        or steps * config["fine_grid"] != grid * config["fine_steps"]):
    sys.exit("Choose a grid finer than this Richardson campaign and preserve its steps/grid ratio")
PY
srun --kill-on-bad-exit=1 --ntasks=1 --gpus-per-task=1 --gpu-bind=closest --cpu-bind=cores \
    "$PWD/src/test_no_eb_3d/best_dss_profile.sh" profile "$campaign" "$PWD/build/incflo.ex" \
    "$out" --grid "$2" --steps "$3"
test -s "$out/richardson-holdout.json"
