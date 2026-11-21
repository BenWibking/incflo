#!/bin/bash
#SBATCH --job-name=dss-orbit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=similarity-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module load pytorch/2.13.0-rocm7.2.4
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
replay_exe=""
if [[ ${1:-} == --replay ]]; then
    [[ $# -ge 2 && -x $2 ]] || { echo 'Supply an executable incflo build after --replay' >&2; exit 2; }
    replay_exe=$(realpath "$2")
    shift 2
fi
driver="$PWD/src/test_no_eb_3d/solve_similarity_orbit.py"
out="$PWD/similarity-$SLURM_JOB_ID-${SLURM_RESTART_COUNT:-0}"
py=(python3)
envdir=${DSS_PETSC_ENV:-"$PWD/snes-env"}
if [[ -x $envdir/bin/python ]]; then
    py=(pytorch-exec "$envdir/bin/python")
fi
run=(srun --kill-on-bad-exit=1 --ntasks=1 --gpus-per-task=1 --gpu-bind=closest --cpu-bind=cores "${py[@]}")
"${run[@]}" "$driver" search --out "$out" --device gpu "$@"
"${run[@]}" "$driver" audit "$out/orbit.npz" --device gpu --grids 128 160 192 --out "$out/audit.json"
if [[ -n $replay_exe ]]; then
    "${run[@]}" "$driver" export "$out/orbit.npz" --out "$out/replay" \
        --grid "${DSS_REPLAY_GRID:-128}" --steps "${DSS_REPLAY_STEPS:-144}"
    (
        cd "$out/replay"
        export ROCM_PATH=/opt/rocm-7.0.1
        export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
        srun --kill-on-bad-exit=1 --ntasks=1 --gpus-per-task=1 --gpu-bind=closest --cpu-bind=cores \
            "$replay_exe" inputs > incflo.log
        grep -q 'HIP initialized with 1 device' incflo.log
        python3 - <<'PY'
import json
import math
from pathlib import Path
from numpy import loadtxt
config = json.loads(Path("replay.json").read_text())
loss, error, end, steps = loadtxt("dss_result.txt")
if not (steps == config["steps"] and abs(end-config["end_time"]) < 1e-12
        and math.isfinite(error) and error >= 0 and math.isclose(2*loss, error**2, rel_tol=1e-10)):
    raise SystemExit("Incomplete incflo replay")
print(f"Independent {config['grid']}^3 physical-time velocity return error = {error:.8g}")
PY
    )
fi
