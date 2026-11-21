#!/bin/bash
#SBATCH --job-name=dss-half-wave
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:35:00
#SBATCH --output=half-wave-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module load pytorch/2.13.0-rocm7.2.4
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
driver="$PWD/src/test_no_eb_3d/solve_rotating_wave.py"
envdir=${DSS_PETSC_ENV:-"$PWD/snes-env"}
mode=${1:-search}
if [[ $# -gt 0 ]]; then shift; fi
if [[ $mode == check ]]; then
    export DSS_TEST_DEVICE=gpu
    args=(-m unittest discover -s "$PWD/src/test_no_eb_3d" -p 'test_similarity*.py' -v)
elif [[ $mode == axisymmetric ]]; then
    args=("$PWD/src/test_no_eb_3d/solve_axisymmetric.py" --out "$PWD/axisymmetric-$SLURM_JOB_ID-${SLURM_RESTART_COUNT:-0}" --device gpu "$@")
elif [[ $mode == probe ]]; then
    args=("$PWD/src/test_no_eb_3d/probe_rotating_modes.py" --out "$PWD/mode-probe-$SLURM_JOB_ID-${SLURM_RESTART_COUNT:-0}" --device gpu "$@")
elif [[ $mode == full-wave ]]; then
    args=("$PWD/src/test_no_eb_3d/solve_full_rotating_wave.py" --out "$PWD/full-wave-$SLURM_JOB_ID-${SLURM_RESTART_COUNT:-0}" --device gpu "$@")
elif [[ $mode == evolve-wave ]]; then
    args=("$PWD/src/test_no_eb_3d/evolve_full_wave.py" --out "$PWD/evolve-wave-$SLURM_JOB_ID-${SLURM_RESTART_COUNT:-0}" --device gpu "$@")
elif [[ $mode == fit-wave ]]; then
    args=("$PWD/src/test_no_eb_3d/polish_full_wave.py" --out "$PWD/fit-wave-$SLURM_JOB_ID-${SLURM_RESTART_COUNT:-0}" --device gpu "$@")
else
    args=("$driver" "$mode" --out "$PWD/wave-$SLURM_JOB_ID-${SLURM_RESTART_COUNT:-0}" --device gpu "$@")
fi
srun --kill-on-bad-exit=1 --ntasks=1 --gpus-per-task=1 --gpu-bind=closest --cpu-bind=cores \
    pytorch-exec "$envdir/bin/python" "${args[@]}"
