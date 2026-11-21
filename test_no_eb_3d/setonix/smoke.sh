#!/bin/bash
#SBATCH --partition=gpu-dev
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --job-name=dss-8-seeds
#SBATCH --output=smoke-%j.log
set -euo pipefail
module load cray-python/3.12.12
export ROCM_PATH=/opt/rocm-7.0.1
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export RUN="$PWD/smoke-$SLURM_JOB_ID" EXE="$PWD/build/incflo.ex"
export DRIVER="$PWD/src/test_no_eb_3d/optimize_dss.py"
mkdir "$RUN"
# ROCm is installed on Setonix GPU nodes. Cross-compile on this same allocation.
SLURM_CPUS_PER_TASK=24 bash "$PWD/src/test_no_eb_3d/setonix/build.sh" > "$RUN/build.log" 2>&1
"$ROCM_PATH/bin/rocminfo" > "$RUN/rocminfo.txt"
# shellcheck disable=SC2016
srun -N1 -n1 -c8 --gres=gpu:1 bash -c '
  set -eu
  export ROCR_VISIBLE_DEVICES=0
  python3 "$DRIVER" check --exe "$EXE" --out "$RUN/check" > "$RUN/check.log" 2>&1'
# shellcheck disable=SC2016
srun --kill-on-bad-exit=1 -N1 -n8 -c8 --gres=gpu:8 --cpu-bind=cores bash -c '
  set -eu
  export ROCR_VISIBLE_DEVICES="$SLURM_LOCALID"
  seed=$((19001+SLURM_LOCALID))
  python3 "$DRIVER" search --exe "$EXE" --random-seed "$seed" --iterations 3 --out "$RUN/seed-$seed" > "$RUN/seed-$seed.log" 2>&1'
python3 - "$RUN" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
runs=[json.loads(f.read_text()) for f in sorted(root.glob("seed-*/search.json"))]
assert len(runs)==len({r["initial_sha256"] for r in runs})==8
assert all(r["grids"] == [128,160,192] for r in runs)
assert all(r["iterations"]>=1 and 0 < r["final_guided_return"] < r["first_guided_return"] for r in runs)
(root/"summary.json").write_text(json.dumps(runs,indent=2)+"\n")
print([(r["random_seed"],r["first_guided_return"],r["final_guided_return"]) for r in runs])
PY
