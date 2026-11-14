#!/bin/bash
# Run and post-process one case of the study, in the folder of this script (next to params_/pproc_tokamak_equilibrium.py):
#   bash run_case.sh <name> [flags of params_tokamak_equilibrium.py]
# e.g. bash run_case.sh shift_b0 --shift B0
# Environment: NP (MPI ranks, default 4), PYTHON (default python).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
NAME=$1
shift
NP=${NP:-4}
PYTHON=${PYTHON:-python}
# all runs of the study save (and bin) every 5 steps (101 saved times up to t=100)
FLAGS=(--save-step 5 "$@")
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mpirun -n "$NP" -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS "$PYTHON" params_tokamak_equilibrium.py --sim-folder "$NAME" "${FLAGS[@]}"
"$PYTHON" pproc_tokamak_equilibrium.py "$NAME" "${FLAGS[@]}" --video
