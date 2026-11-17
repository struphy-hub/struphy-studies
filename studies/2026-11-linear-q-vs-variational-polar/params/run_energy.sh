#!/bin/bash
# Energy runs of the study (params_linear.py), stable case, T = 100, on 4 MPI ranks; output folders beside this file
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT="${STUDY_RUNS:-$DIR}"
PY="${PYTHON:-python}"
for m in q var; do for dt in 0.5 0.125; do
    name="${m}_stable_dt${dt}"
    [ -f "$OUT/$name/run_metadata.json" ] && [ -n "$(ls "$OUT/$name/data" 2>/dev/null)" ] && { echo "skip $name"; continue; }
    mpirun -n 4 -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS $PY params_linear.py --model $m --case stable --dt $dt \
        --tend 100 --out-folder "$OUT" --sim-folder "$name" > "$OUT/$name.log" 2>&1 && echo "DONE $name" || echo "Error $name"
done; done
echo ENERGY_DONE
