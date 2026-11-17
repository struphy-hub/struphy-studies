#!/bin/bash
# All eigenvalue blocks of the study, one after the other on 4 MPI ranks (output beside this file:
# eig_<model>_<case>_dt<dt>[_r32].json, the matrix *_A.npy and the fastest modes *_modes.npz for compare_modes.py)
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT="${STUDY_RUNS:-$DIR}"
mkdir -p "$OUT"
run() {  # model case dt [extra args] [suffix]
    local tag="eig_$1_$2_dt$3${5:-}"
    [ -f "$OUT/$tag.json" ] && { echo "skip $tag"; return; }
    mpirun -n 4 -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS ${PYTHON:-python} eig_block.py --model "$1" --case "$2" --dt "$3" $4 --modes 3 \
        --out "$OUT/$tag.json" > "$OUT/$tag.log" 2>&1
    grep -E "DONE|Error" "$OUT/$tag.log" | tail -1
}
for case in stable suydam; do for m in q var; do for dt in 0.5 0.25 0.125; do run $m $case $dt ""; done; done; done
for dt in 0.5 0.25 0.125 0.0625; do run q kink $dt ""; done
for dt in 1.0 0.5 0.25 0.125; do run var kink $dt ""; done
run var_pi kink 0.25 ""
run q kink 0.25 "--num-elements 32 12 12" _r32
run var kink 0.25 "--num-elements 32 12 12" _r32
for m in q var; do run $m suydam 0.0625 ""; run $m suydam 0.125 "--num-elements 32 12 12" _r32; done
# cost per step (8 steps after a warm-up), kink case
for spec in "q 0.0625" "var 0.125" "q 0.25" "var 0.25"; do
    set -- $spec
    [ -f "$OUT/cost_$1_dt$2.json" ] || mpirun -n 4 -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS ${PYTHON:-python} \
        profile_cost.py --model $1 --case kink --dt $2 --steps 8 --out "$OUT/cost_$1_dt$2.json"
done
# polar torus (AdhocTorus), block n = -1: about 15 min (q) and 45-65 min (var) per run; only with TORUS=1
if [ "${TORUS:-0}" = "1" ]; then
    for m in q var; do for dt in 0.5 0.25 0.125; do run $m torus $dt ""; done; done
fi
echo EIG_ALL_DONE
