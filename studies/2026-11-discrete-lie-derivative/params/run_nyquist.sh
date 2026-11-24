#!/bin/bash
# The static grid-scale mode of the shortest axial wave (finding 9), one run after the other on 4 MPI ranks, output
# beside this file (or in $STUDY_RUNS); figures: python compare_nyquist.py, python compare_timerun.py
#  1. kink time runs at dt = 0.5, T = 400 (q: about 8 min, variational: about 19 min)
#  2. eigenvalues of the block (1, N3 // 2) for N3 = 8, 11, 12, 13, 16, 24 (16 radial) and N3 = 12 (32 radial), with the
#     mode in the plane z = 0 and, for N3 = 12, half an element further (from the saved matrix)
#  3. the projections of B0 x u for smooth waves in each velocity component, N3 = 12 and 11
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT="${STUDY_RUNS:-$DIR}"
MPI="mpirun -n 4 -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS ${PYTHON:-python}"
for m in q var; do
    [ -d "$OUT/${m}_kink_dt0.5" ] || $MPI params_linear.py --model $m --case kink --dt 0.5 --tend 400 --save-velocity \
        --out-folder "$OUT" --sim-folder ${m}_kink_dt0.5 > "$OUT/${m}_kink_dt0.5.log" 2>&1
done
for m in q var; do
    for spec in "16 8" "16 11" "16 12" "16 13" "16 16" "16 24" "32 12"; do
        set -- $spec
        t=nyq_${m}_r$1_z$2
        [ -f "$OUT/$t.json" ] || $MPI eig_block.py --model $m --case kink --dt 0.5 --mn 1 $(( $2 / 2 )) \
            --num-elements $1 12 $2 --modes 2 --out "$OUT/$t.json" > "$OUT/$t.log" 2>&1
    done
    [ -f "$OUT/nyq_${m}_r16_z12_mid.json" ] || $MPI eig_block.py --model $m --case kink --dt 0.5 --mn 1 6 \
        --num-elements 16 12 12 --modes 2 --plane-eta3 0.041666666666666664 --load-A "$OUT/nyq_${m}_r16_z12_A.npy" \
        --out "$OUT/nyq_${m}_r16_z12_mid.json" > "$OUT/nyq_${m}_r16_z12_mid.log" 2>&1
done
for nz in 12 11; do $MPI check_nyquist_kernel.py --nz $nz; done
for g in hollow slab; do $MPI check_nyquist_kernel.py --nz 12 --geom $g; done
echo NYQUIST_DONE
