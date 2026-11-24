#!/bin/bash
# LinearMHD with the magnetic field in H(curl) (model "cc" of cases.py: weak induction, projected=True), against the
# q and variational models: slab spectra along and across B0 (seconds) and screw-pinch blocks (5 to 25 min each),
# 4 MPI ranks; outputs beside this file (or in $STUDY_RUNS), figure: python compare_slab.py
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MPI="mpirun -n 4 -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS ${PYTHON:-python}"
# slab: along B0 (blocks (1, 0, n3)) for N3 = 12, 13; across B0 (blocks (n1, 0, 1), N1 = 8) for all three couplings
for nz in 12 13; do $MPI eig_slab.py --model cc --nz $nz; done
for model in cc var q; do $MPI eig_slab.py --model $model --nz 12 --scan n1 --n3 1 --dt 0.005; done
mkdir -p cc_runs suydam_runs
# kink screw pinch, 16 x 12 x 12: the alternating axial block (1, 6) at two time steps, the kink (1, -1), the
# alternating poloidal blocks (6, -1), (6, 0)
$MPI eig_block.py --model cc --case kink --dt 0.5 --mn 1 6 --out cc_runs/eig_cc_kink_mn1_6.json
$MPI eig_block.py --model cc --case kink --dt 0.25 --mn 1 6 --out cc_runs/eig_cc_kink_mn1_6_dt0.25.json
$MPI eig_block.py --model cc --case kink --dt 0.5 --mn 1 -1 --out cc_runs/eig_cc_kink_mn1_-1_dt0.5.json
for mn in "6 -1" "6 0"; do set -- $mn; $MPI eig_block.py --model cc --case kink --dt 0.5 --mn $1 $2 --out cc_runs/eig_cc_kink_mn$1_$2.json; done
# Suydam screw pinch, dt = 0.25: physical (2, -1) (16 and 32 radial elements), alternating poloidal blocks at N2 = 12,
# the resolved m = 6 block and the new alternating block at N2 = 24 (with the q model for comparison)
for mn in "2 -1" "6 0" "6 -1" "6 -3"; do set -- $mn; $MPI eig_block.py --model cc --case suydam --dt 0.25 --mn $1 $2 --out suydam_runs/eig_cc_suydam_mn$1_$2.json; done
$MPI eig_block.py --model q --case suydam --dt 0.25 --mn 6 -3 --out suydam_runs/eig_q_suydam_mn6_-3.json
$MPI eig_block.py --model cc --case suydam --dt 0.25 --mn 2 -1 --num-elements 32 12 12 --out suydam_runs/eig_cc_suydam_mn2_-1_r32.json
for spec in "q 6 -3" "cc 6 -3" "cc 12 -6"; do set -- $spec; $MPI eig_block.py --model $1 --case suydam --dt 0.25 --mn $2 $3 --num-elements 16 24 12 --out suydam_runs/eig_$1_suydam_mn$2_$3_n24.json; done
echo CC_DONE
