#!/bin/bash
# Is the zero coupling of the alternating wave also there in the poloidal direction (cylinder, m = N2/2 = 6) and in the
# toroidal direction of the torus (n = N3/2)? Kernel checks, then eigen-blocks, kink case / AdhocTorus, dt = 0.5.
# Results in ../runs/poltor/.  Run in params/: bash run_poltor.sh
set -e
OUT=../runs/poltor
mkdir -p $OUT
export STUDY_RUNS=$OUT OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MPI="mpirun -n 4 -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS -x STUDY_RUNS python"
# 1. coupling |b|/|u|: poloidal waves m = 1, 3, 5, 6 (cylinder, n = 0 .. 6); toroidal waves n = 1 .. N3/2 (torus)
$MPI check_nyquist_kernel.py --nz 12 --m 1 3 5 6 > $OUT/kernel_pol.log 2>&1
$MPI check_nyquist_kernel.py --geom torus --nz 8 > $OUT/kernel_tor8.log 2>&1
$MPI check_nyquist_kernel.py --geom torus --nz 12 > $OUT/kernel_tor12.log 2>&1
# 2. eigen-blocks of the cylinder: (0, 6) validates the normalization of real blocks (was ln 2 / dt);
#    (6, 0), (6, -1): alternating poloidal wave; (5, 0): the one below
for m in q var; do
  for mn in "0 6" "6 0" "6 -1" "5 0"; do
    tag=$(echo $mn | tr ' ' '_')
    $MPI eig_block.py --model $m --case kink --dt 0.5 --mn $mn --out $OUT/cyl_${m}_mn${tag}.json > $OUT/cyl_${m}_mn${tag}.log 2>&1
  done
done
# 3. eigen-blocks of the torus (16 x 12 x 8, as in the study): toroidal block n = 4 (alternating)
for m in q var; do
  $MPI eig_block.py --model $m --case torus --dt 0.5 --mn 0 4 --out $OUT/tor_${m}_n4.json > $OUT/tor_${m}_n4.log 2>&1
done
echo POLTOR_DONE
