#!/bin/bash
# Static checks of the discrete Lie derivative curl P(B0,h x u) (seconds to minutes each, 4 MPI ranks), tables beside
# this file (or in $STUDY_RUNS); figures: python compare_slab.py, python compare_lie.py (also needs nyquist_kernel_z12_m1-3-5-6.md of
# run_poltor.sh), python splines_parity.py
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MPI="mpirun -n 4 -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS ${PYTHON:-python}"
# which velocity space loses which wave (periodic slab, B0 = e_z; FEEC check, 2 ranks)
mpirun -n 2 -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS ${PYTHON:-python} check_velocity_space.py
# condition (K): spurious bending of parallel flows (aligned slab, kink screw pinch with 12 and 13 axial elements)
$MPI check_kernel_K.py --geom slab --nz 12
for nz in 12 13; do $MPI check_kernel_K.py --geom polar --nz $nz; done
# spectrum of a homogeneous periodic slab (B0 = e_z), blocks (1, 0, n3), N3 = 12 and 13, both models (~1 min)
for model in var q; do for nz in 12 13; do $MPI eig_slab.py --model $model --nz $nz; done; done
# 1D model of the bending ratio (no MPI)
${PYTHON:-python} ratio_1d.py
echo CHECKS_DONE
