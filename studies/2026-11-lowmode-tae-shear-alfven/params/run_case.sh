#!/bin/bash
# One run of the low-mode TAE study, in this folder: simulation on 4 MPI ranks, post-processing (figures and video
# frames, see pproc_ShearAlfven.py) and the mp4 videos (needs ffmpeg). The output goes to ./<name>/.
# Usage: bash run_case.sh <name> [flags of params_ShearAlfven.py, e.g. --num-elements 32 16 8] [--max-frames N]
# Example: bash run_case.sh r32 --num-elements 32 16 8 --max-frames 150
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="$1"
shift
cd "$DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mpirun -n 4 -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS python params_ShearAlfven.py --tend 1200 --sim-folder "$NAME" "$@"
# the pproc fails (after writing all figures) if the frequency deviates by more than 10% from the expected one
python pproc_ShearAlfven.py --sim-folder "$NAME" --tend 1200 "$@" || echo "pproc: frequency check failed, see above"
for v in u_r_rz u_r_3d_cut u_r_3d; do
    frames="$NAME/results/frames/$v"
    if [ -d "$frames" ]; then
        ffmpeg -y -loglevel error -framerate 25 -i "$frames/frame_%04d.png" -pix_fmt yuv420p \
            -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" "$frames.mp4"
    fi
done
echo "Done: $NAME (figures and videos in $NAME/results/)"
