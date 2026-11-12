#!/bin/bash
# Post-processing of one run of the ITPA TAE study, in this folder: figures, TAE frequency and video frames
# (pproc_ShearAlfven.py), then the mp4 videos (needs ffmpeg). The run is ./<name>/ (the default output folder of
# params_ShearAlfven.py with --sim-folder <name>); pass the same flags as to the run.
# Usage: bash pproc_case.sh <name> [flags of the run, e.g. --num-elements 24 88 8 --b-space Hdiv]
# Then: python compare.py (frequency table, spectra, (r, omega) maps, mode structure and energy of all runs).
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="$1"
shift
cd "$DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# the pproc fails (after writing all figures) if the frequency deviates by more than 10% from the expected one
python pproc_ShearAlfven.py "$DIR/$NAME" "$@" > "$NAME/pproc.out" 2>&1 ||
    echo "$NAME pproc: exit $? (frequency check failed?), see $NAME/pproc.out"
for v in u_r_rz u_r_3d_cut u_r_3d; do
    frames="$NAME/results/frames/$v"
    if [ -d "$frames" ]; then
        ffmpeg -y -loglevel error -framerate 25 -i "$frames/frame_%04d.png" -pix_fmt yuv420p \
            -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" "$frames.mp4"
    fi
done
grep -E "^TAE|rad/s" "$NAME/pproc.out" || true
echo "Done: $NAME (figures and videos in $NAME/results/)"
