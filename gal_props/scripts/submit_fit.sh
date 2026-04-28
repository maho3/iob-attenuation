#!/bin/bash
set -euo pipefail

# directory to store per-job wrappers
WRAPPER_DIR="./queued_scripts"
mkdir -p "$WRAPPER_DIR"

# working directory and python command
WORKDIR="/mnt/users/deaglan/symbolic_regression/iob-attenuation/gal_props"
PYCMD="python3 run_operon.py"

# change this pattern or list of ini files as needed
shopt -s nullglob
INIS=(
  ../conf/B1_2.ini
  ../conf/Av_2.ini
)
if (( ${#INIS[@]} == 0 )); then
  echo "No ini files found" >&2
  exit 1
fi
echo "Found ${#INIS[@]} ini files to process"

for INI in "${INIS[@]}"; do
  # make path absolute to avoid relative-path problems on compute node
  INI_ABS="$(readlink -f "$INI")"
  TAG="$(basename "$INI" .ini)"
  WRAPPER="$WRAPPER_DIR/run_${TAG}.sh"
  # create wrapper script
  cat > "$WRAPPER" <<EOF
#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate pyoperon

cd "$WORKDIR"
$PYCMD "$INI_ABS"

conda deactivate
EOF
  chmod +x "$WRAPPER"

  # build comment and submit - adjust resources as needed
  COMMENT="Operon ${TAG} (2hr)"
  addqueue -q berg -s -n 1x28 -m 5.0 -c "$COMMENT" -e "$WRAPPER"

  echo "Submitted $TAG using wrapper $WRAPPER"
done
