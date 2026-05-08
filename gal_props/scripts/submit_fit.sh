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
  # ../conf/Av_0.ini
  # ../conf/Av_1.ini
  # ../conf/Av_2.ini
  # ../conf/Av_3.ini
  # ../conf/Av_4.ini
  # ../conf/Av_5.ini
  # ../conf/Av_6.ini
  # ../conf/Av_7.ini
  # ../conf/Av_8.ini
  # ../conf/Av_9.ini
  # ../conf/Av_10.ini
  # ../conf/Av_11.ini
  # ../conf/B1_0.ini
  # ../conf/B1_1.ini
  # ../conf/B1_2.ini
  # ../conf/B1_3.ini
  # ../conf/B1_4.ini
  # ../conf/B1_5.ini
  # ../conf/B1_6.ini
  # ../conf/B1_7.ini
  # ../conf/B1_8.ini
  ../conf/B1_9.ini
  ../conf/B1_10.ini
  # ../conf/B3_0.ini
  # ../conf/B3_1.ini
  # ../conf/B3_2.ini
  # ../conf/B3_3.ini
  # ../conf/B3_4.ini
  # ../conf/B3_5.ini
  # ../conf/B3_6.ini
  # ../conf/B3_7.ini
  # ../conf/B0_0.ini
  # ../conf/B0_1.ini
  # ../conf/B0_2.ini
  # ../conf/B0_3.ini
  # ../conf/B0_4.ini
  # ../conf/B0_5.ini
  # ../conf/B0_6.ini
  # ../conf/B0_7.ini
  # ../conf/B0_8.ini
  # ../conf/B0_9.ini
  # ../conf/B0_10.ini
  # ../conf/B0_11.ini
  # ../conf/B0_12.ini
  # ../conf/B0_13.ini
  # ../conf/B0_14.ini
  # ../conf/B0_15.ini
  # ../conf/B0_16.ini
  # ../conf/B0_17.ini
  # ../conf/B0_18.ini
  # ../conf/B0_19.ini
  # ../conf/B2_0.ini
  # ../conf/B2_1.ini
  # ../conf/B2_2.ini
  # ../conf/B2_3.ini
  # ../conf/B2_4.ini
  # ../conf/B2_5.ini
  # ../conf/B2_6.ini
  # ../conf/B2_7.ini
  # ../conf/B2_8.ini
  # ../conf/B2_9.ini
  # ../conf/B2_10.ini
  # ../conf/B2_11.ini
  # ../conf/B2_12.ini
  # ../conf/B2_13.ini
  # ../conf/B2_14.ini
  # ../conf/B2_15.ini
  # ../conf/B2_16.ini
  # ../conf/B2_17.ini
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
