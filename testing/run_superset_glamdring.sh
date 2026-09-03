#!/bin/bash
source  ~/miniforge3/etc/profile.d/conda.sh
conda activate pyoperon

cd /mnt/users/deaglan/symbolic_regression/iob-attenuation/testing
python3 superset_script.py

conda deactivate
