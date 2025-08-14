#!/bin/bash
source  ~/miniforge3/etc/profile.d/conda.sh
conda activate pyoperon

cd ~/symbolic_regression/iob-attenuation/symbolic_regression
python3 run_operon.py conf/iob_10.ini

conda deactivate
