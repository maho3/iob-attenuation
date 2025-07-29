#!/bin/bash
source  ~/miniforge3/etc/profile.d/conda.sh
conda activate pyoperon

cd ~/symbolic_regression/iob-attenuation/literature_fits
python3 fit_literature.py

conda deactivate
