#!/bin/bash
#SBATCH --job-name=iob_0
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128
#SBATCH --time=1:10:00
#SBATCH --partition=pscomp
#SBATCH --output=/data101/bartlett/symbolic_regression/iob-attenuation/out_files/iob_attenuation_fit_%j.out
#SBATCH --error=/data101/bartlett/symbolic_regression/iob-attenuation/out_files/iob_attenuation_fit_%j.err
#SBATCH --mail-user=deaglan.bartlett@iap.fr
#SBATCH --mail-type=END,FAIL  

# Modules
module purge

# Environment
source /home/bartlett/.bashrc
source /home/bartlett/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate pyoperon

# Kill job if there are any errors
set -e

cd
cd symbolic_regression/iob-attenuation/symbolic_regression
python3 run_operon.py conf/iob_0.ini

conda deactivate

exit 0
