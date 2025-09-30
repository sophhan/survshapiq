#!/bin/bash

#SBATCH --job-name=shapiq
#SBATCH --array=0-29
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=16G
#SBATCH --output=sbatch_logs/log_%A_%a.log
#SBATCH --partition=short
#SBATCH --time=00-12:00:00
#SBATCH --account=mi2lab-normal

set -e
hostname; pwd; date

module load anaconda/4.0
source $CONDA_SOURCE
conda activate survshapiq

date

python run_approximators.py --seed $SLURM_ARRAY_TASK_ID

date
