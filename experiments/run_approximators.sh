#!/bin/bash

#SBATCH --job-name=shapiq
#SBATCH --array=0-29
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --output=sbatch_logs/log_%A_%a.log
#SBATCH --partition=short,experimental
#SBATCH --time=00-23:00:00
#SBATCH --account=mi2lab-normal
#SBATCH --exclude=dgx-2

set -e
hostname; pwd; date

module load anaconda/4.0
source $CONDA_SOURCE
conda activate survshapiq

date

python run_approximators.py --seed $SLURM_ARRAY_TASK_ID

date
