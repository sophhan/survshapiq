#!/bin/bash

#SBATCH --job-name=survshapiq
#SBATCH --array=0-29
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --output=sbatch_logs/log_%A_%a.log
#SBATCH --time=00-23:00:00

set -e
hostname; pwd; date

conda activate survshapiq

date

python run_approximators_benchmark.py --seed $SLURM_ARRAY_TASK_ID

date
