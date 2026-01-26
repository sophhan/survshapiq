#!/bin/bash

#SBATCH --job-name=cpu
#SBATCH --account=mi2lab-hi
#SBATCH --gpus=0
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=14G
#SBATCH --partition=hopper,short,long
#SBATCH --exclude=dgx-[1-4]
#SBATCH --time=23:00:00
#SBATCH --output=logs/log_%A_%a.log  # %A is JobID, %a is ArrayID
#SBATCH --array=0-143

set -e

# Configuration
PATH_OUTPUT="/mnt/evafs/groups/mi2lab/hbaniecki/survshapiq/nki70_v2"

# Setup environment
module load anaconda/4.0
source $CONDA_SOURCE
conda activate survshapiq

echo "Task ID: $SLURM_ARRAY_TASK_ID"
hostname; date

# Execute script using the array index as the --id
python s_nki70.py --id $SLURM_ARRAY_TASK_ID --output $PATH_OUTPUT

date