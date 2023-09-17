#!/bin/bash

#SBATCH --job-name=np_np
#SBATCH --partition=iaifi_gpu
#SBATCH --array=0
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-23:00:00
#SBATCH --output=/n/home07/yitians/fermi/fermi-prob-prog/outputs/slurm/%x-%a.out
#SBATCH --error=/n/home07/yitians/fermi/fermi-prob-prog/outputs/slurm/%x-%a.err
#SBATCH --account=iaifi_lab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yitians@mit.com

source /n/home07/yitians/setup_torch.sh

cd /n/home07/yitians/fermi/fermi-prob-prog/validating/fit_np_to_np

START=$SLURM_ARRAY_TASK_ID
END=$((SLURM_ARRAY_TASK_ID + 1))

python run.py --start $START --end $END