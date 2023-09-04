#!/bin/bash

#SBATCH --job-name=np_p
#SBATCH --array=1
#SBATCH --partition=iaifi_gpu
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-23:00:00
#SBATCH --output=/n/home07/yitians/fermi/fermi-prob-prog/outputs/slurm/%x_%A_%a.out
#SBATCH --error=/n/home07/yitians/fermi/fermi-prob-prog/outputs/slurm/%x_%A_%a.err
#SBATCH --account=iaifi_lab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yitians@mit.com

source /n/home07/yitians/setup_torch.sh

cd /n/home07/yitians/fermi/fermi-prob-prog/validating/fit_np_to_p

srun python run.py #-i ${SLURM_ARRAY_TASK_ID}