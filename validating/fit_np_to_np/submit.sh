#!/bin/bash

#SBATCH --job-name=npnp_prior
#SBATCH --partition=iaifi_gpu
#SBATCH --array=0-19
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

python run.py -i $SLURM_ARRAY_TASK_ID