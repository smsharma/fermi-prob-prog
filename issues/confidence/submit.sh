#!/bin/bash

#SBATCH --job-name=gen
#SBATCH --partition=iaifi_gpu_requeue
#SBATCH --array=0-5
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

cd /n/home07/yitians/fermi/fermi-prob-prog/issues/confidence

python 1_generate.py -i $SLURM_ARRAY_TASK_ID
#python 2_fit.py -i $SLURM_ARRAY_TASK_ID --mode hmc --wdir deltapsf
#python 5_toyfit.py -i $SLURM_ARRAY_TASK_ID