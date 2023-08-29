#!/bin/bash

#SBATCH --job-name=hmc
#SBATCH --partition=iaifi_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-23:00:00
#SBATCH --output=/n/home07/yitians/fermi/fermi-prob-prog/outputs/slurm/%x_%j.out
#SBATCH --error=/n/home07/yitians/fermi/fermi-prob-prog/outputs/slurm/%x_%j.err
#SBATCH --account=iaifi_lab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yitians@mit.com

source /n/home07/yitians/setup_torch.sh

cd /n/home07/yitians/fermi/fermi-prob-prog/validating

srun python run_hmc.py