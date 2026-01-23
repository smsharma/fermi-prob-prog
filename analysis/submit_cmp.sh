#!/bin/bash

#SBATCH --job-name=hmc_cmp
#SBATCH --array=0
#SBATCH --partition=iaifi_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=0-08:00:00
#SBATCH --output=/n/home07/yitians/fermi/fermi-prob-prog/outputs/slurm/%x_%a.out
#SBATCH --error=/n/home07/yitians/fermi/fermi-prob-prog/outputs/slurm/%x_%a.err
#SBATCH --account=iaifi_lab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yitians@mit.edu

source /n/home07/yitians/setup/torch.sh

cd /n/home07/yitians/fermi/fermi-prob-prog/analysis

#===== np =====
# python fit_cmp.py --run_name svi_cmp --fit_type svi --n_step 10000 -n 50000
python fit_cmp.py --run_name hmc_cmp --fit_type hmc -n 10000