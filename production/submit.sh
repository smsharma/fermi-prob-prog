#!/bin/bash

#SBATCH --job-name=hmc_kmaxfix_fexp
#SBATCH --array=0-29
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=0-08:00:00
#SBATCH --output=/n/home07/yitians/fermi/fermi-prob-prog/outputs/slurm/%x_%a.out
#SBATCH --error=/n/home07/yitians/fermi/fermi-prob-prog/outputs/slurm/%x_%a.err
#SBATCH --account=iaifi_lab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yitians@mit.com

source /n/home07/yitians/setup/torch.sh

cd /n/home07/yitians/fermi/fermi-prob-prog/production

# python 1_fit.py -i $SLURM_ARRAY_TASK_ID -n 50000 --data s1k_fexp --fit_type svi --n_step 10000 --comment kmaxfix_fexp
python 1_fit.py -i $SLURM_ARRAY_TASK_ID -n 10000 --data s1k_fexp --fit_type hmc --n_step 0 --comment kmaxfix_fexp

# python 1_fit.py -i 0 -n 100 --data s1k_fexp --fit_type hmc --n_step 0 --comment kmaxfix_fexp