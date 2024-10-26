#!/bin/bash

#SBATCH --job-name=hmc_Dgpsfix_s1k_deltapsf_Mkmaxfix_s1k_deltapsf
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

# python 1_fit.py -i $SLURM_ARRAY_TASK_ID --fit_type svi --n_step 10000 -n 50000 --data base1023_fexp --model np --comment nstest
python 1_fit.py -i $SLURM_ARRAY_TASK_ID --fit_type hmc -n 10000 --data gpsfix_s1k_deltapsf --model kmaxfix_s1k_deltapsf

# run svi: python 1_fit.py -i 0 --fit_type svi -n 50000 --n_step 10000 --data base1023_fexp --model np
# test hmc: python 1_fit.py -i 0 --fit_type test --data gpsfix_s1k_deltapsf --model kmaxfix_s1k_deltapsf