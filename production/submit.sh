#!/bin/bash

#SBATCH --job-name=hmc_Dsim_truth_n30_Mnp
#SBATCH --array=0
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

# python fit.py -i $SLURM_ARRAY_TASK_ID --fit_type svi --n_step 5000 -n 50000 --data sim_truth_n30 --model np --label 0327
python fit.py -i $SLURM_ARRAY_TASK_ID --fit_type hmc --n_step 0    -n 10000 --data sim_truth_n30 --model np --label 0327

# test with: python fit.py -i 0 --fit_type svi --n_step 500 -n 5000 --data sim_truth_n30 --model np --label 0327
# test with: python fit.py -i 0 -n 100 --fit_type hmc --n_step 0 --psf king --model gcfullAlm --data gcfull --label kmax103
