#!/bin/bash

#SBATCH --job-name=fermi-svi
#SBATCH --array=0
#SBATCH --partition=iaifi_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=0-02:00:00
#SBATCH --output=/n/home07/yitians/fermi/fermi-prob-prog/outputs/slurm/%x_%a.out
#SBATCH --error=/n/home07/yitians/fermi/fermi-prob-prog/outputs/slurm/%x_%a.err
#SBATCH --account=iaifi_lab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yitian.sun@mcgill.ca

source /n/home07/yitians/setup/torch.sh

cd /n/home07/yitians/fermi/fermi-prob-prog/analysis


python fit_fermi.py --fit svi --seed 42
# python fit_calibration.py -i $SLURM_ARRAY_TASK_ID --truth old --fit hmc --psf delta
# python fit_oaf.py -i $SLURM_ARRAY_TASK_ID --i_data 2
# python fit_cmp.py --run_name svi_cmp --model model --data cmp --fit_type svi --n_step 10000 -n 50000
# python fit_cmp.py --run_name hmc_cmp --fit_type hmc --model model --data cmp -n 10000