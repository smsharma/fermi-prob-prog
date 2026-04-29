#!/bin/bash

#SBATCH --job-name=hmc-fullprior0Alm-king
#SBATCH --array=0-99
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=0-05:00:00
#SBATCH --output=/n/home07/yitians/fermi/fermi-prob-prog/outputs/slurm/%x_%a.out
#SBATCH --error=/n/home07/yitians/fermi/fermi-prob-prog/outputs/slurm/%x_%a.err
#SBATCH --account=iaifi_lab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yitian.sun@mcgill.ca

source /n/home07/yitians/setup/torch.sh

cd /n/home07/yitians/fermi/fermi-prob-prog/production


# python fit_fermi_svi_process.py
# python fit_fermi.py --fit hmc --seed 42
python fit_calibration.py -i $SLURM_ARRAY_TASK_ID --truth fullprior-zeroAlm --fit hmc --psf king --initwtruth fullprior42-zeroAlm
# python fit_oaf.py -i $SLURM_ARRAY_TASK_ID --i_data 1
# python fit_cmp.py -i $SLURM_ARRAY_TASK_ID --fit_type svi
# python fit_pois.py -i $SLURM_ARRAY_TASK_ID --fit hmc