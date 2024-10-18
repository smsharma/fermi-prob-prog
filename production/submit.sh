#!/bin/bash

#SBATCH --job-name=svi_np_1nfexp_s100_kmax103_ns5k
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

#source /n/home07/yitians/setup/jax.sh
source /n/home07/yitians/setup/torch.sh

cd /n/home07/yitians/fermi/fermi-prob-prog/production

python fit.py -i $SLURM_ARRAY_TASK_ID -n 50000 --fit_type svi --n_step 5000 \
    --psf king --model np --data 1nfexp --label s100_kmax103_ns5k_1018
# python fit.py -i $SLURM_ARRAY_TASK_ID -n 10000 --fit_type hmc --n_step 0 \
#     --psf king --model gcfullAlm --data gcfull --label kmax103
# python fit.py -i $SLURM_ARRAY_TASK_ID -n 10000 --fit_type hmc --psf delta
#python fit.py -i $SLURM_ARRAY_TASK_ID -n 20000 --fit_type hmcnt --n_step 1000

# test with: python fit.py -i 0 -n 50000 --fit_type svi --n_step 100 --psf king --model np --data 1nfexp --label s100_kmax103_ns5k_1018
# test with: python fit.py -i 0 -n 100 --fit_type hmc --n_step 0 --psf king --model gcfullAlm --data gcfull --label kmax103
