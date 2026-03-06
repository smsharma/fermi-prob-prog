#!/bin/bash

#SBATCH --job-name=calib-svi-old-king
#SBATCH --array=0-29
#SBATCH --partition=iaifi_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=0-03:00:00
#SBATCH --output=/n/home07/yitians/fermi/fermi-prob-prog/outputs/slurm/%x_%a.out
#SBATCH --error=/n/home07/yitians/fermi/fermi-prob-prog/outputs/slurm/%x_%a.err
#SBATCH --account=iaifi_lab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yitian.sun@mcgill.ca

source /n/home07/yitians/setup/torch.sh

cd /n/home07/yitians/fermi/fermi-prob-prog/analysis

python fit_calibration.py -i $SLURM_ARRAY_TASK_ID --fit svi --truth old --psf king

#===== np =====
# python fit.py -i $SLURM_ARRAY_TASK_ID --fit_type svi --n_step 10000 -n 50000 \
#    --data fermi --model base --n_exp 7 --seed $((SLURM_ARRAY_TASK_ID * 4224)) --lr 3e-4 --n_par 16 \
#    --guide iaf --num_flows 5 --hidden_dim_n 128 --renyi_alpha 1 --comment ""

# python fit.py -i $SLURM_ARRAY_TASK_ID --fit_type hmc --n_step 0     -n 25000 \
#     --data fermi --model base --n_exp 7 --seed $((SLURM_ARRAY_TASK_ID * 4224)) --comment ""

#===== pois =====
# python fit.py -i $SLURM_ARRAY_TASK_ID --fit_type svi --n_step 10000 -n 50000 \
#     --data pois23new --model pois --n_exp 7 --seed 424242 --lr 3e-4 --n_par 16 \
#     --guide iaf --num_flows 5 --hidden_dim_n 128 --renyi_alpha 1 --comment ""

# python fit.py -i $SLURM_ARRAY_TASK_ID --fit_type hmc --n_step 0     -n 10000 \
#     --data pois23new --model pois --n_exp 7 --seed 4224 --comment ""
