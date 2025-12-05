#!/bin/bash

#SBATCH --job-name=svi_D23new_p6v11_Mbase23fixO
#SBATCH --array=3-9
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

cd /n/home07/yitians/fermi/fermi-prob-prog/analysis

#===== np =====
python fit.py -i $SLURM_ARRAY_TASK_ID --fit_type svi --n_step 10000 -n 50000 \
    --data 23new_p6v11 --model base23fixO --n_exp 7 --seed 424242 --lr 3e-4 --n_par 16 \
    --guide iaf --num_flows 5 --hidden_dim_n 128 --renyi_alpha 1 --comment ""

# python fit.py -i $SLURM_ARRAY_TASK_ID --fit_type hmc --n_step 0     -n 10000 \
#     --data 23new_p6v11 --model base23fixO --n_exp 7 --seed 4224 --comment ""

#===== pois =====
# python fit.py -i $SLURM_ARRAY_TASK_ID --fit_type svi --n_step 10000 -n 50000 \
#     --data pois23new --model pois --n_exp 7 --seed 424242 --lr 3e-4 --n_par 16 \
#     --guide iaf --num_flows 5 --hidden_dim_n 128 --renyi_alpha 1 --comment ""

# python fit.py -i $SLURM_ARRAY_TASK_ID --fit_type hmc --n_step 0     -n 10000 \
#     --data pois23new --model pois --n_exp 7 --seed 4224 --comment ""

#===== test =====
# run svi: python 1_fit.py -i 0 --fit_type svi -n 10000 --n_step 100 \
#     --data base23fix_smalldsk_deltapsf --model base23fixnexp2_deltapsf --n_exp 2 --comment test
# test hmc: python 1_fit.py -i 0 --fit_type testhmc \
#     --data base23fix_deltapsf_2 --model base23fix_1exp_deltapsf --n_exp 7 --comment test
# test pthmc: python 1_fit.py -i 0 --fit_type pthmc -n 10000 --n_step 2000 \
#     --data psc_deltapsf --model base23fix_deltapsf
