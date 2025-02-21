#!/bin/bash

#SBATCH --job-name=hmc_Dgceps_x1exp_deltapsf_Mgceps_7exp_deltapsf
#SBATCH --array=0-29
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
#SBATCH --mail-user=yitians@mit.com

source /n/home07/yitians/setup/torch.sh

cd /n/home07/yitians/fermi/fermi-prob-prog/production

# python 1_fit.py -i $SLURM_ARRAY_TASK_ID --fit_type svi --n_step 30000 -n 50000 --data base23fixfexp_deltapsf --model base23fix_deltapsf --seed 4242
python 1_fit.py -i $SLURM_ARRAY_TASK_ID --fit_type hmc -n 10000 --n_step 0 --data gceps_x1exp_deltapsf --model gceps_7exp_deltapsf --n_exp 7 --seed 4242

# run svi: python 1_fit.py -i 0 --fit_type svi -n 10000 --n_step 100 --data base23fix_smalldsk_deltapsf --model base23fixnexp2_deltapsf --n_exp 2 --comment test
# test hmc: python 1_fit.py -i 0 --fit_type testhmc --data gceps_x5exp_deltapsf --model gceps_fexp_deltapsf --n_exp 7 --comment test
# test pthmc: python 1_fit.py -i 0 --fit_type pthmc -n 10000 --n_step 2000 --data psc_deltapsf --model base23fix_deltapsf