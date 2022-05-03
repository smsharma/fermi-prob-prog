#!/bin/bash

#SBATCH --job-name=simulate
#SBATCH --output=simulate_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=00:59:00
#SBATCH --account=iaifi_lab
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=smsharma@mit.com

source ~/.bashrc
source activate ddp

module load Anaconda3/2020.11
module load gcc/8.2.0-fasrc01
module load cudnn/8.1.0.77_cuda11.2-fasrc01
module load glib/2.56.1-fasrc01
module load openmpi/4.0.1-fasrc01
module load git/2.17.0-fasrc01

cd /n/dvorkin_lab/smsharma/mi-attribution/

python -u simulate.py -n 20000 --name train_${SLURM_ARRAY_TASK_ID} --dir /n/dvorkin_lab/smsharma/mi-attribution/