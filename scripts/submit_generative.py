import os

batch = """#!/bin/bash
#SBATCH --job-name=train
## SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH --time=10:59:00
#SBATCH -p gpu
#SBATCH --gres=gpu:3
#SBATCH --account=dvorkin_lab
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=smsharma@mit.com

source ~/.bashrc

module load Anaconda3/2020.11
module load gcc/8.2.0-fasrc01
module load cudnn/8.1.0.77_cuda11.2-fasrc01
module load glib/2.56.1-fasrc01
module load openmpi/4.0.1-fasrc01
module load git/2.17.0-fasrc01

conda activate ddp

cd /n/dvorkin_lab/smsharma/mi-attribution/
"""

for add_unif_noise in [0, 1]:

    batchn = batch + "\n"

    batchn += "python -u train_generative.py --add_unif_noise {}".format(add_unif_noise)
    fname = "batch/submit.batch"
    f = open(fname, "w")
    f.write(batchn)
    f.close()
    os.system("chmod +x " + fname)
    os.system("sbatch " + fname)
