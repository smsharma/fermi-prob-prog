import os

batch = """#!/bin/bash
#SBATCH --job-name=train
## SBATCH --nodes=1
#SBATCH --mem=64GB
#SBATCH --time=23:59:00
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH --account=dvorkin_lab
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=smsharma@mit.com

source ~/.bashrc
conda activate deepsets
module load gcc/10.2.0-fasrc01

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
