import sys


import numpy as np
import argparse

sys.path.append("/n/home07/yitians/fermi/fermi-prob-prog")
from models.np_model import NPModel


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=int, help="Index")
    parser.add_argument('-n', type=int, help='HMC sample number')
    parser.add_argument('--fit_type', type=str)
    args = parser.parse_args()

    wdir = "/n/home07/yitians/fermi/fermi-prob-prog/production"

    mask_norm = np.load(f"{wdir}/mask_norm.npy")
    mask_roi = np.load(f"{wdir}/mask_roi.npy")

    data = np.load(f"{wdir}/sim_truth_n30.npy")[i]
    data_full = np.zeros(hp.nside2npix(128))
    data_full[~mask_norm] = sims[i]
    data_in = data_full[~mask_roi]
    
    m = NPModel()