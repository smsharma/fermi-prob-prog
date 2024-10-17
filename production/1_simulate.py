import sys
import json
from tqdm import tqdm

import numpy as np
import healpy as hp

wdir = "/n/home07/yitians/fermi/fermi-prob-prog/production"
sys.path.append(f"{wdir}/..")
from simulations.wrapper import simulator_for_model
from models.np_model import NPModel


if __name__ == '__main__':

    out_dir = f"{wdir}/../outputs/simulations"

    truth_dict = json.load(open('truth_dict.json', 'r'))
    m = NPModel(data=np.zeros((hp.nside2npix(128),)), psf_tags=['king', 'old'])
    print('psf set only for model to load. psf not used.')

    sims = []
    for _ in tqdm(range(100)):
        sims.append(simulator_for_model(m, truth_dict, psf_scheme='original'))
    sims = np.array(sims)
    np.save(f"{out_dir}/sim_oldsim_n100.npy", sims)