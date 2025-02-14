import sys

import numpy as np
import healpy as hp
import json
from tqdm import tqdm


sys.path.append("..")
from models.np_model import NPModel
from simulations.wrapper import simulator_for_model


if __name__ == '__main__':

    data_name = 'gceps_x5exp_deltapsf'
    n_sim = 30
    delta_psf = True
    flat_exposure = False

    truth_dict = json.load(open("truth_dict_gceps.json", "r"))
    m = NPModel(data=np.zeros(hp.nside2npix(128), dtype=np.int32)) # dummy data
    m.debug_exaggerate_exposure(5)

    sims = []
    for _ in tqdm(range(n_sim)):
        sims.append(simulator_for_model(m, truth_dict, sim_all=True, delta_psf=delta_psf, flat_exposure=flat_exposure))
    sims = np.array(sims)
    fn = f"{data_name}.npy"

    np.save(f"../outputs/sims/{fn}", sims)