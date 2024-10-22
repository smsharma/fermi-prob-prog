import sys

import numpy as np
import healpy as hp
import json
from tqdm import tqdm


sys.path.append("..")
from models.np_model import NPModel
from simulations.wrapper import simulator_for_model


if __name__ == '__main__':

    data_name = 's1k_fexp'
    n_sim = 100

    truth_dict = json.load(open("truth_dict.json", "r"))
    m = NPModel(
        data=np.zeros((hp.nside2npix(128),)),
        use_flat_exposure=True,
    ) # using dummy data

    sims = []
    for _ in tqdm(range(n_sim)):
        sims.append(simulator_for_model(m, truth_dict, no_psc_mask=True, flat_exposure=True))
    sims = np.array(sims)
    np.save(f"../outputs/sims/sim_{data_name}_n{n_sim}.npy", sims)