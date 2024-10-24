import sys

import numpy as np
import healpy as hp
import json
from tqdm import tqdm


sys.path.append("..")
from models.np_model import NPModel
from simulations.wrapper import simulator_for_model


if __name__ == '__main__':

    data_name = 'test'
    n_sim = 5
    delta_psf = False

    truth_dict = json.load(open("truth_dict_poisszero.json", "r"))
    m = NPModel(
        data=np.zeros((hp.nside2npix(128),)),
        use_flat_exposure=True,
        psf_tag='delta',
    ) # using dummy data

    sims = []
    for _ in tqdm(range(n_sim)):
        psf_scheme = 'true delta' if delta_psf else 'original'
        sims.append(simulator_for_model(m, truth_dict, no_psc_mask=True, flat_exposure=True, delta_psf=delta_psf, psf_scheme=psf_scheme))
    sims = np.array(sims)
    if delta_psf:
        fn = f"{data_name}_deltapsf_n{n_sim}.npy"
    else:
        fn = f"{data_name}_n{n_sim}.npy"

    np.save(f"../outputs/sims/{fn}", sims)