import numpy as np
import healpy as hp
import json
from tqdm import tqdm

from fpp.models.np_model import NPModel
from fpp.simulations.wrapper import simulator_for_model, simulator_for_model_p6v11


if __name__ == '__main__':

    data_name = 'pois2'
    n_sim = 100
    delta_psf = False
    flat_exposure = False
    sim_func = simulator_for_model

    truth_dict = json.load(open("../outputs/truths/truth_dict_pois230927.json", "r"))
    m = NPModel(data=np.zeros(hp.nside2npix(128), dtype=np.int32)) # dummy data
    # m.debug_exaggerate_exposure(5)

    sims = []
    for _ in tqdm(range(n_sim)):
        sims.append(sim_func(m, truth_dict, sim_all=True, delta_psf=delta_psf, flat_exposure=flat_exposure))
    sims = np.array(sims)

    np.save(f"../outputs/simulations/{data_name}.npy", sims)