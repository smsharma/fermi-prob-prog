import sys

import numpy as np
import healpy as hp
import json
from tqdm import tqdm


sys.path.append("..")
from models.np_model import NPModel
from simulations.wrapper import simulator_for_model


if __name__ == '__main__':

    data_name = 'base230927'
    n_sim = 100
    delta_psf = False

    truth_dict = json.load(open("truth_dict_base230927.json", "r"))
    m = NPModel()

    sims = []
    for _ in tqdm(range(n_sim)):
        sims.append(simulator_for_model(m, truth_dict, sim_all=True))
    sims = np.array(sims)
    if delta_psf:
        fn = f"{data_name}_deltapsf_n{n_sim}.npy"
    else:
        fn = f"{data_name}_n{n_sim}.npy"

    np.save(f"../outputs/sims/{fn}", sims)