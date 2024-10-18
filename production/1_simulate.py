import sys

import numpy as np
import json
from tqdm import tqdm


sys.path.append("..")
from models.np_model import NPModel
from simulations.wrapper import simulator_for_model


if __name__ == '__main__':

    data_name = 's1k'
    n_sim = 100

    truth_dict = json.load(open("truth_dict.json", "r"))
    m = NPModel()

    sims = []
    for _ in tqdm(range(n_sim)):
        sims.append(simulator_for_model(m, truth_dict, no_psc_mask=True))
    sims = np.array(sims)
    np.save(f"../outputs/sims/sim_{data_name}_n{n_sim}.npy", sims)