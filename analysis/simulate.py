import numpy as np
import healpy as hp
import json
from tqdm import tqdm

from fpp.models.np_model import NPModel


if __name__ == '__main__':

    data_name = 'nmold_deltapsf_tmp'
    truth_name = 'base230927'
    n_sim = 1
    modifiers = ['deltapsf'] # ['deltapsf', 'flatexp', 'p6v11']
    

    np.random.seed(42)

    m = NPModel()

    sims = []
    for _ in tqdm(range(n_sim)):
        sim_map = m.simulate(
            vd = json.load(open(f"../outputs/truths/truth_dict_{truth_name}.json", "r")),
            modifiers = modifiers,
        )
        sims.append(sim_map)

    np.save(f"../outputs/production/simulations/{data_name}.npy", np.array(sims))