import numpy as np
import healpy as hp
import json
from tqdm import tqdm

from fpp.models.np_model import NPModel
# from fpp.models.np_model_cmp import NPModelCMP
# from fpp.simulations.wrapper import simulator_for_model, simulator_for_model_p6v11
# from fpp.simulations.wrapper import simulator_for_cmp


if __name__ == '__main__':

    data_name = 'testdsk3'
    truth_name = 'testdsk'
    n_sim = 100
    modifiers = [] # ['deltapsf', 'flatexp']
    
    truth_dict = json.load(open(f"../outputs/truths/truth_dict_{truth_name}.json", "r"))
    m = NPModel()

    sims = []
    for _ in tqdm(range(n_sim)):
        sims.append(m.simulate(truth_dict, modifiers=modifiers))
    sims = np.array(sims)

    np.save(f"../outputs/simulations/{data_name}.npy", sims)