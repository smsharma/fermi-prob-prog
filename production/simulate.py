import numpy as np
import json
import os
from tqdm import tqdm

from fpp.models.np_model import NPModel


if __name__ == '__main__':

    truth_name = 'fullprior42-zeroAlm'
    data_name = 'fullprior42-zeroAlm'
    seed = 42
    modifiers = []  # ['deltapsf', 'flatexp']

    truths = json.load(open(f"../outputs/truths/truths_{truth_name}.json", 'r'))
    n_sim = len(truths)

    m = NPModel()

    child_rngs = np.random.default_rng(seed).spawn(n_sim)

    sims = []
    for i in tqdm(range(n_sim)):
        sim_map = m.simulate(truths[i], modifiers=modifiers, rng=child_rngs[i])
        sims.append(sim_map)

    out_path = os.environ['MYSTORE'] + f"/fermi/fermi-prob-prog/outputs/production/simulations/{data_name}.npy"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, np.array(sims))
    print(f"Saved {n_sim} simulations to {out_path}")
