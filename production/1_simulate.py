import sys
import json
from tqdm import tqdm

import numpy as np

wdir = "/n/home07/yitians/fermi/fermi-prob-prog/production"
sys.path.append(f"{wdir}/..")
from simulations.wrapper import simulator_for_model
from models.np_model import NPModel


if __name__ == '__main__':

    out_dir = f"{wdir}/../outputs/simulations"

    truth_dict = json.load(open('truth_dict.json', 'r'))
    m = NPModel()

    sims = []
    for _ in tqdm(range(30)):
        sims.append(simulator_for_model(m, truth_dict, psf_scheme='true delta'))
    sims = np.array(sims)
    np.save("sim_truth_deltapsf_n30.npy", sims)