import sys
import json
from tqdm import tqdm
import dill as pickle

import numpy as np

wdir = "/n/home07/yitians/fermi/fermi-prob-prog/production"
sys.path.append(f"{wdir}/..")
from utils.validation import find_hdi_prob


if __name__ == '__main__':

    samples_dir = f"{wdir}/../outputs/fit/svi_240803"
    theta_true = json.load(open(f"{wdir}/theta_true_flat.json"))

    ks = [
        "S_bub", "S_gce", "S_ics", "S_iso", "S_pib", "S_psc",
        "Sps_dsk", "n1_dsk", "n2_dsk", "n3_dsk", "sb1_dsk", "lambdas_dsk",
        "Sps_gce", "n1_gce", "n2_gce", "n3_gce", "sb1_gce", "lambdas_gce",
        "f_bulge_poiss", "f_bulge_ps", "gamma_poiss", "gamma_ps",
        "C", "zs"
    ]
    n_sim = 30

    prob_samples = {}
    for i in tqdm(range(n_sim)):
        samples = pickle.load(open(f"{samples_dir}/svi_samples_i{i}_n50000.p", 'rb'))
        probs = []
        for k in ks:
            samples_test = samples[k]
            truth_test = theta_true[k]
            probs.append(find_hdi_prob(samples_test, truth_test))
        prob_samples[k] = np.array(probs)
    prob_samples = np.array(prob_samples).T

    p_nominal_actual_dict = {}
    for k in ks:
        p_nominal, p_actual = np.sort(prob_samples[k]), np.linspace(0, 1, len(prob_samples[i]))
        p_nominal_actual_dict[k] = (p_nominal, p_actual)

    pickle.dump(p_nominal_actual_dict, open(f"{samples_dir}/p_nominal_actual_dict.p", 'wb'))