import os
import sys
import json
from tqdm import tqdm
import dill as pickle

import numpy as np

from fpp.utils.validation import find_hdi_prob


if __name__ == '__main__':

    n_sim = 30
    truth_name = 'base230927'
    run_name = 'calibration/hmctd10-old-delta-2'
    print(f"Run name: {run_name}")

    samples_dir = os.environ['MYSTORE'] + "/fermi/fermi-prob-prog/outputs/production/fits/" + run_name
    theta_true = json.load(open(f"../outputs/truths/truth_dict_{truth_name}.json"))

    if 'pois' in run_name:
        ks = [
            "S_bub", "S_gce", "S_ics", "S_iso", "S_pib", "S_psc",
            "f_bulge_poiss", "gamma_poiss"
        ]
    else:
        ks = [
            "S_bub", "S_gce", "S_ics", "S_iso", "S_pib", "S_psc",
            "Sps_dsk", "n1_dsk", "n2_dsk", "n3_dsk", "sb1_dsk", "lambdas_dsk",
            "Sps_gce", "n1_gce", "n2_gce", "n3_gce", "sb1_gce", "lambdas_gce",
            "f_bulge_poiss", "f_bulge_ps", "gamma_poiss", "gamma_ps",
            "C", "zs"
        ]

    samples_list = []
    missing_list = []
    for i in tqdm(range(n_sim)):
        fn = f"{samples_dir}/{i}.p"
        if not os.path.exists(fn):
            missing_list.append(i)
        else:
            samples_list.append(pickle.load(open(fn, 'rb')))
    print(f"Missing {len(missing_list)} run(s): {missing_list}")
    if len(missing_list) == n_sim:
        raise ValueError("No samples found")
    
    prob_samples = {}
    for k in tqdm(ks):
        probs = []
        for i in range(len(samples_list)):
            samples_test = np.array(samples_list[i][k])
            truth_test = theta_true[k]
            probs.append(find_hdi_prob(samples_test, truth_test))
        prob_samples[k] = np.array(probs)

    p_nominal_actual_dict = {}
    for k in ks:
        p_nominal, p_actual = np.sort(prob_samples[k]), np.linspace(0, 1, len(prob_samples[k]))
        p_nominal_actual_dict[k] = (p_nominal, p_actual)

    pickle.dump(p_nominal_actual_dict, open(f"{samples_dir}/p_nominal_actual_dict.p", 'wb'))
