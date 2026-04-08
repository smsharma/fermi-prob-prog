import os
import sys
import json
from tqdm import tqdm
import dill as pickle

import numpy as np

from fpp.utils.validation import find_hdi_prob

# def regularize_sample(s):
#     if 'f_bulge_poiss' in s:
#         return s
#     s['S_pib'] = s['S_pib_0'] + s['S_pib_1'] + s['S_pib_2']
#     s['theta_pib_ModelO'] = s['S_pib_0'] / s['S_pib']
#     s['theta_pib_ModelA'] = s['S_pib_1'] / s['S_pib']
#     s['theta_pib_ModelF'] = s['S_pib_2'] / s['S_pib']
#     s['S_ics'] = s['S_ics_0'] + s['S_ics_1'] + s['S_ics_2']
#     s['theta_ics_ModelO'] = s['S_ics_0'] / s['S_ics']
#     s['theta_ics_ModelA'] = s['S_ics_1'] / s['S_ics']
#     s['theta_ics_ModelF'] = s['S_ics_2'] / s['S_ics']
#     s['S_gce_blg'] = s['S_gce_blg_0'] + s['S_gce_blg_1'] + s['S_gce_blg_2'] + s['S_gce_blg_3'] + s['S_gce_blg_4']
#     s['S_gce'] = s['S_gce_blg'] + s['S_gce_nfw']
#     s['f_bulge_poiss'] = s['S_gce_blg'] / s['S_gce']

#     bulge_models = ["mcdermott2022", "mcdermott2022_bbp", "mcdermott2022_x", "macias2019", "coleman2019"]
#     for i, k in enumerate(bulge_models):
#         s[f'theta_poiss_{k}'] = s[f'S_gce_blg_{i}'] / s['S_gce_blg']
#     s['Sps_gce_blg'] = s['Sps_gce_blg_0'] + s['Sps_gce_blg_1'] + s['Sps_gce_blg_2'] + s['Sps_gce_blg_3'] + s['Sps_gce_blg_4']
#     s['Sps_gce'] = s['Sps_gce_blg'] + s['Sps_gce_nfw']
#     s['f_bulge_ps'] = s['Sps_gce_blg'] / s['Sps_gce']
#     for i, k in enumerate(bulge_models):
#         s[f'theta_ps_{k}'] = s[f'Sps_gce_blg_{i}'] / s['Sps_gce_blg']
#     return s

def regularize_sample(s):
    return s


if __name__ == '__main__':

    n_sim = 100
    truth_name = 'base230927'
    run_name = 'pois/svi'
    print(f"Run name: {run_name}")

    samples_dir = f"../outputs/production/fits/{run_name}"
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
        if 'hmc' in run_name:
            fn = f"{samples_dir}/{i}.p"
        else:
            fn = f"{samples_dir}/{i}.p"
        if not os.path.exists(fn):
            missing_list.append(i)
        else:
            samples_list.append(regularize_sample(pickle.load(open(fn, 'rb'))))
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
