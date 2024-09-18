import sys
import json
from tqdm import tqdm
import dill as pickle

import numpy as np

wdir = "/n/home07/yitians/fermi/fermi-prob-prog/production"
sys.path.append(f"{wdir}/..")
from utils.validation import find_hdi_prob


if __name__ == '__main__':

    samples_dir = f"{wdir}/../outputs/fit/hmc_gc11_deltapsf_20240908"
    theta_true = json.load(open(f"{wdir}/truth_dict_gc11.json"))

    ktype = 'gc11'
    if ktype == 'full':
        ks = [
            "S_bub", "S_gce", "S_ics", "S_iso", "S_pib", "S_psc",
            "Sps_dsk", "n1_dsk", "n2_dsk", "n3_dsk", "sb1_dsk", "lambdas_dsk",
            "Sps_gce", "n1_gce", "n2_gce", "n3_gce", "sb1_gce", "lambdas_gce",
            "f_bulge_poiss", "f_bulge_ps", "gamma_poiss", "gamma_ps",
            "C", "zs"
        ]
    elif ktype == 'single':
        ks = ['Sps_gce', 'lambdas_gce', 'n1_gce', 'n2_gce', 'n3_gce', 'sb1_gce']
    elif ktype == 'gc11':
        ks = ['Sps_nfw', 'lambdas_nfw', 'n1_nfw', 'n2_nfw', 'n3_nfw', 'sb1_nfw',
              'S_pib', 'S_ics', 'S_dsk', 'S_nfw', 'S_bub']
    else:
        raise ValueError(ktype)
    
    n_sim = 30

    samples_list = []
    for i in tqdm(range(n_sim)):
        # samples_list.append(pickle.load(open(f"{samples_dir}/svi_samples_i{i}_n50000_ns10000.p", 'rb')))
        samples_list.append(pickle.load(open(f"{samples_dir}/hmc_samples_i{i}_n10000_ns2000.p", 'rb')))
    
    prob_samples = {}
    for k in tqdm(ks):
        probs = []
        for i in range(n_sim):
            samples_test = np.array(samples_list[i][k])
            truth_test = theta_true[k]
            probs.append(find_hdi_prob(samples_test, truth_test))
        prob_samples[k] = np.array(probs)

    p_nominal_actual_dict = {}
    for k in ks:
        p_nominal, p_actual = np.sort(prob_samples[k]), np.linspace(0, 1, len(prob_samples[k]))
        p_nominal_actual_dict[k] = (p_nominal, p_actual)

    pickle.dump(p_nominal_actual_dict, open(f"{samples_dir}/p_nominal_actual_dict.p", 'wb'))
