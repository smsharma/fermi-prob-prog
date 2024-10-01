import os
import sys
import json
from tqdm import tqdm
import dill as pickle

import numpy as np

wdir = "/n/home07/yitians/fermi/fermi-prob-prog/production"
sys.path.append(f"{wdir}/..")
from utils.validation import find_hdi_prob


if __name__ == '__main__':

    #===== CONTROLS =====#
    model_name = 'gcfull'
    data_name = 'gcfull'
    samples_dir = f"{wdir}/../outputs/fit/svi_{model_name}_{data_name}_deltapsf_0930"
    n_sim = 30

    #===== SETTINGS =====#
    theta_true = json.load(open(f"{wdir}/truth_dict_{data_name}.json"))

    if model_name == 'full':
        ks = [
            "S_bub", "S_gce", "S_ics", "S_iso", "S_pib", "S_psc",
            "Sps_dsk", "n1_dsk", "n2_dsk", "n3_dsk", "sb1_dsk", "lambdas_dsk",
            "Sps_gce", "n1_gce", "n2_gce", "n3_gce", "sb1_gce", "lambdas_gce",
            "f_bulge_poiss", "f_bulge_ps", "gamma_poiss", "gamma_ps",
            "C", "zs"
        ]
    elif model_name == 'single':
        ks = ['Sps_gce', 'lambdas_gce', 'n1_gce', 'n2_gce', 'n3_gce', 'sb1_gce']
    elif model_name == 'gc11':
        ks = ['Sps_nfw', 'lambdas_nfw', 'n1_nfw', 'n2_nfw', 'n3_nfw', 'sb1_nfw',
              'S_pib', 'S_ics', 'S_dsk', 'S_nfw', 'S_bub']
    elif model_name == 'gc17':
        ks = ['Sps_nfw', 'lambdas_nfw', 'n1_nfw', 'n2_nfw', 'n3_nfw', 'sb1_nfw',
              'Sps_dsk', 'lambdas_dsk', 'n1_dsk', 'n2_dsk', 'n3_dsk', 'sb1_dsk',
              'S_pib', 'S_ics', 'S_dsk', 'S_nfw', 'S_bub']
    elif model_name == 'gc2':
        ks = ['Sps_nfw', 'Sps_dsk']
    elif model_name == 'gc2scf':
        ks = ['Sps_nfw', 'lambdas_nfw', 'n1_nfw', 'n2_nfw', 'n3_nfw', 'sb1_nfw',
              'Sps_dsk', 'lambdas_dsk', 'n1_dsk', 'n2_dsk', 'n3_dsk', 'sb1_dsk']
    elif model_name == 'gcfull':
        ks = ["S_pib", "S_ics", "S_iso", "S_bub", "S_psc", "S_blg", "S_nfw", "gamma_poiss",
              "Sps_nfw", "gamma_ps", "Sps_blg", "Sps_dsk", "zs", "C",
              "n1_gce", "n2_gce", "n3_gce", "sb1_gce", "lambdas_gce",
              "n1_dsk", "n2_dsk", "n3_dsk", "sb1_dsk", "lambdas_dsk",
              "Sps_gce", "f_bulge_ps"]
    else:
        raise ValueError(model_name)
    
    #===== COVERAGE =====#
    samples_list = []
    i_list = []
    for i in tqdm(range(n_sim)):
        fn = f"{samples_dir}/svi_samples_i{i}_n50000_ns10000.p"
        if not os.path.exists(fn):
            continue
        sample = pickle.load(open(fn, 'rb'))
        sample['Sps_gce'] = sample['Sps_nfw'] + sample['Sps_blg']
        sample['f_bulge_ps'] = sample['Sps_blg'] / sample['Sps_gce']
        samples_list.append(sample)
        i_list.append(i)
    print('existing files', i_list)
    
    prob_samples = {}
    for k in tqdm(ks):
        probs = []
        for i in range(len(i_list)):
            samples_test = np.array(samples_list[i][k])
            truth_test = theta_true[k]
            probs.append(find_hdi_prob(samples_test, truth_test))
        prob_samples[k] = np.array(probs)

    p_nominal_actual_dict = {}
    for k in ks:
        p_nominal, p_actual = np.sort(prob_samples[k]), np.linspace(0, 1, len(prob_samples[k]))
        p_nominal_actual_dict[k] = (p_nominal, p_actual)

    pickle.dump(p_nominal_actual_dict, open(f"{samples_dir}/p_nominal_actual_dict.p", 'wb'))
