import os
import pickle
import numpy as np
import json

from fpp.utils.posterior import multi_corner

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc_file("../src/fpp/utils/matplotlibrc")

#from fpp.models.np_model import NPModel
#m = NPModel()
#z = np.load(os.environ['MYSTORE'] + f"/fermi/fermi-prob-prog/outputs/production/simulations/smallprior-0Alm.npy")


def  plot_corner():
    point_est = False
    full_prior_i = None
    # truth_fn = "../outputs/truths/truths_fullprior42-zeroAlm.json"
    truth_fn = os.environ['MYSTORE'] + f'/fermi/fermi-prob-prog/outputs/production/fits/fermi/hmc-mapinit-init.json'

    labels = [
        'S_pib', 'S_ics', 'S_iso', 'S_bub', 'S_gce', 'Sps_dsk', 'Sps_gce',
        'f_bulge_poiss', 'f_bulge_ps', 'gamma_poiss', 'gamma_ps', 'C', 'zs'
    ]
    fits_dir = os.environ['MYSTORE'] + f'/fermi/fermi-prob-prog/outputs/production/fits'
    config_dict = {
        'default' : (f'{fits_dir}/fermi/hmc.p', 'k'),
        # 'map init' : (f'{fits_dir}/fermi/hmc-mapinit.p', 'C0'),
        'smallprior' : (f'{fits_dir}/fermi/hmc-mapinit-smallprior-2.p', 'C0')
    }
    
    #==============================================================================
    s_in = {}
    labels_dict = {}
    colors_dict = {}
    for key, (path, color) in config_dict.items():
        s = pickle.load(open(path, 'rb'))
        s_in[key] = {k: s[k] for k in labels}
        labels_dict[key] = key
        colors_dict[key] = color
        
        if point_est:
            truth_dict = json.load(open(truth_fn, 'r'))
            if full_prior_i:
                truth_dict = truth_dict[full_prior_i]
            t_in = {k: truth_dict[k] for k in labels}
        else:
            t_in = None

    multi_corner(s_in, labels, point_est=t_in, colors_dict=colors_dict, legend_dict=labels_dict, save_fn='corner.png')

if __name__ == "__main__":

    plot_corner()

