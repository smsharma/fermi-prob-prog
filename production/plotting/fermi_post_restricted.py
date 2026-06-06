print('started')

import pickle

from fpp.utils.posterior import multi_corner

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc_file("../../matplotlibrc")

from common import *

print('finished imports')

plot_name = 'fermi-post-restricted'

labels = ['S_pib', 'S_ics', 'S_gce', 'Sps_dsk', 'Sps_gce', 'f_bulge_poiss', 'f_bulge_ps', 'gamma_poiss', 'gamma_ps', 'C', 'zs']
fn_dict = {
    '1' : f'{fits_dir}/fermi/hmc.p',
    '2' : f'{fits_dir}/fermi/hmc-mapinit-smallprior.p',
}

fit_type_d = {
    '1': 'Full prior range',
    '2': 'Restricted prior range',
}
colors_dict = {
    '1': 'gray',
    '2': 'C0',
}

s_in = {}
for key, fn in fn_dict.items():
    s = pickle.load(open(fn, 'rb'))
    s_in[key] = {k: s[k] for k in labels}

multi_corner(
    s_in, labels,
    colors_dict=colors_dict, legend_dict=fit_type_d, legend_loc=(0.15, 0.93),
    labels=[label_latex_d[k] for k in labels],
    label_kwargs={"fontsize": 30},
    save_fn=f'{plot_name}.png',
)