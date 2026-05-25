#!/usr/bin/env python
import os
import pickle
import numpy as np
import json

from fpp.utils.validation import pp_finite_sample_band
from fpp.utils.posterior import multi_corner

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc_file("../src/fpp/utils/matplotlibrc")


run_name = 'hmc-smallprior-0Alm-king-mapinit'
z = pickle.load(open(os.environ['MYSTORE'] + f'/fermi/fermi-prob-prog/outputs/production/fits/calibration/{run_name}/p_nominal_actual_dict.p', 'rb'))
keys = list(z.keys())
keys.sort()
print(' '.join(keys))


if 'pois' in run_name:
    labels = [
        'S_pib', 'S_ics', 'S_iso', 'S_bub', 'S_gce',
        'f_bulge_poiss', 'gamma_poiss',
    ]
else:
    labels = [
        'S_pib', 'S_ics', 'S_iso', 'S_bub', 'S_gce',
        'Sps_dsk', 'Sps_gce',
        'f_bulge_poiss', 'f_bulge_ps', 'gamma_poiss', 'gamma_ps', 'C', 'zs'
    ]

probs = [z[k] for k in labels]
ls_s = ['-'] * 9 + [':'] * 9

fig, ax = plt.subplots()

ax.fill_between([0,1], [0,1], color='lightgray')
if labels is None:
    labels = [None for _ in probs]
for prob, label, ls in zip(probs, labels, ls_s):
    ax.plot(prob[0], prob[1], label=label, ls=ls)

n_run = len(probs[0][0])
n_run = 100
lower, upper = pp_finite_sample_band(n_run)
ax.plot(upper, np.linspace(0, 1, n_run), 'k:', label=f'{n_run} sample 95\% \ncontainment')
ax.plot(lower, np.linspace(0, 1, n_run), 'k:')

ax.set(aspect=1, xlim=(0, 1), ylim=(0, 1))
ax.set(xlabel='Nominal coverage', ylabel='Actual coverage', title=run_name)

fig.legend(bbox_to_anchor=(1, 1), loc='upper left', bbox_transform=ax.transAxes)
plt.tight_layout()
plt.savefig(f'{run_name}.pdf')
