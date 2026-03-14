import logging
logger = logging.getLogger(__name__)

import numpy as np
import healpy as hp

from fpp.simulations.simulate_ps import SimulateMap
from fpp.models.scd import dnds, dnds_1b
from fpp.models.psf import KingPSF
from fpp.utils.utils import np_trapezoid


def simulator(
    theta, temps_poiss, temps_ps,
    mask_norm, mask_sim,
    psf_r_func, exp_map, psf_scheme='original', sim1b=False
):

    npix = len(mask_norm)
    map_out = np.zeros(npix, dtype=np.float32)
    s_arr = np.logspace(-1, 2, 1000)

    norms_poiss = theta[:len(temps_poiss)]

    dnds_func = dnds_1b if sim1b else dnds
    PS_THETA_LEN = 4 if sim1b else 6

    while (np.sum(map_out) == 0) or np.sum(np.isnan(map_out)) or np.sum(np.isinf(map_out)):

        dnds_arr = []
        idx_theta_ps = len(temps_poiss)
        for temp_ps in temps_ps:
            dnds_arr_temp = np.array(dnds_func(s_arr, theta[idx_theta_ps : idx_theta_ps + PS_THETA_LEN]))
            s_exp = np_trapezoid(s_arr * dnds_arr_temp, s_arr)
            temp_ratio = np.sum(temp_ps[~mask_norm]) / np.sum(temp_ps)
            exp_ratio = np.mean(exp_map[~mask_norm]) / np.mean(exp_map)
            dnds_arr_temp *= theta[idx_theta_ps] * np.sum(~mask_norm) / s_exp / temp_ratio / exp_ratio
            dnds_arr.append(dnds_arr_temp)
            idx_theta_ps += PS_THETA_LEN
        exp_map_norm = exp_map / np.mean(exp_map) # * exp_ratio

        sm = SimulateMap(temps_poiss, norms_poiss, [s_arr] * len(temps_ps), dnds_arr, temps_ps, psf_r_func, exp_map_norm, mask_roi=mask_sim, psf_scheme=psf_scheme)
        map_out = sm.create_map()
        map_out[mask_sim] = 0.
        map_out = map_out.astype(np.float32)

    return map_out