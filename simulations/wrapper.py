import sys
import logging
logger = logging.getLogger(__name__)

import numpy as np
import healpy as hp
from simulations.simulate_ps import SimulateMap

sys.path.append("..")
from models.scd import dnds
from models.psf import KingPSF
from utils import create_mask as cm


def simulator(theta, temps_poiss, temps_ps, mask_sim, mask_normalize_counts, mask_roi, psf_r_func, exp_map, psf_scheme='original'):

    the_map = np.zeros(np.sum(~mask_sim))
    aux_vars = np.zeros(2)
    s_ary = np.logspace(-1, 2, 1000)

    good_map = False  # Check so map doesn't contain all zeros or nans or infs

    while not good_map:

        # Normalize poiss DM norm to get correct counts/pix in ROI
        norm_gce = theta[0] / np.mean(temps_poiss[0][~mask_normalize_counts])

        # Grab the rest of the poiss norms
        norms_poiss = theta[1 : len(temps_poiss)]

        # Normalize PS map to get correct counts/pix in ROI
        # and construct appropriate dnds arrays for each PS template

        dnds_ary = []
        idx_theta_ps = len(temps_poiss)
        for temp_ps in temps_ps:
            dnds_ary_temp = dnds(s_ary, theta[idx_theta_ps : idx_theta_ps + 6])
            s_exp = np.trapz(s_ary * dnds_ary_temp, s_ary)
            temp_ratio = np.sum(temp_ps[~mask_normalize_counts]) / np.sum(temp_ps)
            exp_ratio = np.mean(exp_map[~mask_normalize_counts]) / np.mean(exp_map)
            dnds_ary_temp *= theta[idx_theta_ps] * np.sum(~mask_normalize_counts) / s_exp / temp_ratio / exp_ratio
            dnds_ary.append(dnds_ary_temp)
            idx_theta_ps += 6

        exp_map_norm = exp_map / np.mean(exp_map)  #  * exp_ratio

        # Draw PSs and simulate map
        nside = hp.get_nside(temps_poiss[0])
        sm = SimulateMap(temps_poiss, [norm_gce] + list(norms_poiss), [s_ary] * len(temps_ps), dnds_ary, temps_ps, psf_r_func, exp_map_norm, mask_roi=mask_roi, nside=nside, psf_scheme=psf_scheme)

        the_map_temp = sm.create_map()

        the_map_temp[mask_roi] = 0.0
        the_map = the_map_temp[~mask_sim].astype(np.float32)

        # Grab auxiliary variables
        mean_map = np.mean(the_map)
        var_map = np.var(the_map)

        the_map = the_map.reshape((1, -1))
        # aux_vars = np.array([np.log(mean_map), np.log(np.sqrt(var_map))]).reshape((1, -1))

        # Resimulate if map is crap
        if (np.sum(the_map) == 0) or np.sum(np.isnan(the_map)) or np.sum(np.isinf(the_map)):
            good_map = False
            logger.info("Resimulating a crap map...")
        else:
            good_map = True

    return the_map


def toy_simulator(temps, vd, delta_psf=True):

    temps_poiss = [np.ones_like(temps[0])]
    theta = [1e-10]

    temps_ps = []
    for i in range(5):
        if f'Sps_t{i}' in vd and vd[f'Sps_t{i}'] > 0:
            temps_ps.append(np.array(temps[i]))
            theta += [vd[f'Sps_t{i}'], vd[f'n1_t{i}'], vd[f'n2_t{i}'], vd[f'n3_t{i}'], vd[f'sb1_t{i}'], vd[f'lambdas_t{i}'] * vd[f'sb1_t{i}']]

    if delta_psf:
        sigma = np.deg2rad(0.001) / 3
        psf_r_func = lambda r: np.exp(-0.5 * (r / sigma) ** 2) / (2 * np.pi * sigma ** 2)
    else:
        kp = KingPSF()
        psf_r_func = lambda r: kp.psf_fermi_r(r)
    exp_map = np.ones_like(temps[0])
    mask_sim = np.zeros_like(temps[0], dtype=bool)
    mask_normalize_counts = np.zeros_like(temps[0], dtype=bool)
    mask_roi = np.zeros_like(temps[0], dtype=bool)

    return simulator(theta, temps_poiss, temps_ps, mask_sim, mask_normalize_counts, mask_roi, psf_r_func, exp_map)[0]


def simulator_for_model(m, vd, no_psc_mask=False, delta_psf=False, no_plane_mask=False, psf_scheme='original'):
    """Wrapper for simulator function.

    Args:
        m (NPModel): model object
        vd (dict): Dictionary of truth parameters
    """

    # mask
    mask_outer = cm.make_mask_total(nside=m.nside, band_mask=False, mask_ring=True, inner=0, outer=25)

    # poiss: nfw iso bub psc pib*3 ics*3 blg*5
    nm = m.normalization_mask
    temps_poiss = [
        m.nfw_template.get_NFW2_template(gamma=vd['gamma_poiss']),
        m.temp_iso / np.mean(m.temp_iso[~nm]),
        m.temp_bub / np.mean(m.temp_bub[~nm]),
        m.temp_psc / np.mean(m.temp_psc[~nm]),
        m.pib[0] / np.mean(m.pib[0][~nm]),
        m.pib[1] / np.mean(m.pib[1][~nm]),
        m.pib[2] / np.mean(m.pib[2][~nm]),
        m.ics[0] / np.mean(m.ics[0][~nm]),
        m.ics[1] / np.mean(m.ics[1][~nm]),
        m.ics[2] / np.mean(m.ics[2][~nm]),
        m.bulge_templates[0] / np.mean(m.bulge_templates[0][~nm]),
        m.bulge_templates[1] / np.mean(m.bulge_templates[1][~nm]),
        m.bulge_templates[2] / np.mean(m.bulge_templates[2][~nm]),
        m.bulge_templates[3] / np.mean(m.bulge_templates[3][~nm]),
        m.bulge_templates[4] / np.mean(m.bulge_templates[4][~nm]),
    ]
    temps_poiss = [np.array(t) for t in temps_poiss]
    theta = [
        vd['S_gce'] * (1 - vd['f_bulge_poiss']),
        vd['S_iso'],
        vd['S_bub'],
        vd['S_psc'],
        vd['S_pib'] * vd['theta_pib'][0],
        vd['S_pib'] * vd['theta_pib'][1],
        vd['S_pib'] * vd['theta_pib'][2],
        vd['S_ics'] * vd['theta_ics'][0],
        vd['S_ics'] * vd['theta_ics'][1],
        vd['S_ics'] * vd['theta_ics'][2],
        vd['S_gce'] * vd['f_bulge_poiss'] * vd['theta_bulge_poiss'][0],
        vd['S_gce'] * vd['f_bulge_poiss'] * vd['theta_bulge_poiss'][1],
        vd['S_gce'] * vd['f_bulge_poiss'] * vd['theta_bulge_poiss'][2],
        vd['S_gce'] * vd['f_bulge_poiss'] * vd['theta_bulge_poiss'][3],
        vd['S_gce'] * vd['f_bulge_poiss'] * vd['theta_bulge_poiss'][4],
    ]

    # ps: nfw+blg*5 dsk
    # temp_ps
    temp_ps_nfw = m.nfw_template.get_NFW2_template(gamma=vd['gamma_ps']) # we are going to assume this is not normalized
    temp_ps_nfw /= np.mean(temp_ps_nfw[~nm])
    temp_ps_blg = np.einsum('i,ij->j', vd['theta_bulge_ps'], m.bulge_templates)
    temp_ps_blg /= np.mean(temp_ps_blg[~nm])
    temp_ps_gce = (1 - vd['f_bulge_ps']) * temp_ps_nfw + vd['f_bulge_ps'] * temp_ps_blg

    temp_ps_dsk = m.disk_template.get_template(zs=vd['zs'], C=vd['C'])
    temp_ps_dsk /= np.mean(temp_ps_dsk[~nm])

    temp_ps_iso = np.ones_like(temp_ps_nfw)

    temps_ps = []
    if vd['Sps_gce'] > 0:
        temps_ps.append(np.array(temp_ps_gce))
        # theta[0] should be expected photon count per pixel in normalization mask region
        theta += [vd['Sps_gce'], vd['n1_gce'], vd['n2_gce'], vd['n3_gce'], vd['sb1_gce'], vd['lambdas_gce'] * vd['sb1_gce']]
    if vd['Sps_dsk'] > 0:
        temps_ps.append(np.array(temp_ps_dsk))
        theta += [vd['Sps_dsk'], vd['n1_dsk'], vd['n2_dsk'], vd['n3_dsk'], vd['sb1_dsk'], vd['lambdas_dsk'] * vd['sb1_dsk']]
    if 'Sps_iso' in vd and vd['Sps_iso'] > 0:
        temps_ps.append(np.array(temp_ps_iso))
        theta += [vd['Sps_iso'], vd['n1_iso'], vd['n2_iso'], vd['n3_iso'], vd['sb1_iso'], vd['lambdas_iso'] * vd['sb1_iso']]

    #mask_sim = np.zeros_like(m.data, dtype=bool) # simulate all
    mask_sim = np.array(m.normalization_mask)
    mask_normalize_counts = np.array(m.normalization_mask)
    mask_roi = np.array(m.normalization_mask)
    if no_psc_mask:
        mask_roi = np.array(m.normalization_mask)
    if no_plane_mask:
        mask_roi = np.array(mask_outer)

    if delta_psf:
        sigma = np.deg2rad(0.001) / 3
        psf_r_func = lambda r: np.exp(-0.5 * (r / sigma) ** 2) / (2 * np.pi * sigma ** 2)
    else:
        kp = KingPSF()
        psf_r_func = lambda r: kp.psf_fermi_r(r)
    exp_map = np.array(m.exposure_map)

    return simulator(theta, temps_poiss, temps_ps, mask_sim, mask_normalize_counts, mask_roi, psf_r_func, exp_map, psf_scheme=psf_scheme)[0]

