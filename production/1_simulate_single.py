import sys
import json
from tqdm import tqdm

import numpy as np

wdir = "/n/home07/yitians/fermi/fermi-prob-prog/production"
sys.path.append(f"{wdir}/..")
from models.np_model import NPModel
from simulations.wrapper import simulator

sys.path.append("..")
from models.psf import KingPSF
from utils import create_mask as cm


def simulator_for_model(m, vd, no_psc_mask=False, delta_psf=False, no_plane_mask=False, psf_scheme='original'):
    """Wrapper for simulator function.

    Args:
        m (NPModel): model object
        vd (dict): Dictionary of truth parameters
    """

    # mask
    mask_outer = cm.make_mask_total(nside=m.nside, band_mask=False, mask_ring=True, inner=0, outer=25)

    # poiss: 0
    nm = m.normalization_mask
    temps_poiss = [m.temp_iso / np.mean(m.temp_iso[~nm])]
    temps_poiss = [np.array(t) for t in temps_poiss]
    theta = [0]

    # ps: nfw+blg*5 dsk
    # temp_ps
    temp_ps_nfw = m.nfw_template.get_NFW2_template(gamma=1.2) # we are going to assume this is not normalized
    temp_ps_nfw /= np.mean(temp_ps_nfw[~nm])
    temp_ps_gce = temp_ps_nfw

    temps_ps = []
    if vd['Sps_gce'] > 0:
        temps_ps.append(np.array(temp_ps_gce))
        # theta[0] should be expected photon count per pixel in normalization mask region
        theta += [vd['Sps_gce'], vd['n1_gce'], vd['n2_gce'], vd['n3_gce'], vd['sb1_gce'], vd['lambdas_gce'] * vd['sb1_gce']]

    #mask_sim = np.zeros_like(m.data, dtype=bool) # simulate all
    mask_sim = np.array(m.normalization_mask)
    mask_normalize_counts = np.array(m.normalization_mask)
    mask_roi = np.array(m.mask_roi)
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


if __name__ == '__main__':

    out_dir = f"{wdir}/../outputs/simulations"
    n_sim = 100

    truth_dict = json.load(open('truth_dict.json', 'r'))
    m = NPModel()

    sims = []
    for _ in tqdm(range(n_sim)):
        sims.append(simulator_for_model(m, truth_dict, psf_scheme='true delta'))
    sims = np.array(sims)
    np.save(f"{out_dir}/sim_Spsgce_deltapsf_n{n_sim}.npy", sims)