import sys
import json
from tqdm import tqdm

import numpy as np

wdir = "/n/home07/yitians/fermi/fermi-prob-prog/production"
sys.path.append(f"{wdir}/..")
from models.np_model_gc import NPModelGC17
from models.np_model import NPModel
from simulations.wrapper import simulator

sys.path.append("..")
from models.psf import KingPSF
from utils import create_mask as cm


def simulator_for_model_gc17(m, vd, delta_psf=False):
    """Wrapper for simulator function.

    Sps: nfw
    S: nfw Opi Oic bub dsk

    Args:
        m (NPModel): model object
        vd (dict): Dictionary of truth parameters
        delta_psf (bool): Whether to use delta PSF
    """

    # poiss: 4
    nm = cm.make_mask_total(nside=m.nside, band_mask=True, band_mask_range=2., mask_ring=True, inner=0, outer=25,)
    dsk_temp = m.disk_template.get_template(zs=vd['zs'], C=vd['C'])
    nfw_temp = m.nfw_template.get_NFW2_template(gamma=vd['gamma_poiss'])
    temps_poiss = [
        m.temp_bub / np.mean(m.temp_bub[~nm]),
        m.pib[0] / np.mean(m.pib[0][~nm]),
        m.ics[0] / np.mean(m.ics[0][~nm]),
        dsk_temp / np.mean(dsk_temp[~nm]),
        nfw_temp / np.mean(nfw_temp[~nm]),
    ]
    temps_poiss = [np.array(t) for t in temps_poiss]
    theta = [
        vd['S_bub'],
        vd['S_pib'],
        vd['S_ics'],
        vd['S_dsk'],
        vd['S_nfw'],
    ]

    # ps: nfw dsk
    temp_ps_nfw = m.nfw_template.get_NFW2_template(gamma=vd['gamma_ps']) # we are not going to assume this is normalized
    temp_ps_nfw /= np.mean(temp_ps_nfw[~nm])

    temp_ps_dsk = dsk_temp / np.mean(dsk_temp[~nm])

    temps_ps = []
    for k, t in zip(['nfw', 'dsk'], [temp_ps_nfw, temp_ps_dsk]):
        if vd['Sps_'+k] > 0:
            temps_ps.append(np.array(t))
            # theta[0] should be expected photon count per pixel in normalization mask region
            theta += [vd['Sps_'+k], vd['n1_'+k], vd['n2_'+k], vd['n3_'+k], vd['sb1_'+k], vd['lambdas_'+k] * vd['sb1_'+k]]

    #mask_sim = np.zeros_like(m.data, dtype=bool) # simulate all
    mask_sim = nm
    mask_normalize_counts = nm
    mask_roi = nm

    kp = KingPSF()
    psf_r_func = lambda r: kp.psf_fermi_r(r)
    psf_scheme = 'true delta' if delta_psf else 'original'
        
    exp_map = np.array(m.exposure_map)

    return simulator(theta, temps_poiss, temps_ps, mask_sim, mask_normalize_counts, mask_roi, psf_r_func, exp_map, psf_scheme=psf_scheme)[0]


if __name__ == '__main__':

    out_dir = f"{wdir}/../outputs/simulations"
    n_sim = 100
    delta_psf = True

    truth_dict = json.load(open('truth_dict_gc17.json', 'r'))
    # m = NPModelGC17(psf_tags=['deltasimple'])
    m = NPModel()
    print('psf not used as model is just for template loading!')

    sims = []
    for _ in tqdm(range(n_sim)):
        sims.append(simulator_for_model_gc17(m, truth_dict, delta_psf=delta_psf))
    sims = np.array(sims)
    psf_name = "delta" if delta_psf else "king"
    np.save(f"{out_dir}/sim_gc17_{psf_name}psf_n{n_sim}_debug_npmodelsim.npy", sims)