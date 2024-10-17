import sys
import json
from tqdm import tqdm

import numpy as np
import healpy as hp

wdir = "/n/home07/yitians/fermi/fermi-prob-prog/production"
sys.path.append(f"{wdir}/..")
from models.np_model_gc import NPModelGCFull
from models.np_model import NPModel
from simulations.wrapper import simulator

sys.path.append("..")
from models.psf import KingPSF
from utils import create_mask as cm


def simulator_for_model_gcfull(m, vd, delta_psf=False, flat_exposure=False):
    """Wrapper for simulator function.

    Sps: nfw
    S: nfw Opi Oic bub dsk

    Args:
        m (NPModel): model object
        vd (dict): Dictionary of truth parameters
        delta_psf (bool): Whether to use delta PSF
    """

    nm = cm.make_mask_total(nside=m.nside, band_mask=True, band_mask_range=2., mask_ring=True, inner=0, outer=25,)

    # poissonian fixed (14): pib*3 ics*3 iso bub psc blg*5
    temps_poiss = [
        m.pib[0] / np.mean(m.pib[0][~nm]),
        m.pib[1] / np.mean(m.pib[1][~nm]),
        m.pib[2] / np.mean(m.pib[2][~nm]),
        m.ics[0] / np.mean(m.ics[0][~nm]),
        m.ics[1] / np.mean(m.ics[1][~nm]),
        m.ics[2] / np.mean(m.ics[2][~nm]),
        m.temp_iso / np.mean(m.temp_iso[~nm]),
        m.temp_bub / np.mean(m.temp_bub[~nm]),
        m.temp_psc / np.mean(m.temp_psc[~nm]),
        m.blg[0] / np.mean(m.blg[0][~nm]),
        m.blg[1] / np.mean(m.blg[1][~nm]),
        m.blg[2] / np.mean(m.blg[2][~nm]),
        m.blg[3] / np.mean(m.blg[3][~nm]),
        m.blg[4] / np.mean(m.blg[4][~nm]),
    ]
    temps_poiss = [np.array(t) for t in temps_poiss]
    theta = [
        vd['S_pib'] * vd['theta_pib'][0],
        vd['S_pib'] * vd['theta_pib'][1],
        vd['S_pib'] * vd['theta_pib'][2],
        vd['S_ics'] * vd['theta_ics'][0],
        vd['S_ics'] * vd['theta_ics'][1],
        vd['S_ics'] * vd['theta_ics'][2],
        vd['S_iso'],
        vd['S_bub'],
        vd['S_psc'],
        vd['S_blg'] * vd['theta_blg'][0],
        vd['S_blg'] * vd['theta_blg'][1],
        vd['S_blg'] * vd['theta_blg'][2],
        vd['S_blg'] * vd['theta_blg'][3],
        vd['S_blg'] * vd['theta_blg'][4],
    ]

    # poissonian variable (2): nfw
    temp_nfw_poiss = m.nfw_template.get_NFW2_template(gamma=vd['gamma_poiss'])
    temp_nfw_poiss = temp_nfw_poiss / np.mean(temp_nfw_poiss[~nm])
    temps_poiss.append(temp_nfw_poiss)
    theta.append(vd['S_nfw'])

    # non-poissonian: gce (7) + (5) dsk (3) + (5)
    temp_nfw_ps = m.nfw_template.get_NFW2_template(gamma=vd['gamma_ps']) # we are not going to assume this is normalized
    temp_nfw_ps /= np.mean(temp_nfw_ps[~nm])
    temp_gce_ps = vd['Sps_nfw'] * temp_nfw_ps
    for i in range(5):
        temp_gce_ps += vd[f'Sps_blg'] * vd['theta_blg_ps'][i] * m.blg[i] / np.mean(m.blg[i][~nm])
    Sps_gce = np.mean(temp_gce_ps[~nm])
    temp_gce_ps = temp_gce_ps / Sps_gce

    Sps_dsk = vd['Sps_dsk']
    temp_dsk_ps = m.disk_template.get_template(zs=vd['zs'], C=vd['C'])
    temp_dsk_ps = temp_dsk_ps / np.mean(temp_dsk_ps[~nm])

    temps_ps = []
    for k, t in zip(['gce', 'dsk'], [temp_gce_ps, temp_dsk_ps]):
        temps_ps.append(np.array(t))
        # theta[0] should be expected photon count per pixel in normalization mask region
        Sps = Sps_gce if k == 'gce' else Sps_dsk
        theta += [Sps, vd['n1_'+k], vd['n2_'+k], vd['n3_'+k], vd['sb1_'+k], vd['lambdas_'+k] * vd['sb1_'+k]]

    # simulate!
    mask_sim = nm
    mask_normalize_counts = nm
    mask_roi = nm

    kp = KingPSF()
    psf_r_func = lambda r: kp.psf_fermi_r(r)
    psf_scheme = 'true delta' if delta_psf else 'original'
    
    if not flat_exposure:
        exp_map = np.array(m.exposure_map)
    else:
        exp_map = np.ones_like(m.exposure_map)

    return simulator(theta, temps_poiss, temps_ps, mask_sim, mask_normalize_counts, mask_roi, psf_r_func, exp_map, psf_scheme=psf_scheme)[0]


if __name__ == '__main__':

    out_dir = f"{wdir}/../outputs/simulations"
    n_sim = 100
    delta_psf = False
    flat_exposure = False
    data_name = 'oldnfexp'

    truth_dict = json.load(open(f"truth_dict_{data_name}.json", 'r'))
    m = NPModelGCFull(psf_tags=['deltasimple'], data=np.zeros((hp.nside2npix(128),))) # dummy data
    # m = NPModel()
    print('psf not used as model is just for template loading!')

    sims = []
    for _ in tqdm(range(n_sim)):
        sims.append(simulator_for_model_gcfull(m, truth_dict, delta_psf=delta_psf, flat_exposure=flat_exposure))
    sims = np.array(sims)
    psf_name = "delta" if delta_psf else "king"
    np.save(f"{out_dir}/sim_{data_name}_{psf_name}psf_n{n_sim}.npy", sims)