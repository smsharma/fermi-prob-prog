import os
import sys
import healpy as hp
import numpy as np
from tqdm import tqdm
import argparse

sys.path.append("../..")
from simulations.wrapper import simulator
from models.psf import KingPSF

from common import *

# simulate
def simulate(psf='king', sigma=...):

    vd = truth_dict
    nm = mask_plane
    temps_poiss = [np.ones(hp.nside2npix(nside))]
    theta = [0.,]
    
    temps_ps = []
    # temp_ps_dsk = disk_template.get_template(zs=vd['zs'], C=vd['C'])
    # temp_ps_dsk /= np.mean(temp_ps_dsk[~nm])
    # temps_ps.append(np.array(temp_ps_dsk))
    # theta += [vd['Sps_dsk'], vd['n1_dsk'], vd['n2_dsk'], vd['n3_dsk'], vd['sb1_dsk'], vd['lambdas_dsk'] * vd['sb1_dsk']]

    temp_ps_iso = np.ones(hp.nside2npix(nside))
    temps_ps.append(np.array(temp_ps_iso))
    theta += [vd['Sps_dsk'], vd['n1_dsk'], vd['n2_dsk'], vd['n3_dsk'], vd['sb1_dsk'], vd['lambdas_dsk'] * vd['sb1_dsk']] # use dsk params for iso

    mask_sim = np.array(nm)
    mask_normalize_counts = mask_sim
    exp_map = np.ones(hp.nside2npix(nside))

    if psf == 'delta':
        sigma = np.deg2rad(0.001) / 3
        psf_r_func = lambda r: np.exp(-0.5 * (r / sigma) ** 2) / (2 * np.pi * sigma ** 2)
        psf_scheme = 'original'
    elif psf == 'gaussian':
        psf_r_func = lambda r: np.exp(-0.5 * (r / sigma) ** 2) / (2 * np.pi * sigma ** 2)
        psf_scheme = 'original'
    elif psf == 'king':
        kp = KingPSF()
        psf_r_func = lambda r: kp.psf_fermi_r(r)
        psf_scheme = 'original'
    elif psf == 'true delta':
        psf_r_func = None
        psf_scheme = 'true delta'

    return simulator(theta, temps_poiss, temps_ps, mask_sim, mask_normalize_counts, nm, psf_r_func, exp_map, psf_scheme=psf_scheme)[0]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=int)
    args = parser.parse_args()
    
    save_dir = f'iso'
    os.makedirs(save_dir, exist_ok=True)

    for i in tqdm(range(100)):
        np.save(f"{save_dir}/counts_{i}.npy", simulate(psf='king', sigma=None))