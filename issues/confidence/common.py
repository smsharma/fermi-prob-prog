import sys
import healpy as hp
import numpy as np

sys.path.append("../..")
from utils.utils import find_max_point
from utils.validation import roc_finite_sample_band
from utils import create_mask as cm
from models.templates import LorimerDiskTemplate
from nptfit_func import psf_corr
from models.psf import KingPSF

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps as cms
mpl.rc_file("../../notebooks/matplotlibrc")

# parameters
nside = 128
data_dir = '../../data'

truth_dict = {
    'Sps_dsk' : 1.3,
    'zs' : 0.5,
    'C' : 2.5,
    'n1_dsk' : 5.0,
    'n2_dsk' : 1.3,
    'n3_dsk' : -5.4,
    'sb1_dsk' : 11.,
    'lambdas_dsk' : 0.4,
}

# templates
disk_template = LorimerDiskTemplate(nside=nside)

# mask
mask_ps = hp.ud_grade(np.load(f"{data_dir}/mask_3fgl_0p8deg.npy"), nside_out=nside) > 0
mask_roi = cm.make_mask_total(nside=nside, band_mask=True, band_mask_range=2., mask_ring=True, inner=0, outer=25, custom_mask=mask_ps)
mask_plane = cm.make_mask_total(nside=nside, band_mask=True, band_mask_range=2., mask_ring=True, inner=0, outer=25,)

# psf
def get_psf_info(psf='king', num_f_bins=30, sigma=0.5):

    if psf == 'king':
        kp = KingPSF()
        psf_r_func = kp.psf_fermi_r
        f_ary, df_rho_div_f_ary = psf_corr(psf_r_func=psf_r_func, num_f_bins=num_f_bins)

    elif psf == 'gaussian':
        psf_r_func = lambda r: np.exp(-0.5 * (r / sigma) ** 2) / (2 * np.pi * sigma ** 2)
        f_ary, df_rho_div_f_ary = psf_corr(psf_r_func=psf_r_func, num_f_bins=num_f_bins)

    elif psf == 'delta':
        sigma = np.deg2rad(0.001) / 3
        psf_r_func = lambda r: np.exp(-0.5 * (r / sigma) ** 2) / (2 * np.pi * sigma ** 2)
        f_ary, df_rho_div_f_ary = psf_corr(psf_r_func=psf_r_func, num_f_bins=num_f_bins)

    elif psf == 'true delta':
        psf_r_func = None
        f_edges = np.linspace(0, 1, num_f_bins + 1)
        f_ary = 0.5 * (f_edges[1:] + f_edges[:-1])
        df_rho_div_f_ary = np.zeros_like(f_ary)
        df_rho_div_f_ary[-1] = 1 / f_ary[-1]**2

    return f_ary, df_rho_div_f_ary

# psf_info_dict = {
#     'king': get_psf_info(psf='king', num_f_bins=30),
#     'delta': get_psf_info(psf='delta', num_f_bins=30),
#     'delta100': get_psf_info(psf='delta', num_f_bins=100),
#     'true delta': get_psf_info(psf='true delta', num_f_bins=100),
# }

def find_pdf_hdi_prob(pdf_arr, x_arr, x_truth):
    i_x = np.argmin(np.abs(x_arr - x_truth))
    pdf_truth = pdf_arr[i_x]
    return np.sum(pdf_arr[pdf_arr >= pdf_truth]) / np.sum(pdf_arr)


def plot_psf(f, df_rho_div_f, title='psf'):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ax.plot(f, f**2*df_rho_div_f)
    ax.fill_between(f, f**2*df_rho_div_f, 0, alpha=0.5)
    ax.set(xlabel=r'$f$', ylabel=r'$f\rho(f)df$', title=title)

    return fig, ax


def plot_1dll(ll_ss, v_s, v_truth, counts_s=None, xlabel='var'):
    """
    Args:
        ll_ss (2D array): first dim = samples, second dim = v
        v_s (1D array): values of v
        v_truth (float): true value of v
        counts_s (1D array): total counts of samples
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # 2 is x, 1 is y
    for i, ll_s in enumerate(ll_ss):
        
        if counts_s is not None:
            color_x = (counts_s[i] - np.min(counts_s)) / (np.max(counts_s) - np.min(counts_s))
            color = cms['viridis'](color_x)
        else:
            color = 'k'

        ax = axs[0]
        ax.plot(v_s, ll_s, color=color, alpha=0.5)
        Sm, llm = find_max_point(v_s, ll_s)
        ax.plot(Sm, llm, 'k.', ms=2)

        ax = axs[1]
        plot_s = ll_s - llm
        ax.plot(v_s, plot_s, color=color, alpha=0.5)

        ax = axs[2]
        plot_s = np.exp(ll_s - llm)
        ax.plot(v_s, plot_s, color=color, alpha=0.5)

    for i in range(3):
        axs[i].axvline(v_truth, color='k', alpha=0.5)
        axs[i].set(xlabel=xlabel)
    axs[0].set(ylabel='log-likelihod', title='Varying Sps_dsk. Color=counts.')
    axs[1].set(ylabel='log-likelihod')
    axs[2].set(ylabel='likelihod')
    axs[1].set(ylim=(-10, 5))
    axs[2].set(ylim=(0, 1))

    return fig, axs


def plot_coverage(probs, labels):
    """
    Args:
        probs (2D array): first dim = curves; second dim = probabilities of HDI needed to include truth
    """
    fig, ax = plt.subplots()

    n_run = len(probs[0])
    ax.fill_between([0,1], [0,1], color='lightgray')
    for prob, label in zip(probs, labels):
        ax.plot(np.sort(prob), np.linspace(0, 1, n_run), label=label)

    invcdf_lower, invcdf_upper = roc_finite_sample_band(n_run)
    ax.plot(invcdf_upper, np.linspace(0, 1, n_run), 'k:', label=f'{n_run} sample 95\% \ncontainment')
    ax.plot(invcdf_lower, np.linspace(0, 1, n_run), 'k:')

    ax.set(aspect=1)
    ax.set(xlabel='Coverage of HDI needed to include truth', ylabel='Fraction of realizations')
    ax.text(0.95, 0.05, 'overconfident', ha='right', va='center')
    ax.text(0.05, 0.95, 'underconfident', ha='left', va='center')

    fig.legend(bbox_to_anchor=(1, 1), loc='upper left', bbox_transform=ax.transAxes)
    plt.tight_layout()

    return fig, ax