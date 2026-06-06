import os

import numpy as np
import healpy as hp
import json
import pickle

from fpp.utils.validation import pp_finite_sample_band
from fpp.utils.posterior import multi_corner, dnds_posterior

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.image import imread
mpl.rc_file("../../matplotlibrc")

from common import *

#========== main: config ==========

def main():
    marginal()
    # marginal_extra()
    # fixed_truth('new')
    # pois()


#========== marginal ==========

def marginal():

    fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharey=True, sharex=True)

    coverage(axs[0,0], test_name='smallprior-0Alm', psf='delta', fit_type='hmc', show_ylabel=True,  show_xlabel=False)
    coverage(axs[0,1], test_name='smallprior-0Alm', psf='delta', fit_type='svi', show_ylabel=False, show_xlabel=False)
    coverage(axs[1,0], test_name='smallprior-0Alm', psf='king',  fit_type='hmc', show_ylabel=True,  show_xlabel=True)
    coverage(axs[1,1], test_name='smallprior-0Alm', psf='king',  fit_type='svi', show_ylabel=False, show_xlabel=True)
    axs[0,0].legend(loc=(0.02, 1.02), frameon=False, ncol=4)

    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.savefig(f'{PLOTS_DIR}/marginal-coverage.png', bbox_inches='tight', dpi=300)

def marginal_extra():

    fig, axs = plt.subplots(1, 2, figsize=(8, 5), sharey=True, sharex=True)

    coverage(axs[0], test_name='smallprior-0Alm', psf='delta', fit_type='hmc', show_ylabel=True,  show_xlabel=True)
    coverage(axs[1], test_name='smallprior-0Alm', psf='king', fit_type='hmc', show_ylabel=False, show_xlabel=True)
    axs[0].legend(loc=(0.02, 1.02), frameon=False, ncol=4)

    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.savefig(f'{PLOTS_DIR}/marginal-coverage-extra.png', bbox_inches='tight', dpi=300)



#========== old/new ==========

def fixed_truth_post_only():
    posterior(f'{PLOTS_DIR}/npold-delta-post.png', test_name='npold', deltapsf=True, i=0)

def fixed_truth(test_name): # 'old' or 'new'

    #===== coverage =====
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharey=True, sharex=True)

    coverage(axs[0,0], test_name=test_name, psf='delta', fit_type='hmc', show_ylabel=True,  show_xlabel=False)
    coverage(axs[0,1], test_name=test_name, psf='delta', fit_type='svi', show_ylabel=False, show_xlabel=False)
    coverage(axs[1,0], test_name=test_name, psf='king',  fit_type='hmc', show_ylabel=True,  show_xlabel=True)
    coverage(axs[1,1], test_name=test_name, psf='king',  fit_type='svi', show_ylabel=False, show_xlabel=True)
    axs[0,0].legend(loc=(0.02, 1.02), frameon=False, ncol=4)

    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.savefig(f'{PLOTS_DIR}/{test_name}-coverage.png', bbox_inches='tight', dpi=300)

    #===== posterior =====
    posterior(f'{PLOTS_DIR}/{test_name}-post.png', test_name=test_name, psf='king', i=0)

    #===== combined =====
    fn1 = f'{PLOTS_DIR}/{test_name}-coverage.png'
    fn2 = f'{PLOTS_DIR}/{test_name}-post.png'

    img1 = imread(fn1)
    img2 = imread(fn2)

    width_ratios = [1, 1.1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 9), gridspec_kw={'width_ratios': width_ratios})

    axes[0].imshow(img1)
    axes[0].axis('off')
    axes[1].imshow(img2)
    axes[1].axis('off')

    fig.subplots_adjust(wspace=0.)
    plt.savefig(f'{PLOTS_DIR}/{test_name}-combined.png', bbox_inches='tight', dpi=300)


#========== pois ==========

def pois():

    #===== coverage =====
    fig, axs = plt.subplots(1, 2, figsize=(8, 6), sharey=True)

    coverage(axs[0], test_name='pois', fit_type='hmc', show_ylabel=True, psf_names=False)
    axs[0].legend(loc=(0.02, 1.02), frameon=False, ncol=4, fontsize=14)
    coverage(axs[1], test_name='pois', fit_type='svi', show_ylabel=False, psf_names=False)

    fig.subplots_adjust(wspace=0.1)
    fig.savefig(f'{PLOTS_DIR}/pois-coverage.png', bbox_inches='tight', dpi=300)

    #===== posterior =====
    posterior(f'{PLOTS_DIR}/pois-post.png', test_name='pois', psf_names=False)

    #==== combined =====
    fn1 = f'{PLOTS_DIR}/pois-coverage.png'
    fn2 = f'{PLOTS_DIR}/pois-post.png'
    # Read the images
    img1 = imread(fn1)
    img2 = imread(fn2)

    width_ratios = [2, 1.2]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': width_ratios})
    axes[0].imshow(img1)
    axes[0].axis('off')
    axes[1].imshow(img2)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/pois-combined.png', bbox_inches='tight', dpi=300)


#========== coverage and posterior ==========

def coverage(ax, test_name, fit_type, psf=None, psf_names=True, show_legend=False, show_xlabel=True, show_ylabel=False):

    if test_name == 'pois':
        p_name = f'pois/{fit_type}/p_nominal_actual_dict.p'
        z = pickle.load(open(f'{FITS_DIR}/{p_name}', 'rb'))
    elif test_name in ['new', 'old', 'smallprior-0Alm', 'fullprior-0Alm']:
        comment = '-mapinit' if ('hmc' in fit_type and 'prior' in test_name) else ''
        p_name = f'calibration/{fit_type}-{test_name}-{psf}{comment}/p_nominal_actual_dict.p'
        z = pickle.load(open(f'{FITS_DIR}/{p_name}', 'rb'))
    else:
        raise ValueError(f'Unknown test_name: {test_name}')

    if 'prior' in test_name and 'svi' in fit_type:
        fit_type_label = fit_type_d[fit_type] + ' restricted prior'
    else:
        fit_type_label = fit_type_d[fit_type]

    if test_name == 'pois':
        labels = ['S_pib', 'S_ics', 'S_iso', 'S_bub', 'S_gce', 'f_bulge_poiss', 'gamma_poiss']
    else:
        labels = ['S_pib', 'S_ics', 'S_iso', 'S_bub', 'S_gce', 'Sps_dsk', 'Sps_gce', 'f_bulge_poiss', 'f_bulge_ps', 'gamma_poiss', 'gamma_ps', 'C', 'zs']

    probs = [z[k] for k in labels]
    ls_s = ['-'] * 7 + [':'] * 7

    # plt.rcParams["xtick.labelsize"] = 14
    # plt.rcParams["ytick.labelsize"] = 14

    ax.fill_between([0,1], [0,1], color='lightgray')
    for prob, label, ls in zip(probs, labels, ls_s):
        ax.plot(prob[0], prob[1], label=label_latex_d[label], ls=ls)

    n_run = 100
    invcdf_lower, invcdf_upper = pp_finite_sample_band(n_run)
    ax.plot(invcdf_upper, np.linspace(0, 1, n_run), 'k:', label=f'{n_run}-sample 95\%\ncontainment')
    ax.plot(invcdf_lower, np.linspace(0, 1, n_run), 'k:')

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    ax.set(aspect=1, xlim=(0, 1), ylim=(0, 1))
    if show_xlabel:
        ax.set(xlabel='Nominal coverage')
    if show_ylabel:
        ax.set(ylabel='Actual coverage')
    ax.text(0.05, 0.95, fit_type_label, fontsize=16, va='top', ha='left', transform=ax.transAxes)
    if psf_names:
        if psf == 'delta':
            ax.text(0.05, 0.85, r'Trivial PSF', fontsize=16, va='top', ha='left', transform=ax.transAxes)
        else:
            ax.text(0.05, 0.85, r'King PSF', fontsize=16, va='top', ha='left', transform=ax.transAxes)


def posterior(save_fn, test_name, psf=None, psf_names=True, i=0):

    if test_name == 'pois':
        labels = ['S_pib', 'S_ics', 'S_gce', 'f_bulge_poiss', 'gamma_poiss']
        fn_dict = {
            'hmc' : f'{FITS_DIR}/pois/hmc/{i}.p',
            'svi' : f'{FITS_DIR}/pois/svi/{i}.p',
        }
        truth_dict = json.load(open(TRUTH_DIR + '/truth_dict_pois230927.json', 'r'))
        legend_loc = (0.3, 0.86)

    elif test_name in ['old', 'new', 'smallprior-0Alm', 'fullprior-0Alm']:
        labels = ['S_pib', 'S_ics', 'S_gce', 'Sps_dsk', 'Sps_gce', 'f_bulge_poiss', 'f_bulge_ps', 'gamma_poiss', 'gamma_ps']
        fn_dict = {
            'hmc' : f'{FITS_DIR}/calibration/hmc-{test_name}-{psf}/{i}.p',
            'svi' : f'{FITS_DIR}/calibration/svi-{test_name}-{psf}/{i}.p',
        }
        truth_fn = f'{TRUTH_DIR}/truth_dict_base230927{"new" if test_name == "new" else ""}.json'
        truth_dict = json.load(open(truth_fn, 'r'))
        legend_loc = (0.19, 0.92)

    s_in = {}
    for key, fn in fn_dict.items():
        s = pickle.load(open(fn, 'rb'))
        s_in[key] = {k: s[k] for k in labels}
    if truth_dict is not None:
        t_in = {k: truth_dict[k] for k in labels}
    else:
        t_in = None

    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20

    if psf_names:
        legend_dict = {k : v + (r'$\quad$ Trivial PSF' if psf=='delta' else r'$\quad$ King PSF') for k, v in fit_type_d.items() if v is not None}
    else:
        legend_dict = fit_type_d

    multi_corner(
        s_in, labels, point_est=t_in, colors_dict=colors_dict, legend_dict=legend_dict, save_fn=save_fn,
        labels=[label_latex_d[k] for k in labels], label_kwargs={"fontsize": 40}, legend_loc=legend_loc
    )

if __name__ == "__main__":
    main()