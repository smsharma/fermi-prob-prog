"""Functions for model validation."""

import os
import sys

import numpy as np
import arviz as az
import healpy as hp
from scipy import special
from scipy.optimize import minimize_scalar

import matplotlib.pyplot as plt
from matplotlib import colormaps as cms

WDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(WDIR)
from models.psf import KingPSF


#========== Coverage test ==========

def find_pdf_hdi_prob(pdf_arr, x_arr=None, x_truth=None, pdf_truth=None):
    """Find the probability for which the HDI includes the truth."""
    if x_truth is not None and x_arr is not None:
        pdf_truth = np.interp(x_truth, x_arr, pdf_arr)
    if pdf_truth is None:
        raise ValueError('Either (x_arr, x_truth) or pdf_truth must be provided.')
    
    return np.sum(pdf_arr[pdf_arr >= pdf_truth]) / np.sum(pdf_arr)

def find_hdi_prob(samples, value, low=0, high=1, level=15):
    """Recursively find probability for which the hdi have value on boundary."""
    mid = (low + high) / 2
    if level == 0:
        return mid
    hdi = az.hdi(samples, mid)
    if hdi[0] <= value and value <= hdi[1]:
        return find_hdi_prob(samples, value, low, mid, level-1)
    else:
        return find_hdi_prob(samples, value, mid, high, level-1)

def find_max_point(xs, ys, degree=4):
    # Preliminary max point based on raw data
    prelim_max_index = np.argmax(ys)
    # Determine indices for the surrounding 5 points on each side
    left = max(0, prelim_max_index - 5)
    right = min(len(xs) - 1, prelim_max_index + 5)
    # Slice arrays to focus on these points
    xs_local = xs[left:right+1]
    ys_local = ys[left:right+1]
    # Fit a polynomial of up to the specified degree to this local data
    coefs = np.polyfit(xs_local, ys_local, degree)
    # Find the local max within this region
    res = minimize_scalar(lambda x: -np.polyval(coefs, x), bounds=(min(xs_local), max(xs_local)), method='bounded')
    max_x = res.x
    max_y = np.polyval(coefs, max_x)
    return max_x, max_y

def roc_finite_sample_band(n_samples, mc_samples=10000):
    """Using MC, find the 95% containment band for ROC curve for a gaussian distribution."""
    invcdf_arr = []
    for _ in range(mc_samples):
        x_sample = np.random.normal(size=n_samples)
        p_sample = (special.erf(np.abs(x_sample)/np.sqrt(2)) - special.erf(-np.abs(x_sample)/np.sqrt(2))) / 2
        invcdf_arr.append(np.sort(p_sample))
    invcdf_arr = np.array(invcdf_arr)
    invcdf_upper = np.quantile(invcdf_arr, 0.975, axis=0)
    invcdf_lower = np.quantile(invcdf_arr, 0.025, axis=0)
    return invcdf_lower, invcdf_upper


#========== PSF ==========

def dnds(s, theta):
    a, n1, n2, n3, sb1, sb2 = theta
    dnds = a * (sb2 / sb1) ** -n2 * np.where(s < sb2, (s / sb2) ** (-n3), np.where((s >= sb2) * (s < sb1), (s / sb2) ** (-n2), (sb1 / sb2) ** (-n2) * (s / sb1) ** (-n1)))
    return dnds

class PDFSampler:
    def __init__(self, xvals, pofx):
        self.xvals = xvals
        self.pofx = pofx

        # Check p(x) >= 0 for all x, otherwise stop
        assert(np.all(pofx >= 0)), "pdf cannot be negative"

        # Sort values by their p(x) value, for more accurate sampling
        self.sortxvals = np.argsort(self.pofx)
        self.pofx = self.pofx[self.sortxvals]

        # Calculate cdf
        self.cdf = np.cumsum(self.pofx)

    def __call__(self, samples):
        unidraw = np.random.uniform(high=self.cdf[-1], size=samples)
        cdfdraw = np.searchsorted(self.cdf, unidraw)
        cdfdraw = self.sortxvals[cdfdraw]
        return self.xvals[cdfdraw]


def psf_corr(nside=128, num_f_bins=10, n_psf=50000, n_pts_per_psf=1000, f_trunc=0.01, psf_r_func=...,
             sample_psf_max=0.02, psf_samples=1000):
    # PSF can't extend beyond 180 degrees, so check hasn't been asked for
    assert (sample_psf_max <= np.pi), \
        "PSF on a sphere cannot extend more than 180 degrees"
    
    # Setup pdf of the psf
    # On a sphere the PSF correction as a function of r is sin(r)*PSF(r)
    radial_pdf = lambda r: np.sin(r) * psf_r_func(r)
    rvals = np.linspace(0, sample_psf_max, psf_samples)
    pofr = radial_pdf(rvals)
    dist = PDFSampler(rvals, pofr)

    # Create an array of n_psf points to put down psfs
    # Establish an array of n_ps unit vectors
    # By sampling vals from a normal, end up with uniform normed vectors
    xyz = np.random.normal(size=(n_psf, 3))
    xyz_unit = np.divide(xyz, np.linalg.norm(xyz, axis=1)[:, None])

    # Convert to array of theta and phi values
    # theta = arccos(z/r), and here r=1. Similar expression for phi
    theta_c = np.arccos(xyz_unit[:, 2])
    phi_c = np.arctan2(xyz_unit[:, 1], xyz_unit[:, 0])

    # Now put a point source down at each of these locations
    outlist = []
    for ps_i in range(n_psf):
        # For each point source put down n_pts_per_psf counts
        # Determine where they are placed on the map as determine by the psf
        dr = dist(n_pts_per_psf)
        dangle = np.random.uniform(0, 2 * np.pi, n_pts_per_psf)
        dtheta = dr * np.sin(dangle)
        dphi = dr * np.cos(dangle) / (np.sin(theta_c[ps_i] + dtheta / 2))

        # Now combine with position of point source to get the exact location
        theta_base = theta_c[ps_i] + dtheta
        phi_base = phi_c[ps_i] + dphi

        # Want 0 <= theta < pi; 0 <= phi < 2pi
        # Carefully map to ensure this is true
        theta_remap_north = np.where(theta_base > np.pi)[0]
        theta_base[theta_remap_north] = 2 * np.pi - theta_base[theta_remap_north]
        theta_remap_south = np.where(theta_base < 0)[0]
        theta_base[theta_remap_south] = -theta_base[theta_remap_south]

        phi_base[theta_remap_north] += np.pi
        phi_base[theta_remap_south] += np.pi
        phi_base = np.mod(phi_base, 2 * np.pi)

        # As the PSF extends to infinity, if draw a value a long way from the
        # centre can occasionally still have a theta value outside the default
        # range above. For any sensible PSF (much smaller than the size of the
        # sky) this happens rarely. As such we just cut these values out.
        #print(np.any(~ ((theta_base <= np.pi) & (theta_base >= 0))))
        good_val = np.where((theta_base <= np.pi) & (theta_base >= 0))[0]
        theta = theta_base[good_val]
        phi = phi_base[good_val]

        # Convert these values back to a healpix pixel
        pixel = hp.ang2pix(nside, theta, phi)

        # From this information determine the flux fraction per pixel
        mn = np.min(pixel)
        mx = np.max(pixel) + 1
        pixel_hist = np.histogram(pixel, bins=mx - mn, range=(mn, mx), density=True)[
            0]
        outlist.append(pixel_hist)

    f_values = np.concatenate(outlist)
    # f_values is now the full list of flux fractions from all psfs
    # Ignore values which fall below the cutoff f_trunc
    f_values_trunc = f_values[f_values >= f_trunc]

    # Rebin into the user defined number of bins
    rho_ary, f_bin_edges = np.histogram(f_values_trunc, bins=num_f_bins,
                                        range=(0., 1.))

    # Convert to output format
    df = f_bin_edges[1] - f_bin_edges[0]
    f_ary = (f_bin_edges[:-1] + f_bin_edges[1:]) / 2.
    rho_ary = rho_ary / (df * n_psf)
    rho_ary /= np.sum(df * f_ary * rho_ary)
    df_rho_div_f_ary = df * rho_ary / f_ary

    return f_ary, df_rho_div_f_ary


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



#========== Plotting functions ==========

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

    n_sim = len(ll_ss)
    for i in range(3):
        axs[i].axvline(v_truth, color='k', alpha=0.5)
        axs[i].set(xlabel=xlabel)
    axs[0].set(ylabel='log-likelihod', title='Varying Sps_dsk. Color=counts.')
    axs[1].set(ylabel='log-likelihod')
    axs[2].set(ylabel='likelihod')
    axs[1].set(ylim=(-10, 5))
    axs[2].set(ylim=(0, 1))

    return fig, axs


def plot_coverage(probs, labels=None):
    """
    Args:
        probs (2D array): first dim = curves; second dim = probabilities of HDI needed to include truth
        labels (list): labels for the curves
    """
    fig, ax = plt.subplots()

    n_run = len(probs[0])
    ax.fill_between([0,1], [0,1], color='lightgray')
    if labels is None:
        labels = [None for _ in probs]
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