import numpy as np
import healpy as hp

def dnds(s, theta):
    a, n1, n2, n3, sb1, sb2 = theta
    dnds = a * (sb2 / sb1) ** -n2 * np.where(s < sb2,
                                             (s / sb2) ** (-n3),
                                             np.where((s >= sb2) * (s < sb1),
                                                      (s / sb2) ** (-n2),
                                                      (sb1 / sb2) ** (-n2) * (s / sb1) ** (-n1)))
    return dnds


def rejection_sample(temp, mask, n_ps, n_sample=1000):
    """Adapted from NPTF Sim
    
    temp: full map
    mask: full boolean map"""

    n_accept = 0
    th_s_accept = []
    ph_s_accept = []

    nside = hp.npix2nside(len(temp))

    # Do rejection sampling until more than the required number of PS coordinates are produced
    while n_accept < n_ps:

        th_s = np.arccos(np.random.uniform(-1, 1, size=n_sample))
        ph_s = np.random.uniform(0, 2*np.pi, size=n_sample)

        temp = temp.copy() / np.max(temp)
        nside = hp.npix2nside(len(temp))

        rnd = np.random.random(n_sample)
        pix = hp.ang2pix(nside, th_s, ph_s)

        accept = rnd <= temp[pix]

        th_s_accept += list(th_s[accept])
        ph_s_accept += list(ph_s[accept])

        n_accept += np.sum(accept)

    return th_s_accept[:n_ps], ph_s_accept[:n_ps]


def simulate(temp_pois, temp_ps_s, theta_s, mask_norm, psf_r_func, psf_scheme='original'):
    """
    theta_s = [theta_0, theta_1, ...]
    theta_0 = [Sps, n1, n2, n3, sb1, lambdas]
    Sps is the expected number of photons in mask_norm
    mask_sim = mask_norm

    temp_pois must be a full map with normalization as the expected number of photons in each pixel
    temp_ps_s must be a list of full maps, but they will be renormalized such that the mean number of photons in mask_norm is theta_i[0]
    """

    # poisson
    counts = np.random.poisson(temp_pois)

    # ps
    s_arr = np.logspace(-1, 2, 1000)
    n_pix = np.sum(~mask_norm)

    for theta, temp_ps in zip(theta_s, temp_ps_s):
        
        theta_1 = np.copy(theta)
        theta_1[0] = 1.0
        dnds_arr = dnds(s_arr, theta_1)
        expected_photon_per_source = np.trapz(s_arr*dnds_arr, s_arr) / np.trapz(dnds_arr, s_arr)
        expected_n_source = theta[0] * n_pix / expected_photon_per_source # in all of mask_norm
        n_source = np.random.poisson(expected_n_source)