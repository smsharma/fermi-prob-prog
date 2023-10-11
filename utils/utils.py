import os
import numpy as np
import healpy as hp
from scipy.optimize import minimize_scalar


def load_and_check(filename, use_memmap=False):
    # Don't load image files > 1 GB into memory
    if use_memmap and os.stat(filename).st_size > 1.0 * 1024 ** 3:
        data = np.load(filename, mmap_mode="c")
    else:
        data = np.load(filename)
    return data


def ring2nest(the_map, the_mask, nside=128, return_masked=True):
    embed_map = np.zeros((the_map.shape[0], hp.nside2npix(nside)))
    embed_map[:, ~the_mask] = the_map
    the_map = hp.reorder(embed_map, r2n=True)
    if return_masked:
        the_mask = hp.reorder(the_mask, r2n=True)
        return the_map[:, ~the_mask]
    return the_map


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