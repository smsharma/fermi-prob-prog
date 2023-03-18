"""Utilities for producing ps masks using Fermi PSF.
See ../fermi_data_processing/psc/pro for reference.
"""

import os
import warnings
import logging
from tqdm import tqdm

from astropy.io import fits
from astropy.table import Table
import numpy as np
import healpy as hp
from scipy import interpolate


PSF_FITS_DIR = '../../data/fermi_psf'
# copied from  /zfs/yitians/anaconda3/envs/fermi/share/fermitools/data/caldb/data/glast/lat/bcf/psf/ or equivalent fermitools installation
MASK_SAVE_DIR = '../../data/masks'


def fermi_psf_base(x, sigma, gamma):
    """Modified from fermi_psf_base in fermi_psf_pass8.pro"""
    return (1-1/gamma) * (1 + 0.5*(x**2) / (gamma*sigma**2))**(-gamma) / (2*np.pi*sigma**2)


def fermi_psf_with_tail(x, score, gcore, Ntail, stail, gtail):
    """Modified from fermi_psf_with_tail in fermi_psf_pass8.pro"""
    fcore = 1 / (1 + Ntail * stail**2 / score**2)
    return fcore * fermi_psf_base(x, score, gcore) + (1-fcore) * fermi_psf_base(x, stail, gtail)


def fermi_psf_pass8(dangle, energy,
                    eventclass='ULTRACLEANVETO', eventtype=3,
                    irfs='P8R3', version='V3'):
    """Modified from fermi_psf_pass8.pro (removed all pass 7 options).
    
    Parameters
    ----------
    dangle : float or array-like
        Angular distance(s) from the source in radians, either a single value or an array.
    energy : float
        Energy value in MeV for which the point spread function (PSF) will be computed.

    Returns
    -------
    float or numpy.ndarray
        PSF values corresponding to the provided dangle(s).
    """
    
    assert np.ndim(energy) == 0
    
    if eventtype >= 1 and eventtype <= 2:
        cuttype = 'FB'
        partitionval = eventtype - 1
        # eventtype=1 corresponds to front, eventtype=2 corresponds to back
    if eventtype >= 3 and eventtype <= 6:
        cuttype = 'PSF'
        partitionval = 6 - eventtype
        # eventtype=3 corresponds to the best PSF, which is last in the file
    if eventtype >= 6 and eventtype <= 9:
        cuttype = 'EDISP'
        partitionval = 9 - eventtype
        # likewise, eventtype=6 will correspond to the best EDISP, which is last in the file
            
    psf_file = f'{PSF_FITS_DIR}/psf_{irfs}_{eventclass}_{version}_{cuttype}.fits'
    print(f'fermi_psf_pass8: reading {psf_file}')

    p = fits.getdata(psf_file, ext=1+3*partitionval)
    pp = fits.getdata(psf_file, ext=2+3*partitionval)
    # what does no_tdim do???
    # IDL code: pp = mrdfits(psf_path, 2 + 3 * partitionval, silent=True, no_tdim=True)

    c0, c1, bet = pp['PSFSCALE'][0]
    scalefac = np.sqrt((c0 * (energy / 100)**bet)**2 + c1**2)
    x = dangle / scalefac
    
    ctheta = (p['CTHETA_LO'][0] + p['CTHETA_HI'][0]) / 2
    ebin = np.sqrt(p['ENERG_LO'][0] * p['ENERG_HI'][0])

    eind = interpolate.interp1d(ebin,   np.arange(len(ebin)))   ( energy )
    cind = interpolate.interp1d(ctheta, np.arange(len(ctheta))) ( 0.9 )
    
    kwargs = {}
    for k in ['score', 'gcore', 'Ntail', 'stail', 'gtail']:
        table = p[k][0]
        kwargs[k] = interpolate.interp2d(
            np.arange(table.shape[1]), # x, ebin
            np.arange(table.shape[0]), # y, ctheta
            table
        ) (eind, cind)
    
    psf = np.squeeze(fermi_psf_with_tail(x, **kwargs))
    
    return psf


def get_CL_angle(psf_func, CL=0.95, FWHM=False):
    """
    Calculate the containment angle for a given PSF function and confidence level.
    
    Modified from fermi_cl_psf_pass8.pro.

    Parameters
    ----------
    psf_func : callable
        The point spread function to be used.
    CL : float
        Confidence level for containment angle. Default is 0.95.
    FWHM : bool
        If True, returns the full width at half maximum instead.

    Returns
    -------
    float
        Containment angle (degrees) or FWHM (degrees) depending on the `FWHM` flag.
    """
    
    if FWHM:
        logging.warning('FWHM set. Setting CL = 0.68!')
        CL = 0.68
        
    a_s = np.linspace(0, np.deg2rad(10), 1000)
    psf_s = psf_func(a_s)
    psf_intg = np.cumsum(psf_s * a_s) / np.sum(psf_s * a_s)
    
    CL_angle = interpolate.interp1d(psf_intg, a_s)(CL)
    
    if FWHM:
        CL_angle *= np.sqrt(4 * np.log(4)) / np.sqrt(2 * np.log(1 / (1 - 0.68)))
        
    return CL_angle


def make_psc_masks_pass8(energy=2000., CL=0.95, nside=512, catalog='4FGL-DR3',
                         eventclass='ULTRACLEANVETO', eventtype=3,
                         irfs='P8R3', version='V3', regenerate=False):
    """
    Generates a point spread function (PSF) mask for Fermi 4FGL Pass 8 data.
    Modified from fermi_psc_all_pass8_nick_4fgl.pro.
    
    Parameters
    ----------
    energy : float
        Energy of the PSF in MeV.
    CL : float
        Confidence level for the PSF mask, ranging from 0 to 1.
    nside : int
        HEALPix Nside parameter.
    catalog : str
        Source catalog name.
    eventclass : str
        Event class used for the PSF mask.
    eventtype : int
        Event type used for the PSF mask.
    irfs : str
        Instrument response functions (IRFs) used for the PSF mask.
    version : str
        Version of the data used for the PSF mask.
    regenerate : bool
        If True, the PSF mask will be regenerated.

    Returns
    -------
    numpy.ndarray
        A HEALPix map (1D array) representing the desired mask.
    """


    if eventtype == 3:
        eventtypename = 'bestpsf'
    else:
        raise NotImplementedError(eventtype)
        
    # Check whether file exists already
    MASK_FN = f'{MASK_SAVE_DIR}/allpscmask_{catalog}_CL{CL:.3f}_{energy:.3e}MeV_NSIDE{nside}_' \
              + f'{irfs}_{eventclass}_{version}_{eventtypename}.npy'
    if os.path.exists(MASK_FN) and not regenerate:
        print(f'Reusing generated {MASK_FN}')
        return np.load(MASK_FN)
    
    print(f'Generating {MASK_FN}...')
        
    psf_func = lambda dangle: fermi_psf_pass8(
        dangle, energy,
        eventclass=eventclass, eventtype=eventtype, irfs=irfs, version=version
    )
    CL_angle = get_CL_angle(psf_func)

    # Healpix coords
    l, b = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    npix = hp.nside2npix(nside)
    uv = hp.rotator.dir2vec(l, b, lonlat=True)

    # Source catalog
    if catalog == '4FGL-DR3':
        psc = Table.read('../../data/psc/gll_psc_v31.fit')
    else:
        raise NotImplementedError
    
    # Big object mask LMC, SMC, Orion, NGC 5090
    # last two are: 3C 454.3, LAT PSR J1836+5925, and Geminga
    lbig = [279.00, 302.30, 207, 309.15, 86.12,  88.8, 195.13]
    bbig = [-33.60, -44.7, -17, 18.91, -38.18, 25.0, 4.27]
    rad = [5, 2, 6, 4, 1.5, 1.5, 2]

    # objects chosen by eye
    leye, beye = np.loadtxt('../../data/psc/ptsource.dat', unpack=True)
    reye = np.zeros_like(beye)
    
    pscl = np.concatenate([psc['GLON'], leye, lbig])
    pscb = np.concatenate([psc['GLAT'], beye, bbig])
    pscr = np.concatenate([np.zeros(len(psc)), reye, rad])

    uvpsc = hp.rotator.dir2vec(pscl, pscb, lonlat=True)
    
    # Build mask
    mask = np.full(npix, False)

    for i, r in enumerate(tqdm(pscr)):
        cs = np.dot(uv.T, uvpsc[:, i])
        cs_pscri = np.cos(np.sqrt(CL_angle**2 + np.deg2rad(r)**2))
        mask = np.logical_or(mask, cs > cs_pscri)

    np.save(MASK_FN, mask)
    
    return mask


def get_GC_mask(hp_mask, extent=20, pixel_size=0.5, threshold=0.5):
    """
    Generates a Cartesian mask center in the galactic center from an all-sky
    HEALPix map.
    
    Parameters
    ----------
    hp_mask : numpy.ndarray
        Input HEALPix map (1D array) representing the all-sky mask.
    extent : float
        Semi-extent of the ROI in degrees.
    pixel_size : float
        Pixel size of the cartesian ROI, in degrees.
    threshold : float
        Threshold value used to determine if a pixel should be masked. Pixels
        with values greater than or equal to the threshold will be masked.
    threshold : float
        Threshold value used to determine if a pixel should be masked. Pixels
        with values greater than or equal to the threshold will be masked. When
        set to None, no thresholding is performed, and a smooth mask is
        returned.

    Returns
    -------
    numpy.ndarray
        A 2D array representing the galactic center ROI mask derived from the
        input HEALPix map.
    """

    
    l_edge_s = np.linspace(-extent, extent, int(2*extent/pixel_size+1))
    b_edge_s = np.linspace(-extent, extent, int(2*extent/pixel_size+1))
    l_s = (l_edge_s[1:] + l_edge_s[:-1]) / 2
    b_s = (b_edge_s[1:] + b_edge_s[:-1]) / 2
    l_grid, b_grid = np.meshgrid(l_s, b_s)

    mask = hp.pixelfunc.get_interp_val(hp_mask, l_grid, b_grid, lonlat=True)
    
    if threshold is None:
        return mask
    else:
        return mask >= threshold