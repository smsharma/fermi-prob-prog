"""Poisson models with energy binning."""

import numpy as np
jnp = np
import healpy as hp
from scipy.stats import poisson

from utils import create_mask as cm

from models.templates import NFWTemplate, LorimerDiskTemplate
from models.bulge_models import BulgeTemplates

class EbinTemplate:
    """Simple class for templates with energy binning.
    Currently only supports a predetermined energy binning.
    
    Parameters
    ----------
    data : ndarray, shape=(nebin, npix)
        Energy binned Healpix data arrays.
    fit_type : {'total norm', 'bin norm', 'power law'}
        Types of fit that should be carried out for this template.
    norm_mask : ndarray
        Mask used to normalize template. Not energy dependent. Not stored.
    norm_ie_range : tuple, (ie_from, ie_to), end exclusive
        Range of energy to normalize over.
    """
    
    def __init__(self, data, fit_type='bin norm', norm_mask=None, norm_ie_range=None):
        
        self.data = data
        self.fit_type = fit_type
        self.norm_ie_range = norm_ie_range
            
        if norm_mask is not None: # mask: 1 means to mask out, 0 means to include
            if self.fit_type == 'total norm':
                self.data /= jnp.mean(self.fit_data[self.norm_ie_range[0]:self.norm_ie_range[1], ~norm_mask])
            elif self.fit_type == 'bin norm':
                self.data /= jnp.mean(self.fit_data[:, ~norm_mask], axis=1)
            else:
                raise NotImplementedError(self.fit_type)
        
        
class EbinPoissonModel:
    
    def __init__(
        self,
        nside = 256,
        ps_cat = '3fgl',
        data_class = 'bestpsf-nopsc',
        temp_class = 'ultracleanveto-bestpsf',
        mask_class = 'fwhm000-0512-bestpsf-mask',
        fit_ie_range = (10, 20), # end exclusive
        mask_roi_r_outer = 20., # [deg]
        mask_roi_b = 2., # [deg]
        #dif_names = ['ccwa', 'ccwf'], # add best of CZMS
        dif_name = 'ccwa',
        #dif_hybrid = False,
        #bulge_names = ['mcdermott2022', 'mcdermott2022_bbp', 'mcdermott2022_x', 'macias2019', 'coleman2019'],
        bulge_name = 'mcdermott2022',
        #bulge_hybrid = False,
        #vary_gamma = False,
        #vary_disk = False,
        #l_max=0,
    ):
        
        self.nside = nside
        self.ps_cat = ps_cat
        self.data_class = data_class
        self.temp_class = temp_class
        self.mask_class = mask_class
        self.fit_ie_range = fit_ie_range
        self.mask_roi_r_outer = mask_roi_r_outer
        self.mask_roi_b = mask_roi_b
        
        #========== Data ==========
        self.data_dir = f'../data/fermi_data_573w/ebin'
        to_nside = lambda x: hp.pixelfunc.ud_grade(x, self.nside)
        self.counts = jnp.array(to_nside(np.load(f'{self.data_dir}/counts-{self.data_class}.npy')).astype(np.int32))
        self.exposure = to_nside(np.load(f'{self.data_dir}/exposure-{self.data_class}.npy'))
        
        #========== Mask ==========
        self.mask_ps = to_nside(np.load(f'{self.data_dir}/mask-{self.mask_class}.npy')) > 0
        norm_mask_r_outer = 25.
        norm_mask_b = 2.
        self.mask_rois = np.array([
            cm.make_mask_total(
                nside=self.nside,
                band_mask=True,
                band_mask_range=self.mask_roi_b,
                mask_ring=True,
                inner=0,
                outer=self.mask_roi_r_outer,
                custom_mask=mask_ps
            )
            for mask_ps in self.mask_ps
        ])
        self.mask_plane = cm.make_mask_total(
            nside=self.nside,
            band_mask=True,
            band_mask_range=norm_mask_b,
            mask_ring=True,
            inner=0,
            outer=norm_mask_r_outer,
        )
        self.normalization_mask = self.mask_plane
        
        #========== Load ==========
        temp_bub_slice = to_nside(np.load(f'../data/fermi_data_573w/fermi_data_256/template_bub.npy'))
        temp_dsk_slice = to_nside(np.load(f'../data/fermi_data_573w/fermi_data_256/template_dsk_z1p0.npy'))
        temp_blg_slice = BulgeTemplates(template_name=bulge_name, nside_out=self.nside)()
        temp_nfw_slice = to_nside(np.load(f'../data/fermi_data_573w/fermi_data_256/template_nfw_g1p0.npy'))
        
        temp_options = dict(
            fit_type = 'bin norm',
            norm_mask = self.normalization_mask,
            norm_ie_range = self.fit_ie_range,
        )
        self.temps = {
            'iso' : EbinTemplate(self.exposure.copy(), **temp_options),
            'psc' : EbinTemplate(to_nside(np.load(f'{self.data_dir}/psc-bestpsf-3fgl.npy')), **temp_options),
            'bub' : EbinTemplate(np.repeat([temp_bub_slice], 40), **temp_options),
            'dsk' : EbinTemplate(np.repeat([temp_dsk_slice], 40), **temp_options),
            'pib' : EbinTemplate(to_nside(np.load(f'{self.data_dir}/{dif_name}pibrem-{self.temp_class}.npy')), **temp_options),
            'ics' : EbinTemplate(to_nside(np.load(f'{self.data_dir}/{dif_name}ics-{self.temp_class}.npy')), **temp_options),
            'blg' : EbinTemplate(np.repeat([temp_blg_slice], 40), **temp_options),
            'nfw' : EbinTemplate(np.repeat([temp_nfw_slice], 40), **temp_options),
        }
        
        
    def log_likelihood_at_bin(self, params, ie):
        
        # params = [S_...]
        mask = self.mask_rois[ie]
        temps = np.array([
            self.temps[k][ie, ~mask]
            for k in ['iso', 'psc', 'bub', 'dsk', 'pib', 'ics', 'blg', 'nfw']
        ])
        data = self.counts[ie, ~mask]
        return - poisson.logpmf(data, np.dot(params, temps)).mean()
    
        
    def log_likelihood(self, params):
        
        # params = [S_...]
        temps = np.array(
            [self.temps[k] for k in ['iso', 'psc', 'bub', 'dsk', 'pib', 'ics', 'blg', 'nfw']]
        )
        data = self.counts
        
        total_ll = 0
        for ie in range(ie_from, ie_to):
            temps = np.array([t.data[ie, ~self.mask_rois[ie]] for t in temps_float_total])
            total_ll += - poisson.logpmf(
                data[ie, ~self.mask_rois[ie]],
                np.dot(params, temps)
            ).mean()
        
        return total_ll