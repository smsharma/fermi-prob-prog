"""Poisson models with energy binning."""

import numpy as np
jnp = np
import healpy as hp
from scipy.stats import poisson

from utils import create_mask as cm

from models.templates import NFWTemplate#, LorimerDiskTemplate
from models.bulge_models import BulgeTemplates

class EbinTemplate:
    """Simple class for templates with energy binning.
    Currently only supports a predetermined energy binning.
    
    Parameter
    ---------
    data : ndarray with shape=(nebin, npix)
    fit_type : {'total norm', 'power law', 'bin norm'}
    """
    
    def __init__(self, data, fit_type):
        self.data = data
        self.fit_type = fit_type
    
    def normalize_total_to_mask(self, mask, ie_from=None, ie_to=None):
        """mask: 1 means to mask out, 0 means to include"""
        if len(mask.shape) > 1:
            raise NotImplementedError # energy dependent mask, perhaps
            
        self.data /= jnp.mean(self.data[ie_from:ie_to, ~mask])
        
    def normalize_each_bin_to_mask(self, mask):
        """mask: 1 means to mask out, 0 means to include"""
        if len(mask.shape) > 1:
            raise NotImplementedError # energy dependent mask, perhaps
        
        self.data /= jnp.mean(self.data[:, ~mask], axis=1)
        
        
class EbinPoissonModel:
    
    def __init__(
        self,
        nside = 256,
        ps_cat = '3fgl',
        data_class = 'bestpsf-masked-nopsc',
        temp_class = 'ultracleanveto-bestpsf',
        mask_class = 'fwhm000-0512-bestpsf-mask',
        mask_roi_r_outer = 20., # [deg]
        mask_roi_b = 2., # [deg]
        dif_names = ['ccwa', 'ccwf'], # add best of CZMS
        dif_name = 'ccwa',
        #dif_hybrid = False,
        bulge_names = ['mcdermott2022', 'mcdermott2022_bbp', 'mcdermott2022_x', 'macias2019', 'coleman2019'],
        bulge_name = 'mcdermott2022',
        #bulge_hybrid = False,
        #vary_gamma = False,
        #vary_disk = False,
        #l_max=0,
    ):
        
        #========== General ==========
        self.nside = nside
        to_nside = lambda x: hp.pixelfunc.ud_grade(x, self.nside)
        self.ps_cat = ps_cat
        self.data_class = data_class
        self.temp_class = temp_class
        self.mask_class = mask_class
        self.mask_roi_r_outer = mask_roi_r_outer
        self.mask_roi_b = mask_roi_b
        
        print('Loading...', end=' ', flush=True)
        self.data_dir = f'../data/fermi_data_573w/ebin'
        self.counts = jnp.array(to_nside(np.load(f'{self.data_dir}/counts-{self.data_class}.npy')).astype(np.int32))
        print(np.sum(self.counts))
        print('counts', end=' ', flush=True)
        self.exposure = to_nside(np.load(f'{self.data_dir}/exposure-{self.data_class}.npy'))
        print('exposure', end=' ', flush=True)
        
        #========== Mask ==========
        self.mask_ps = to_nside(np.load(f'{self.data_dir}/mask-{self.mask_class}.npy')) > 0
        norm_mask_r_outer = 25.
        norm_mask_b = 2.
        self.mask_rois   = np.array([cm.make_mask_total(nside=self.nside, band_mask=True, band_mask_range=self.mask_roi_b, mask_ring=True, inner=0, outer=self.mask_roi_r_outer, custom_mask=mask_ps) for mask_ps in self.mask_ps])
        self.mask_plane = cm.make_mask_total(nside=self.nside, band_mask=True, band_mask_range=norm_mask_b,     mask_ring=True, inner=0, outer=norm_mask_r_outer,)
        self.normalization_mask = self.mask_plane
        
        #========== Load ========== (tmp)
        self.temp_iso = EbinTemplate(
            self.exposure.copy(),
            'float total'
        )
        self.temp_iso.normalize_total_to_mask(self.normalization_mask, 10, 20)
        print('iso', end=' ', flush=True)
        
        self.temp_psc = EbinTemplate(
            to_nside(np.load(f'{self.data_dir}/psc-bestpsf-3fgl.npy')),
            'float total'
        )
        self.temp_psc.normalize_total_to_mask(self.normalization_mask, 10, 20)
        print('psc', end=' ', flush=True)
        
        temp_bub_slice = to_nside(np.load(f'../data/fermi_data_573w/fermi_data_256/template_bub.npy'))
        self.temp_bub = EbinTemplate(
            np.repeat([temp_bub_slice], 40, axis=0),
            'float total'
        )
        self.temp_bub.normalize_total_to_mask(self.normalization_mask, 10, 20)
        print('bub', end=' ', flush=True)
        
        temp_dsk_slice = to_nside(np.load(f'../data/fermi_data_573w/fermi_data_256/template_dsk_z1p0.npy'))
        self.temp_dsk = EbinTemplate(
            np.repeat([temp_dsk_slice], 40, axis=0),
            'float total'
        )
        self.temp_dsk.normalize_total_to_mask(self.normalization_mask, 10, 20)
        print('dsk', end=' ', flush=True)
        
        self.temp_pib = EbinTemplate(
            to_nside(np.load(f'{self.data_dir}/{dif_name}pibrem-{self.temp_class}.npy')),
            'float total'
        )
        self.temp_pib.normalize_total_to_mask(self.normalization_mask, 10, 20)
        print('pib', end=' ', flush=True)
        
        self.temp_ics = EbinTemplate(
            to_nside(np.load(f'{self.data_dir}/{dif_name}ics-{self.temp_class}.npy')),
            'float total'
        )
        self.temp_ics.normalize_total_to_mask(self.normalization_mask, 10, 20)
        print('ics', end=' ', flush=True)
        
        temp_blg_slice = BulgeTemplates(template_name=bulge_name, nside_out=self.nside)()
        self.temp_blg = EbinTemplate(
            np.repeat([temp_blg_slice], 40, axis=0),
            'float total'
        )
        self.temp_blg.normalize_total_to_mask(self.normalization_mask, 10, 20)
        print('blg', end=' ', flush=True)
        
        # temp_nfw_slice = NFWTemplate(nside=self.nside).get_NFW2_template(gamma=1.)
        # self.temp_nfw = EbinTemplate(
        #     np.repeat([temp_nfw_slice], 40, axis=0),
        #     'float total'
        # )
        # self.temp_nfw.normalize_total_to_mask(self.normalization_mask, 10, 20)
        # print('nfw', end=' ', flush=True)
        
        temp_nfw_slice = to_nside(np.load(f'../data/fermi_data_573w/fermi_data_256/template_nfw_g1p0.npy'))
        self.temp_nfw = EbinTemplate(
            np.repeat([temp_nfw_slice], 40, axis=0),
            'float total'
        )
        self.temp_nfw.normalize_total_to_mask(self.normalization_mask, 10, 20)
        print('nfw', end=' ', flush=True)
        
        print('done.')
        
        
    def log_likelihood(self, params, ie_from=10, ie_to=20):
        
        # params = S_iso, S_psc, S_bub, S_dsk, S_pib, S_ics, S_blg, S_nfw
        
        temps_float_total = np.array([
            self.temp_iso,
            self.temp_psc,
            self.temp_bub,
            self.temp_dsk,
            self.temp_pib,
            self.temp_ics,
            self.temp_blg,
            self.temp_nfw,
        ])
        
        temps_float_bin = []
        
        data = self.counts
        
        total_ll = 0
        for ie in range(ie_from, ie_to):
            temps = np.array([t.data[ie, ~self.mask_rois[ie]] for t in temps_float_total])
            total_ll += - poisson.logpmf(
                data[ie, ~self.mask_rois[ie]],
                np.einsum("ij,i->j", temps, params)
            ).mean()
        
        return total_ll