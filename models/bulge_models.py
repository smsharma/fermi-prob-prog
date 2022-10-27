import healpy as hp
import numpy as np
from reproject import reproject_to_healpix
from utils.cart import make_wcs
from astropy.io import fits

class BulgeTemplates:

    def __init__(self, template_name="macias2019", nside_project=512, nside_out=128):

        self.nside_out = nside_out
        self.nside_project = nside_project

        # From https://github.com/chrisgordon1/galactic_bulge_templates
        if template_name == "macias2019":
            self.template = fits.open("../data/BoxyBulge_arxiv1901.03822_Normalized.fits")[0].data
            self.wcs = make_wcs([0, 0], [200, 200], 0.2)
        elif template_name == "coleman2019":
            self.template = fits.open("../data/Bulge_modulated_Coleman_etal_2019_Normalized.fits")[0].data
            self.wcs = make_wcs([0, 0], [200, 200], 0.2)

        # From https://github.com/samueldmcdermott/gcepy/tree/main/gcepy/inputs/excesses
        elif template_name == "mcdermott2022":
            self.template = np.flip(np.load("../data/external/bb_front_only_14_Ebin_20x20window_normal.npy")[0], -1)
            self.wcs = make_wcs([0, 0], [400, 400], 0.1)
        elif template_name == "mcdermott2022_bbp":
            self.template = np.flip(np.load("../data/external/bbp_front_only_14_Ebin_20x20window_normal.npy")[0], -1)
            self.wcs = make_wcs([0, 0], [400, 400], 0.1)
        elif template_name == "mcdermott2022_x":
            self.template = np.flip(np.load("../data/external/x_front_only_14_Ebin_20x20window_normal.npy")[0], -1)
            self.wcs = make_wcs([0, 0], [400, 400], 0.1)

    def __call__(self):
        template_hp, _ = np.nan_to_num(reproject_to_healpix((self.template, self.wcs), 'galactic', nside=self.nside_project))
        template_hp = hp.ud_grade(template_hp, nside_out=self.nside_out)
        return template_hp
