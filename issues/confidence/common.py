import healpy as hp
import numpy as np


from utils import create_mask as cm
from models.templates import LorimerDiskTemplate


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