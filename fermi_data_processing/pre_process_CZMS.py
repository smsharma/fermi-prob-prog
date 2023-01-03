import numpy as np
import sys
from astropy.io import fits

in_fn = sys.argv[1]
out_fn = sys.argv[2]
print(f'pre-processing {in_fn}... ')
hdul_czms = fits.open(in_fn)

hdul_ccw = fits.open('ccw_base.fits')

hdul_ccw[0].data = hdul_czms[0].data
hdul_ccw[0].header['NAXIS1'] = 240
hdul_ccw[0].header['NAXIS2'] = 240
hdul_ccw[0].header['CRVAL1'] = 0.
hdul_ccw[0].header['CDELT1'] = 0.25
hdul_ccw[0].header['CRPIX1'] = 120.5
hdul_ccw[0].header['CRVAL2'] = 0.
hdul_ccw[0].header['CDELT2'] = 0.25
hdul_ccw[0].header['CRPIX2'] = 120.5
hdul_ccw[0].header['FLUX'] = -1
hdul_ccw.writeto(out_fn, overwrite=True)

print(f'written to {out_fn}.')