import sys
from astropy.io import fits

filename = sys.argv[1]
hdul = fits.open(filename)

print(f'post processing {filename}... ', end='')

## Copy relevant header from hdu[0] to hdu[3]
for k in ['CTYPE1', 'CRPIX1', 'CRVAL1', 'CDELT1', 'CUNIT1',
          'CTYPE2', 'CRPIX2', 'CRVAL2', 'CDELT2', 'CUNIT2']:
    hdul[3].header.append(hdul[0].header.cards[k])
    
hdul.writeto(filename, overwrite=True)

print('done.')