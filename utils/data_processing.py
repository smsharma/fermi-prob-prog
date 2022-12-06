"""For processing fermi data and templates"""

import numpy as np
from decimal import Decimal


def float_mod(x, y):
    r = Decimal(str(x)) % Decimal(str(y))
    return float(r)


def get_GC_data(hdu, extent, verbose=False):
    """Get square block around the galactic center from fits file with required (semi-)extent.
    Note: cannot wrap around. extent is semi-extent."""
    
    if not (hdu.header['CTYPE1'] == 'GLON-CAR' and hdu.header['CTYPE2'] == 'GLAT-CAR'):
        raise ValueError('unknown CTYPE.')
        
    l_pixel_size = np.abs(hdu.header['CDELT1'])
    b_pixel_size = np.abs(hdu.header['CDELT2'])
    
    if not (float_mod(extent, l_pixel_size) == 0 and float_mod(extent, b_pixel_size) == 0):
        raise ValueError('extent % pixel_size != 0.')
        
    l_crpix_ind = hdu.header['CRPIX1'] - 1
    b_crpix_ind = hdu.header['CRPIX2'] - 1
    l_crpix_left_edge_val = hdu.header['CRVAL1'] - l_pixel_size / 2 # left/right defined as array index small/large
    b_crpix_left_edge_val = hdu.header['CRVAL2'] - b_pixel_size / 2 # left/right defined as array index small/large
    
    l_ind_start = l_crpix_ind + (- extent - l_crpix_left_edge_val) / l_pixel_size
    l_ind_endp1 = l_crpix_ind + (  extent - l_crpix_left_edge_val) / l_pixel_size # end plus 1
    b_ind_start = b_crpix_ind + (- extent - b_crpix_left_edge_val) / b_pixel_size
    b_ind_endp1 = b_crpix_ind + (  extent - b_crpix_left_edge_val) / b_pixel_size # end plus 1
    
    if not (float_mod(l_ind_start, 1) == 0 and float_mod(l_ind_endp1, 1) == 0 \
            and float_mod(b_ind_start, 1) == 0 and float_mod(b_ind_endp1, 1) == 0):
        raise ValueError('non-integer index.')
    l_ind_start = int(l_ind_start)
    l_ind_endp1 = int(l_ind_endp1)
    b_ind_start = int(b_ind_start)
    b_ind_endp1 = int(b_ind_endp1)
    
    if verbose:
        print(f'get_GC_data: l: [{l_ind_start}:{l_ind_endp1}]')
        print(f'get_GC_data: b: [{b_ind_start}:{b_ind_endp1}]')
    
    if len(hdu.data.shape) == 2: # axes: b, l
        return hdu.data[b_ind_start:b_ind_endp1, l_ind_start:l_ind_endp1]
    elif len(hdu.data.shape) == 3: # axes: energy, b, l
        return hdu.data[:, b_ind_start:b_ind_endp1, l_ind_start:l_ind_endp1]
    else:
        raise ValueError(f'unknown dimension len(hdu.data.shape)={len(hdu.data.shape)}.')

        
def downsample(arr, f, verbose=False):
    """Converts a shape (h, w) array into a (h//f, w//f) array
    by adding the values in f*f blocks."""
    h, w = arr.shape
    if not (h%f == 0 and w%f == 0):
        raise ValueError('h and w need to be multiples of f.')
    if verbose:
        print(f'downsample: shape:({h},{w}) -> ({h//f},{w//f})')
    return np.sum([[arr[i::f, j::f] for j in range(f)] for i in range(f)], axis=(0,1))