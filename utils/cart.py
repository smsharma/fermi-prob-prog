import numpy as np
from reproject import reproject_from_healpix
from astropy.wcs import WCS
from astropy.coordinates import ICRS, Galactic

def make_wcs(center, size, pixelsize, frame="Galactic", projection="CAR"):
    xcenter, ycenter = center 
    xsize, ysize = size
    if projection.upper() not in ["TAN", "CAR"]:
        raise ValueError("unsupported projection: " % projection)
    if frame.upper() == "ICRS":
        ctype = ["RA---" + projection.upper(), "DEC--" + projection.upper()]
    elif frame.upper() == "GALACTIC":
        ctype = ["GLON-" + projection.upper(), "GLAT-" + projection.upper()]
    else:
        raise ValueError("unknown frame: " % frame)

    w = WCS(naxis=2)
    w.wcs.ctype = ctype
    w.wcs.crval = np.array([xcenter, ycenter])
    w.wcs.crpix = np.array([xsize / 2.0 - 0.5, ysize / 2.0 - 0.5])
    w.wcs.cdelt = np.array([-pixelsize, pixelsize])
    w.wcs.cunit = ["deg", "deg"]
    w.wcs.equinox = 2000.0
    return w

def to_cart(temp_hp, n_pixels=96, pixelsize=0.5, frame="Galactic"):
    wcs = make_wcs(center=(0.,0.), size=(n_pixels,n_pixels), pixelsize=pixelsize, frame=frame)
    return reproject_from_healpix((temp_hp, "Galactic"), wcs, shape_out=(n_pixels, n_pixels), nested=False)[0]