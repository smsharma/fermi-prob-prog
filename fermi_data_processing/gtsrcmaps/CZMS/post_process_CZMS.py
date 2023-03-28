import sys
from astropy.io import fits

filename = sys.argv[1]
hdul = fits.open(filename)

print(f'[post_process_CZMS.py]: post-processing {filename}...')

data_GC = hdul[3].data[:, 600:1200, 600:1200]

phdu = fits.PrimaryHDU(data_GC)

phdu.header.append(('CTYPE1', 'GLON-CAR'), end=True)
phdu.header.append(('CRPIX1', 300.5, 'Reference pixel'), end=True)
phdu.header.append(('CRVAL1', 0., 'GLON at the reference pixel'), end=True)
phdu.header.append(('CDELT1', 0.1, 'GLON increment'), end=True)
phdu.header.append(('CUNIT1', 'deg', 'GLON unit'), end=True)

phdu.header.append(('CTYPE2', 'GLAT-CAR'), end=True)
phdu.header.append(('CRPIX2', 300.5, 'Reference pixel'), end=True)
phdu.header.append(('CRVAL2', 0., 'GLAT at the reference pixel'), end=True)
phdu.header.append(('CDELT2', 0.1, 'GLAT increment'), end=True)
phdu.header.append(('CUNIT2', 'deg', 'GLAT unit'), end=True)

phdu.header.append(('CTYPE3', 'Energy', 'Log binned energy'), end=True)
phdu.header.append(('CUNIT3', 'MeV'), end=True)
phdu.header.append(('COMMENT', 'see energy table'), end=True)

ehdu = fits.open('/work/submit/yitians/fermi/fermi-prob-prog/fermi_data_processing/ccw_base.fits')[1]

hdul_new = fits.HDUList([phdu, ehdu])
hdul_new.writeto(filename, overwrite=True)

print(f'[post_process_CZMS.py]: written to {filename}.')