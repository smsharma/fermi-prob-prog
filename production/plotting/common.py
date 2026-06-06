import matplotlib as mpl
import os

PROD_DIR = os.environ['MYSTORE'] + "/fermi/fermi-prob-prog/outputs/production"
FITS_DIR = PROD_DIR + '/fits'
TRUTH_DIR = PROD_DIR + '/../truths'
PLOTS_DIR = '../../outputs/plots'

label_latex_d = {
    'S_pib': r'$S_\mathrm{pib}$',
    'S_ics': r'$S_\mathrm{ics}$',
    'S_iso': r'$S_\mathrm{iso}$',
    'S_bub': r'$S_\mathrm{bub}$',
    'S_gce': r'$S_\mathrm{gce}$',
    'S_nfw': r'$S_\mathrm{nfw}$',
    'Sps_dsk': r'$S_\mathrm{dsk}^\mathrm{ps}$',
    'Sps_gce': r'$S_\mathrm{gce}^\mathrm{ps}$',
    'Sps_nfw': r'$S_\mathrm{nfw}^\mathrm{ps}$',
    'A_dsk': r'$A_\mathrm{dsk}^\mathrm{ps}$',
    'A_nfw': r'$A_\mathrm{nfw}^\mathrm{ps}$',
    'f_bulge_poiss': r'$f_\mathrm{bulge}^\mathrm{pois}$',
    'f_bulge_ps':    r'$f_\mathrm{bulge}^\mathrm{ps}$',
    'gamma_poiss': r'$\gamma^\mathrm{pois}$',
    'gamma_ps':    r'$\gamma^\mathrm{ps}$',
    'C': r'$C$',
    'zs': r'$z_s$',
    'sb_nfw': r'$S_b^\mathrm{nfw}$',
    'sb_dsk': r'$S_b^\mathrm{dsk}$',
}
fit_type_d = {
    'hmc': r'$\textbf{HMC}$',
    'svi': r'$\textbf{SVI}$',
    'nptfit': r'$\textbf{NPTFit}$',
    'oaf0': r'$\textbf{O}$+$\textbf{A}$+$\textbf{F}$',
    'oaf1': None,
    'oaf2': None,
    'o0': r'$\textbf{O}$',
    'o1': None,
    'o2': None
}
colors_dict = {
    'hmc': 'gray',
    'svi': 'C0',
    'nptfit': 'C1',
    'oaf0': mpl.colormaps['binary'](0.8),
    'oaf1': mpl.colormaps['binary'](0.5),
    'o0': mpl.colormaps['Blues'](0.8),
    'o1': mpl.colormaps['Blues'](0.5),
}