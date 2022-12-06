import numpy as np
from scipy import interpolate
from scipy import signal

import pickle

import sys
sys.path.append('..')
from models.psf import KingPSF


class TemplateWithEnergy:
    """n*100*100 images of GC with semi-extent 25 deg, pixel size 0.5 deg.
    Initialize with eng=..., templates=... or file=..."""
    
    def __init__(self, **kwargs):
        
        if ('engs' in kwargs and 'templates' in kwargs):
            engs = kwargs['engs']
            templates = kwargs['templates']
        elif ('file' in kwargs):
            file_dict = pickle.load(open(kwargs['file'], 'rb'))
            engs = file_dict['engs']
            templates = file_dict['templates']
        else:
            raise ValueError('unknown initialization method.')
        
        if templates.shape[-2:] != (100, 100):
            raise ValueError('incorrect shape.')
            
        self.engs = engs # [MeV]
        self.templates = templates
        self.interpolator = interpolate.interp1d(self.engs, self.templates, axis=0)
    
    
    def at_eng(self, eng): # ([MeV])
        
        return self.interpolator(eng)
    
    
    def save(self, filename):
        
        save_dict = {
            'engs': self.engs,
            'templates': self.templates
        }
        pickle.dump(save_dict, open(filename, 'wb'))
        

class Exposure:
    """n*100*100 exposure images of GC with semi-extent 25 deg, pixel size 0.5 deg.
    Currently uses self.engs and central energy (and inner product with templates)."""
    
    def __init__(self, engs, exposures):
        
        if exposures.shape[-2:] != (100, 100):
            raise ValueError('incorrect shape.')
            
        self.engs = engs # [MeV]
        self.exposures = exposures
        self.init_psf()
        
    def init_psf(self, r_multiplier=1):
        # psf kernel
        kernel_semi_extent = 1.5 # [deg]
        pixel_size_kernel = 0.5 # [deg]
        n = int(2*kernel_semi_extent / pixel_size_kernel + 1)
        xs = np.linspace(-kernel_semi_extent, kernel_semi_extent, n)
        psf = KingPSF()
        self.psf_kernel = np.array([[psf.psf_fermi_r(np.deg2rad(np.sqrt(x**2+y**2))/r_multiplier) for x in xs] for y in xs])
        self.psf_kernel /= np.sum(self.psf_kernel)
        
    def expose(self, template, psf_smooth=False):
        
        template_padded = np.zeros((100, 100))
        for i, eng in enumerate(self.engs):
            template_padded += template.at_eng(eng) * self.exposures[i]
            
        if psf_smooth:
            template_padded = signal.convolve(template_padded, self.psf_kernel, mode='same')
            
        return template_padded[10:90, 10:90]