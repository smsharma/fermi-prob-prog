import os
import sys
import argparse
import logging

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from astropy.io import fits

from utils import create_mask as cm
from utils.pdf_sampler import PDFSampler
from utils.cart import to_cart

from models.scd import dnds

logger = logging.getLogger(__name__)
sys.path.append("./")
sys.path.append("../")


class SimulateCartesianMaps:
    def __init__(self, extent=25, n_pixels=100, upsample_factor=8, nside=128, sigma_psf=0.1218, data_dir='.'):

        self.upsample_factor = upsample_factor
        self.n_pixels = n_pixels
        self.sigma_psf = sigma_psf

        pixelsize = 2 * extent / n_pixels
        self.pixelsize = pixelsize
        
        # Load standard templates
        temp_gce_cart = to_cart(np.load("{}/data/fermi_data/template_gce.npy".format(data_dir)), n_pixels=n_pixels, pixelsize=pixelsize)
        temp_dif_cart = to_cart(np.load("{}/data/fermi_data/template_dif.npy".format(data_dir)), n_pixels=n_pixels, pixelsize=pixelsize)
        temp_psc_cart = to_cart(np.load("{}/data/fermi_data/template_psc.npy".format(data_dir)), n_pixels=n_pixels, pixelsize=pixelsize)
        temp_iso_cart = to_cart(np.load("{}/data/fermi_data/template_iso.npy".format(data_dir)), n_pixels=n_pixels, pixelsize=pixelsize)
        temp_dsk_cart = to_cart(np.load("{}/data/fermi_data/template_dsk.npy".format(data_dir)), n_pixels=n_pixels, pixelsize=pixelsize)
        temp_bub_cart = to_cart(np.load("{}/data/fermi_data/template_bub.npy".format(data_dir)), n_pixels=n_pixels, pixelsize=pixelsize)

        # Load original exposure
        exp_original = to_cart(np.load("{}/data/fermi_data/fermidata_exposure.npy".format(data_dir)), n_pixels=n_pixels, pixelsize=pixelsize)
        exp_original_norm = exp_original / np.mean(exp_original)

        # Load Model O templates
        temp_mO_pibrem_cart = to_cart(np.load("{}/data/fermi_data/ModelO_r25_q1_pibrem.npy".format(data_dir)), n_pixels=n_pixels, pixelsize=pixelsize)
        temp_mO_ics_cart = to_cart(np.load("{}/data/fermi_data/ModelO_r25_q1_ics.npy".format(data_dir)), n_pixels=n_pixels, pixelsize=pixelsize)

        # Exposure of real data
        exp_fits = fits.open("/n/holyscratch01/iaifi_lab/yitians/exposure_ultracleanveto_bestpsf_joined.fits")

        pixel_size_data = 0.1
        pixel_size_target = 0.5
        upsample_factor_data = int(pixel_size_target / pixel_size_data)
        extent = 25
        i_e = 40

        exp = exp_fits[0].data[i_e:i_e + 11].sum(0)[int((90 - extent) / pixel_size_data):int((90 + extent) / 0.1),int((90 - extent) / 0.1):int((90 + extent) / pixel_size_data)]
        b = exp.shape[0] // upsample_factor_data
        exp_downsampled = exp.reshape(-1, upsample_factor_data, b, upsample_factor_data).sum((-1, -3))
        exp_downsampled_norm = exp_downsampled / np.mean(exp_downsampled)

        exp_ratio = exp_original_norm / exp_downsampled_norm

        # Templates
        self.temps_ps = np.array([temp_gce_cart / exp_original_norm, temp_dsk_cart / exp_original_norm])
        self.temps_poiss = np.array([temp_gce_cart * exp_ratio, temp_iso_cart * exp_ratio, temp_bub_cart * exp_ratio, temp_psc_cart * exp_ratio, temp_mO_pibrem_cart * exp_ratio, temp_mO_ics_cart * exp_ratio])

        # Make upsampled versions of PS templates

        self.temps_ps_upsampled = []

        mesh_dim = np.linspace(0, n_pixels, n_pixels * upsample_factor)

        for temp_ps in self.temps_ps:
            interp = RectBivariateSpline(np.arange(n_pixels), np.arange(n_pixels), temp_ps)
            temp_ps_upsampled = interp(mesh_dim, mesh_dim)
            self.temps_ps_upsampled.append(temp_ps_upsampled)
            
        interp = RectBivariateSpline(np.arange(n_pixels), np.arange(n_pixels), exp_downsampled_norm)
        self.exp_upsampled_norm = interp(mesh_dim, mesh_dim)

        # Priors
        self.prior_ps = [[0.001, 10.0, 1.1, -10.0, 5.0, 1., 0.001, 10.0, 1.1, -10.0, 5.0, 1.], 
                    [3., 20.0, 1.99, 1.99, 40.0, 4.99, 3., 20.0, 1.99, 1.99, 40.0, 4.99]]

        self.prior_poiss = [[0.001, 0.001, 0.001, 0.001, 10.0, 4.0], 
                    [3., 1.5, 1.5, 4., 20.0, 8.0]]

    def simulate(self, n_sim=50000):
            
        # Generate parameters from prior
        thetas = np.random.uniform(low=self.prior_poiss[0] + self.prior_ps[0], high=self.prior_poiss[1] + self.prior_ps[1], size=(n_sim, len(self.prior_ps[0]) + len(self.prior_poiss[0])))

        s_ary = np.logspace(-1, 2, 1000)
        logs_ary = np.log10(s_ary)
        dlogs_ary = np.diff(logs_ary)[0] # Spacing in log-space

        s_for_ds_ary = np.logspace(logs_ary[0] - dlogs_ary / 2.0, logs_ary[-1] + dlogs_ary / 2.0, len(s_ary) + 1)
        ds_ary = np.diff(s_for_ds_ary)

        x_maps = np.zeros((n_sim, self.n_pixels, self.n_pixels))
        theta_mean_counts = np.zeros((n_sim, len(self.temps_ps) + len(self.temps_poiss)))
        dnds_ary = []

        for idx_theta, theta in enumerate(tqdm(thetas)):

            idx_theta_ps = len(self.temps_poiss)

            for idx, temp_ps in enumerate(self.temps_ps):

                dnds_ary_temp = dnds(s_ary, theta[idx_theta_ps:idx_theta_ps + 6])
                s_exp = np.trapz(s_ary * dnds_ary_temp, s_ary)
                dnds_ary_temp *= theta[idx_theta_ps] * np.prod(temp_ps.shape) / s_exp
                dnds_ary.append(dnds_ary_temp)
                idx_theta_ps += 6

                n_ps = np.random.poisson(np.trapz(dnds_ary_temp, s_ary))

                # Sample, accounting for dS factor for log-space sampling
                sample = PDFSampler(s_ary, ds_ary * dnds_ary_temp)(n_ps)

                dist = self.temps_ps_upsampled[idx]
                dist /= dist.sum() 
                pairs = np.indices(dimensions=self.temps_ps_upsampled[idx].shape).T # here are all of the x,y pairs 
                inds = np.random.choice(np.arange(np.prod(self.temps_ps_upsampled[idx].shape)), p=dist.reshape(-1),size=n_ps,replace=True)
                selections = pairs.reshape(-1, 2)[inds]
                hist = np.histogram2d(x=selections[:, 1], y=selections[:, 0], bins=self.n_pixels * self.upsample_factor, weights=sample)[0]

                sigma_mu = gaussian_filter(hist, sigma=self.sigma_psf / (self.pixelsize / self.upsample_factor), truncate=8.)
                sigma_mu *= self.exp_upsampled_norm
                signal = np.random.poisson(sigma_mu)

                b = signal.shape[0] // self.upsample_factor
                signal_reshaped = signal.reshape(-1, self.upsample_factor, b, self.upsample_factor).sum((-1, -3))

                x_maps[idx_theta, :, :] += signal_reshaped

                theta_mean_counts[idx_theta, idx] = signal_reshaped.mean()
            
            theta[0] /= self.temps_poiss[0].mean()
            
            theta_mean_counts[idx_theta, 2:] = (self.temps_poiss.T * theta[:len(self.temps_poiss)]).mean((0,1))
            x_maps[idx_theta, :, :] += np.random.poisson(np.einsum('i,ijk->jk', theta[:len(self.temps_poiss)], self.temps_poiss))

        results = {}
        results["x"] = x_maps
        results["theta"] = theta_mean_counts

        return results

    def save(self, data_dir, name, data):
        """Save simulated data to file"""

        logger.info("Saving results with name %s", name)

        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        if not os.path.exists("{}/data".format(data_dir)):
            os.mkdir("{}/data".format(data_dir))
        if not os.path.exists("{}/data/samples".format(data_dir)):
            os.mkdir("{}/data/samples".format(data_dir))

        for key, value in data.items():
            np.save("{}/data/samples/{}_{}.npy".format(data_dir, key, name), value)

def parse_args():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(description="Main high-level script that starts the GCE simulations")

    parser.add_argument("-n", type=int, default=10000, help="Number of samples to generate. Default is 10k.")
    parser.add_argument("--name", type=str, default="train", help='Sample name, like "train" or "test".')
    parser.add_argument("--dir", type=str, default=".", help="Base directory. Results will be saved in the data/samples subfolder.")
    parser.add_argument("--debug", action="store_true", help="Prints debug output.")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    logging.basicConfig(
        format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s",
        datefmt="%H:%M",
        level=logging.DEBUG if args.debug else logging.INFO,
    )
    logger.info("Hi!")

    sim_class = SimulateCartesianMaps(data_dir=args.dir)
    results = sim_class.simulate(n_sim=args.n)
    sim_class.save(args.dir, args.name, results)

    logger.info("All done! Have a nice day!")
