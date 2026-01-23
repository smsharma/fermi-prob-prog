import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np
import healpy as hp

from jax.example_libraries import stax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, autoguide
from numpyro.infer import MCMC, NUTS
from numpyro import optim
from numpyro import handlers

import optax
from einops import repeat

from fpp.likelihoods.npll_jax_1b import log_like_np
from fpp.models.scd import dnds_1b as dnds
from fpp.utils import create_mask as cm
from fpp.models.psf import KingPSF
from fpp.utils.psf_correction import PSFCorrection



class NPModelCMP:


    def __init__(self):

        self.n_exp = 7
        self.nside = 128
        self.data_dir = "/n/home07/yitians/fermi/fermi-prob-prog/data/fermi_data_573w/fermi_data_128/"

        self.data = jnp.array(np.load(f"{self.data_dir}/fermidata_counts.npy").astype(np.int32))
        self.exposure_map = jnp.array(np.load(f"{self.data_dir}/fermidata_exposure.npy"))

        mask_ps = hp.ud_grade(np.load(f"{self.data_dir}/../../mask_3fgl_0p8deg.npy"), nside_out=self.nside) > 0
        self.mask_roi   = cm.make_mask_total(nside=self.nside, band_mask=True, band_mask_range=2., mask_ring=True, inner=0, outer=25, custom_mask=mask_ps)
        self.mask_plane = cm.make_mask_total(nside=self.nside, band_mask=True, band_mask_range=2., mask_ring=True, inner=0, outer=25)
        self.normalization_mask = self.mask_plane

        self.k_max = np.max(np.array(self.data)[~self.mask_roi])

        self.pib = jnp.array(np.load(f"{self.data_dir}/template_Opi.npy"))
        self.ics = jnp.array(np.load(f"{self.data_dir}/template_Oic.npy"))
        self.psc = jnp.array(np.load(f"{self.data_dir}/template_psc_3fgl.npy"))
        self.bub = jnp.array(np.load(f"{self.data_dir}/template_bub.npy"))
        self.iso = jnp.array(np.load(f"{self.data_dir}/template_iso.npy"))
        self.dsk = jnp.array(np.load(f"{self.data_dir}/template_dsk_z0p3.npy"))
        self.nfw_1p0 = jnp.array(np.load(f"{self.data_dir}/template_nfw_g1p0.npy"))
        self.nfw_1p2 = jnp.array(np.load(f"{self.data_dir}/template_nfw_g1p2.npy"))
        self.blg = jnp.array(np.load(f"{self.data_dir}/../../nptfit_cmp/coleman2019_bulge_template.npy"))
        for t in [self.pib, self.ics, self.psc, self.bub, self.iso, self.dsk, self.nfw_1p0, self.nfw_1p2, self.blg]:
            t /= jnp.mean(t[~self.normalization_mask])

        self.get_psf_correction()
        self.get_exp_regions(self.n_exp)


    def get_psf_correction(self):

        kp = KingPSF()

        pc_inst = PSFCorrection(delay_compute=True, num_f_bins=15, nside=self.nside)
        pc_inst.psf_r_func = lambda r: kp.psf_fermi_r(r)
        pc_inst.sample_psf_max = 10.0 * kp.spe * (kp.score + kp.stail) / 2.0
        pc_inst.psf_samples = 10000
        pc_inst.psf_tag = "Fermi_PSF_2GeV2_nside{}".format(self.nside)
        pc_inst.make_or_load_psf_corr()

        self.f_ary = pc_inst.f_ary
        self.df_rho_div_f_ary = pc_inst.df_rho_div_f_ary
        self.df_rho_ary = self.f_ary * self.df_rho_div_f_ary


    def get_exp_regions(self, nexp):
        """ Divide up ROI into exposure regions
        """

        # Determine the pixels of the exposure regions
        pix_array = np.where(self.mask_roi == False)[0]
        exp_array = np.array([[pix_array[i], self.exposure_map[pix_array[i]]] for i in range(len(pix_array))])
        array_sorted = exp_array[np.argsort(exp_array[:, 1])]

        # Convert from list of exreg pixels to masks (int as used to index)
        array_split = np.array_split(array_sorted, nexp)
        expreg_array = [np.array([array_split[i][j][0] for j in range(len(array_split[i]))], dtype="int32") for i in range(len(array_split))]

        npix = len(self.mask_roi)

        self.expreg_mask = []
        for i in range(nexp):
            temp_mask = np.logical_not(np.zeros(npix))
            for j in range(len(expreg_array[i])):
                temp_mask[expreg_array[i][j]] = False
            self.expreg_mask.append(temp_mask)

        # Store the total and region by region mean exposure
        expreg_values = [[array_split[i][j][1] for j in range(len(array_split[i]))] for i in range(len(array_split))]

        self.exposure_means_list = jnp.array([np.mean(expreg_values[i]) for i in range(nexp)])
        self.exposure_mean = jnp.mean(self.exposure_means_list)

        self.expreg_indices = []
        for i in range(nexp):
            expreg_indices_temp = np.array([np.where(pix_array == elem)[0][0] for elem in expreg_array[i]])
            self.expreg_indices.append(jnp.array(expreg_indices_temp))
            
        self.expreg_indices = jnp.array(self.expreg_indices)


    def model(self):

        PRIOR_LOW = 1e-3
        PRIOR_HIGH_PIB = 14.
        PRIOR_HIGH_ICS = 14.
        PRIOR_HIGH = 5.
        
        mu = jnp.zeros_like(self.data)

        # poissonian templates
        mu += numpyro.sample(f'S_pib', dist.Uniform(PRIOR_LOW, PRIOR_HIGH_PIB)) * self.pib
        mu += numpyro.sample(f'S_ics', dist.Uniform(PRIOR_LOW, PRIOR_HIGH_ICS)) * self.ics
        mu += numpyro.sample(f'S_iso', dist.Uniform(PRIOR_LOW, PRIOR_HIGH)) * self.iso
        mu += numpyro.sample(f'S_bub', dist.Uniform(PRIOR_LOW, PRIOR_HIGH)) * self.bub
        mu += numpyro.sample(f'S_psc', dist.Uniform(PRIOR_LOW, PRIOR_HIGH)) * self.psc
        mu += numpyro.sample(f'S_nfw', dist.Uniform(PRIOR_LOW, PRIOR_HIGH)) * self.nfw_1p0
        # mu += numpyro.sample(f'S_blg', dist.Uniform(PRIOR_LOW, PRIOR_HIGH)) * self.blg

        # np templates
        npt_compressed = jnp.array([self.nfw_1p2, self.dsk])
        theta = []
        for ips, ps in enumerate(["nfw", "dsk"]):
            Sps = numpyro.sample(f"Sps_{ps}", dist.Uniform(PRIOR_LOW, PRIOR_HIGH))
            n1  = numpyro.sample(f"n1_{ps}",  dist.Uniform(4.0, 6.0))
            n2  = numpyro.sample(f"n2_{ps}",  dist.Uniform(-6., -5.))
            sb1 = numpyro.sample(f"sb1_{ps}", dist.Uniform(5., 40.0))

            theta_tmp = jnp.array([1., n1, n2, sb1])
            s_ary = jnp.logspace(-1., 2., 1000)
            dnds_ary = dnds(s_ary, theta_tmp)
            A = Sps / jnp.trapz(s_ary * dnds_ary, s_ary)
            theta.append([A, n1, n2, sb1])
        theta = jnp.array(theta)

        
        # likelihoods and exposure
        # Pad the last exposure region so that all are the same size
        exp_lens = [len(self.expreg_indices[i]) for i in range(len(self.expreg_indices))]
        n_pad = exp_lens[0] - exp_lens[-1]
        
        expreg_indices = jnp.zeros_like(self.expreg_indices)
        expreg_indices = expreg_indices.at[:-1].set(self.expreg_indices[:-1])
        expreg_indices = expreg_indices.at[-1].set(jnp.pad(self.expreg_indices[-1], (0, n_pad)))

        log_like_np_exp_vmapped = jax.vmap(log_like_np, in_axes=(0, 0, 1, 0, None, None, None, None))
                
        # Get relevant arrays for different exposure regions
        mu_batch = mu[~self.mask_roi][jnp.array(expreg_indices)]
        npt_compressed_batch = npt_compressed[:, ~self.mask_roi][:, jnp.array(expreg_indices)]
        data_batch = self.data[~self.mask_roi][jnp.array(expreg_indices)]
        exposure_multiplier = self.exposure_means_list / self.exposure_mean
        
        theta = repeat(theta, "n_ps n_param -> n_exp n_ps n_param", n_exp=len(expreg_indices))
        theta = theta.at[:, :, 0].set(theta[:, :, 0] / exposure_multiplier[:, None])
        theta = theta.at[:, :, -1].set(theta[:, :, -1] * exposure_multiplier[:, None])
        theta = theta.at[:, :, -2].set(theta[:, :, -2] * exposure_multiplier[:, None])
        
        with numpyro.plate("data", size=len(mu[~self.mask_roi]), dim=-1):
            
            log_like_exp = log_like_np_exp_vmapped(theta, mu_batch, npt_compressed_batch, data_batch, self.f_ary, self.df_rho_ary, self.k_max, len(expreg_indices[0]))
            loglike = jnp.concatenate(log_like_exp)[:len(mu[~self.mask_roi])]

            with handlers.mask(mask=~jnp.logical_or(jnp.isinf(loglike), jnp.isnan(loglike))):
                return numpyro.factor('log-likelihood', loglike)


    def fit_svi(self, n_steps=10000, lr=5e-3):

        self.guide = autoguide.AutoIAFNormal(
            self.model, num_flows=5, hidden_dims=[128, 128], nonlinearity=stax.Tanh
        )
        optimizer = optim.optax_to_numpyro(optax.chain(optax.clip(1.), optax.adam(lr)))
        loss = Trace_ELBO(num_particles=8, vectorize_particles=True)

        self.svi = SVI(self.model, self.guide, optimizer, loss)
        self.svi_results = self.svi.run(jax.random.PRNGKey(42), n_steps)
        return self.svi_results


    def get_svi_samples(self, num_samples=50000):
        
        self.svi_samples = self.guide.sample_posterior(
            rng_key=jax.random.PRNGKey(42),
            params=self.svi_results.params,
            sample_shape=(num_samples,)
        )

        return self.svi_samples


    def run_nuts(self, num_chains=4, num_warmup=500, num_samples=5000, step_size=0.1, **model_static_kwargs):
        
        kernel = NUTS(self.model, max_tree_depth=4, dense_mass=False, step_size=step_size)
        self.nuts_mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, chain_method='vectorized')
        self.nuts_mcmc.run(jax.random.PRNGKey(42), **model_static_kwargs)
        
        return self.nuts_mcmc