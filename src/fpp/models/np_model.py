import os

import healpy as hp
import numpy as np

import jax.numpy as jnp
import jax
from jax.example_libraries import stax

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Predictive, Trace_ELBO, TraceGraph_ELBO, RenyiELBO, autoguide
from numpyro.infer.initialization import init_to_median, init_to_uniform
from numpyro.infer.reparam import NeuTraReparam
from numpyro.infer import MCMC, NUTS
from numpyro import optim
from numpyro.contrib.tfp.mcmc import ReplicaExchangeMC
from numpyro import handlers
from tensorflow_probability.substrates import jax as tfp

import optax
from einops import repeat

from fpp.models.svi import run_svi_with_beta
from fpp.models.scd import dnds
from fpp.models.templates import NFWTemplate, LorimerDiskTemplate
from fpp.models.bulge_models import BulgeTemplates
from fpp.likelihoods.npll_jax import log_like_np
from fpp.utils.sph_harm import Ylm
from fpp.utils import create_mask as cm
from fpp.models.psf import KingPSF
from fpp.utils.psf_correction import PSFCorrection
from fpp.simulations.simulator import simulator

import logging

wdir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(wdir, '../../../data')


class NPModel:
    """Non-Poissonian model.

    Args:
        nside (int):                 healpix nside.
        ps_cat {'3fgl', '4fgl'}:     point source catalog to use for masking.
        psf_tag {'king', 'delta'}:   point source PSF.
        n_exp (int):                 number of exposure regions to divide the ROI into (for non-Poissonian likelihood).
        l_max (int):                 maximum l for spherical harmonic modulation of the pib template.
        diffuse_names (list of str): which diffuse templates to include. Must be in ["ModelO", "ModelA", "ModelF"].
        bulge_names (list of str):   which bulge templates to include. Must be in ["mcdermott2022", "mcdermott2022_bbp", "mcdermott2022_x", "macias2019", "coleman2019"].
        data (np.ndarray):           optional data array to use instead of Fermi data.
    """
    def __init__(
        self,
        nside=128,
        ps_cat="3fgl",
        psf_tag='king',
        n_exp=7,
        l_max=2,
        diffuse_names=["ModelO", "ModelA", "ModelF"],
        bulge_names=["mcdermott2022", "mcdermott2022_bbp", "mcdermott2022_x", "macias2019", "coleman2019"],
        data=None,
    ):
        self.nside = nside
        self.ps_cat = ps_cat
        self.psf_tag = psf_tag
        self.n_exp = n_exp
        self.l_max = l_max
        self.dif_names = diffuse_names
        self.blg_names = bulge_names

        #===== data and masks =====
        self.data_dir = f"{data_dir}/fermi_data_573w/fermi_data_{self.nside}"
        if data is None:
            logging.warning('No data provided. Using Fermi data.')
            data = np.load(f"{self.data_dir}/fermidata_counts.npy")
        self.data = jnp.array(data).astype(jnp.int32)
        self.exposure = np.load(f"{self.data_dir}/fermidata_exposure.npy")
    
        if ps_cat == "3fgl":
            mask_ps = hp.ud_grade(np.load(f"{data_dir}/mask_3fgl_0p8deg.npy"), nside_out=self.nside) > 0
        elif ps_cat == "4fgl":
            mask_ps = hp.ud_grade(np.load(f"{data_dir}/fermi_data_573w/fermi_data_{nside}/fermidata_pscmask_4fgl.npy"), nside_out=self.nside) > 0
        else:
            raise NotImplementedError(ps_cat)
            
        self.mask_roi   = cm.make_mask_total(nside=self.nside, band_mask=True, band_mask_range=2., mask_ring=True, inner=0, outer=25, custom_mask=mask_ps)
        self.mask_plane = cm.make_mask_total(nside=self.nside, band_mask=True, band_mask_range=2., mask_ring=True, inner=0, outer=25,)
        self.nm = self.mask_plane # normalization mask
        print(f'Number of pixels in ROI: {np.sum(~self.mask_roi)}')

        #===== templates =====
        self.load_templates()
        self.get_sphharms()
            
        #==== point sources =====
        self.get_psf_correction(psf_tag=self.psf_tag)
        self.k_max = np.max(np.array(self.data)[~self.mask_roi])
        print(f'Max photon count is {self.k_max}')
        self.get_exp_regions(n_exp)

        #===== misc =====
        self.svi_results = None
        
        
    def get_psf_correction(self, psf_tag):
        if psf_tag == 'king':
            kp = KingPSF()
            pc_inst = PSFCorrection(delay_compute=True, num_f_bins=15, nside=self.nside)
            pc_inst.psf_r_func = lambda r: kp.psf_fermi_r(r)
            pc_inst.sample_psf_max = 10.0 * kp.spe * (kp.score + kp.stail) / 2.0
            pc_inst.psf_samples = 10000
            pc_inst.psf_tag = f"Fermi_PSF_2GeV2_nside{self.nside}"
            pc_inst.make_or_load_psf_corr()
            self.f_ary = pc_inst.f_ary
            self.df_rho_div_f_ary = pc_inst.df_rho_div_f_ary
            self.df_rho_ary = self.f_ary * self.df_rho_div_f_ary
        elif psf_tag == 'delta':
            self.f_ary = np.array([0., 1.])
            self.df_rho_ary = np.array([0., 1.])
        else:
            raise NotImplementedError(psf_tag)

    def get_sphharms(self):
        theta_s, phi_s = hp.pix2ang(self.nside, np.arange(hp.nside2npix(self.nside)))
        self.Ylm_temps = np.concatenate([[np.real(Ylm(l, m, theta_s, phi_s)) for m in range(0, l + 1)] for l in range(1, self.l_max + 1)])

    def get_sphharms_old(self): # Contains degenerate templates
        theta_s, phi_s = hp.pix2ang(self.nside, np.arange(hp.nside2npix(self.nside)))
        self.Ylm_temps = np.concatenate([[np.real(Ylm(l, m, theta_s, phi_s)) for m in range(- l + 1, l + 1)] for l in range(1, self.l_max + 1)])

    def get_exp_regions(self, nexp):
        """ Divide up ROI into exposure regions."""

        # Determine the pixels of the exposure regions
        pix_array = np.where(self.mask_roi == False)[0]
        exp_array = np.array([[pix_array[i], self.exposure[pix_array[i]]] for i in range(len(pix_array))])
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

    def load_templates(self):

        self.nfw_temp_gen = NFWTemplate(nside=self.nside)
        self.dsk_temp_gen = LorimerDiskTemplate(nside=self.nside)

        self.temp_psc = np.load(f"{self.data_dir}/template_psc_{self.ps_cat}.npy")
        self.temp_iso = np.load(f"{self.data_dir}/template_iso.npy")
        self.temp_bub = np.load(f"{self.data_dir}/template_bub.npy")
        self.temp_dsk = np.load(f"{self.data_dir}/template_dsk_z0p3.npy")
        self.temp_p6v11 = np.load(f"{self.data_dir}/template_dif.npy")
        self.temp_mO_pib = np.load(f"{self.data_dir}/template_Opi.npy")
        self.temp_mO_ics = np.load(f"{self.data_dir}/template_Oic.npy")
        self.temp_mA_pib = np.load(f"{self.data_dir}/template_Api.npy")
        self.temp_mA_ics = np.load(f"{self.data_dir}/template_Aic.npy")
        self.temp_mF_pib = np.load(f"{self.data_dir}/template_Fpi.npy")
        self.temp_mF_ics = np.load(f"{self.data_dir}/template_Fic.npy")

        self.temp_psc /= np.mean(self.temp_psc[~self.nm])
        self.temp_iso /= np.mean(self.temp_iso[~self.nm])
        self.temp_bub /= np.mean(self.temp_bub[~self.nm])
        self.temp_dsk /= np.mean(self.temp_dsk[~self.nm])
        self.temp_p6v11 /= np.mean(self.temp_p6v11[~self.nm])
        self.temp_mO_pib /= np.mean(self.temp_mO_pib[~self.nm])
        self.temp_mO_ics /= np.mean(self.temp_mO_ics[~self.nm])
        self.temp_mA_pib /= np.mean(self.temp_mA_pib[~self.nm])
        self.temp_mA_ics /= np.mean(self.temp_mA_ics[~self.nm])
        self.temp_mF_pib /= np.mean(self.temp_mF_pib[~self.nm])
        self.temp_mF_ics /= np.mean(self.temp_mF_ics[~self.nm])
                
        self.pib = []
        self.ics = []
        if "ModelO" in self.dif_names:
            self.pib.append(self.temp_mO_pib)
            self.ics.append(self.temp_mO_ics)
        if "ModelA" in self.dif_names:
            self.pib.append(self.temp_mA_pib)
            self.ics.append(self.temp_mA_ics)
        if "ModelF" in self.dif_names:
            self.pib.append(self.temp_mF_pib)
            self.ics.append(self.temp_mF_ics)
        self.pib = jnp.array(self.pib)
        self.ics = jnp.array(self.ics)
        self.n_dif = len(self.dif_names)

        blg_s = jnp.array([BulgeTemplates(template_name=n, nside_out=self.nside)() for n in self.blg_names])
        self.n_blg = len(blg_s)
        self.blg_s = blg_s / jnp.mean(blg_s[:, ~self.nm], axis=-1)[:, None]

        self.samples_expand_keys = { # used for expanding samples with dirichlet priors
            'theta_pib' : [f'theta_pib_{n}' for n in self.dif_names],
            'theta_ics' : [f'theta_ics_{n}' for n in self.dif_names],
            'theta_bulge_poiss' : [f'theta_poiss_{n}' for n in self.blg_names],
            'theta_bulge_ps' : [f'theta_ps_{n}' for n in self.blg_names],
        }

    def debug_exaggerate_exposure(self, multiplier=5):
        """Falsely exaggerate the exposure map difference while keeping the mean."""
        mean = np.mean(self.exposure)
        delta = self.exposure - mean
        self.exposure = mean + multiplier * delta
        self.get_exp_regions(self.n_exp)
        logging.warning(f'!!! DEBUG: Exposure map exaggerated by {multiplier} !!!')
    
            
    def model(self, data=None, beta=1.):
        """Main numpyro model."""

        mu = jnp.zeros_like(data)

        #=== pib with spherical harmonics, ics ===
        theta_pib = numpyro.sample("theta_pib", dist.Dirichlet(jnp.ones((self.n_dif,)) / self.n_dif))
        temp_pib = jnp.sum(theta_pib[:, None] * self.pib, 0)
        theta_ics = numpyro.sample("theta_ics", dist.Dirichlet(jnp.ones((self.n_dif,)) / self.n_dif))
        temp_ics = jnp.sum(theta_ics[:, None] * self.ics, 0)

        pib_modifier = jnp.zeros_like(data)
        for i in range(len(self.Ylm_temps)):
            Alm = numpyro.sample(f'Alm_{i}', dist.Uniform(-0.05, 0.05))
            pib_modifier += Alm * self.Ylm_temps[i]
        temp_pib = (1 + pib_modifier) * temp_pib
        temp_pib /= jnp.mean(temp_pib[~self.nm]) # re-normalize after modulation

        mu += numpyro.sample("S_pib", dist.Uniform(1e-3, 14)) * temp_pib
        mu += numpyro.sample("S_ics", dist.Uniform(1e-3, 14)) * temp_ics

        #=== other fixed diffuse templates ===
        mu += numpyro.sample("S_iso", dist.Uniform(1e-3, 5.)) * self.temp_iso
        mu += numpyro.sample("S_bub", dist.Uniform(1e-3, 5.)) * self.temp_bub
        mu += numpyro.sample("S_psc", dist.Uniform(1e-3, 5.)) * self.temp_psc

        #=== diffuse gce (blg + nfw) ===
        S_gce = numpyro.sample("S_gce", dist.Uniform(1e-5, 4.))
        f_bulge_poiss = numpyro.sample("f_bulge_poiss", dist.Uniform(0., 1.))

        theta_blg_poiss = numpyro.sample("theta_bulge_poiss", dist.Dirichlet(jnp.ones((self.n_blg,)) / self.n_blg))
        temp_blg_poiss = jnp.sum(theta_blg_poiss[:, None] * self.blg_s, 0)

        temp_nfw_poiss = self.nfw_temp_gen.get_NFW2_template(gamma=numpyro.sample("gamma_poiss", dist.Uniform(0.2, 2.)))
        temp_nfw_poiss /= jnp.mean(temp_nfw_poiss[~self.nm])

        mu += S_gce * (f_bulge_poiss * temp_blg_poiss + (1 - f_bulge_poiss) * temp_nfw_poiss)
                                            
        #=== PS gce (blg + nfw) ===
        Sps_gce = numpyro.sample("Sps_gce", dist.Uniform(1e-5, 8.))
        f_bulge_ps = numpyro.sample("f_bulge_ps", dist.Uniform(0., 1.))

        theta_blg_ps = numpyro.sample("theta_bulge_ps", dist.Dirichlet(jnp.ones((self.n_blg,)) / self.n_blg))
        temp_blg_ps = jnp.sum(theta_blg_ps[:, None] * self.blg_s, 0)

        temp_nfw_ps = self.nfw_temp_gen.get_NFW2_template(gamma=numpyro.sample("gamma_ps", dist.Uniform(0.2, 2.)))
        temp_nfw_ps /= jnp.mean(temp_nfw_ps[~self.nm])

        temp_gce_ps = f_bulge_ps * temp_blg_ps + (1 - f_bulge_ps) * temp_nfw_ps

        #=== PS disk ===
        Sps_dsk = numpyro.sample("Sps_dsk", dist.Uniform(1e-5, 8.))
        zs = numpyro.sample("zs", dist.Uniform(0.1, 2.5))
        C = numpyro.sample("C", dist.Uniform(0.05, 8.))
        temp_dsk_ps = self.dsk_temp_gen.get_template(zs=zs, C=C)
        temp_dsk_ps /= jnp.mean(temp_dsk_ps[~self.nm])
        
        #=== PS processing ===
        Sps_list = [Sps_gce, Sps_dsk]
        npt_compressed = jnp.array([temp_gce_ps, temp_dsk_ps])
        theta = []
        s_arr = jnp.logspace(-1., 2., 1000)
        for i, ps in enumerate(["gce", "dsk"]):
            n1 = numpyro.sample(f'n1_{ps}', dist.Uniform(4.0, 6.0))
            n2 = numpyro.sample(f'n2_{ps}', dist.Uniform(0.5, 1.99))
            n3 = numpyro.sample(f'n3_{ps}', dist.Uniform(-6., -5.))
            sb1 = numpyro.sample(f'sb1_{ps}', dist.Uniform(5., 40.0))
            lambda_s = numpyro.sample(f'lambdas_{ps}', dist.Uniform(0.1, 0.95))

            theta_tmp = jnp.array([1., n1, n2, n3, sb1, lambda_s * sb1])
            dnds_arr = dnds(s_arr, theta_tmp)
            A = Sps_list[i] / jnp.trapz(s_arr * dnds_arr, s_arr)
            theta.append([A, n1, n2, n3, sb1, lambda_s * sb1])
        theta = jnp.array(theta)
        
        #=== exposure regions ===
        # pad the last exposure region so that all are the same size
        exp_lens = [len(self.expreg_indices[i]) for i in range(len(self.expreg_indices))]
        n_pad = exp_lens[0] - exp_lens[-1]
        
        expreg_indices = jnp.zeros_like(self.expreg_indices)
        expreg_indices = expreg_indices.at[:-1].set(self.expreg_indices[:-1])
        expreg_indices = expreg_indices.at[-1].set(jnp.pad(self.expreg_indices[-1], (0, n_pad)))

        log_like_np_exp_vmapped = jax.vmap(log_like_np, in_axes=(0, 0, 1, 0, None, None, None, None))
                
        mu_batch = mu[~self.mask_roi][jnp.array(expreg_indices)]
        npt_compressed_batch = npt_compressed[:, ~self.mask_roi][:, jnp.array(expreg_indices)]
        data_batch = data[~self.mask_roi][jnp.array(expreg_indices)]
        exposure_multiplier = self.exposure_means_list / self.exposure_mean
        
        # scale non-Poissonian parameters (norm divided by exposure ratio, breaks multiplied)
        theta = repeat(theta, "n_ps n_param -> n_exp n_ps n_param", n_exp=len(expreg_indices))
        theta = theta.at[:, :, 0].set(theta[:, :, 0] / exposure_multiplier[:, None])
        theta = theta.at[:, :, -1].set(theta[:, :, -1] * exposure_multiplier[:, None])
        theta = theta.at[:, :, -2].set(theta[:, :, -2] * exposure_multiplier[:, None])

        #=== likelihood ===
        with numpyro.plate("data", size=len(mu[~self.mask_roi]), dim=-1):
            
            log_like_exp = log_like_np_exp_vmapped(
                theta,
                mu_batch,
                npt_compressed_batch,
                data_batch,
                self.f_ary,
                self.df_rho_ary,
                self.k_max,
                len(expreg_indices[0])
            )
            loglike = jnp.concatenate(log_like_exp)[:len(mu[~self.mask_roi])]

            with handlers.mask(mask=~jnp.logical_or(jnp.isinf(loglike), jnp.isnan(loglike))):
                with handlers.scale(scale=beta):
                    return numpyro.factor('log-likelihood', loglike)

            
    def fit_svi(
        self,
        rng_key=jax.random.PRNGKey(42),
        guide='iaf', num_flows=5, hidden_dims=[128, 128],
        n_steps=7500, lr=5e-3, num_particles=8, renyi_alpha=1,
        lr_exp_decay=False,
        tempering_schedule='none',
        **model_static_kwargs
    ):

        #=== guide ===
        iaf_kwargs = dict(num_flows=num_flows, hidden_dims=hidden_dims, nonlinearity=stax.Tanh)

        if guide == "mvn":
            self.guide = autoguide.AutoMultivariateNormal(self.model)
            
        elif guide == "iaf":
            self.guide = autoguide.AutoIAFNormal(self.model, **iaf_kwargs)
            
        elif guide == "iafm":
            class AutoIAFMixture(autoguide.AutoIAFNormal):
                def get_base_dist(self):
                    C = 8
                    mixture = dist.MixtureSameFamily(
                        dist.Categorical(probs=jnp.ones(C) / C),
                        dist.Normal(jnp.arange(float(C)), 1.)
                    )
                    return mixture.expand([self.latent_dim]).to_event()
            self.guide = AutoIAFMixture(self.model, **iaf_kwargs)

        elif guide == "iafst":
            class AutoIAFStudentT(autoguide.AutoIAFNormal):
                def get_base_dist(self):
                    # For instance, a single StudentT distribution
                    # with df=5, loc=0, scale=1 for the entire latent dimension
                    return dist.StudentT(df=5.0, loc=0.0, scale=1.0).expand([self.latent_dim]).to_event(1)
            self.guide = AutoIAFStudentT(self.model, **iaf_kwargs)
            
        else:
            raise NotImplementedError(guide)
        
        #=== optimizer ===
        if lr_exp_decay:
            lr_schedule = optax.join_schedules(
                schedules=[
                    optax.constant_schedule(lr),
                    optax.exponential_decay(
                        init_value = lr,
                        transition_steps = 100,
                        decay_rate = 0.97,
                        staircase = False
                    )
                ],
                boundaries=[2000]
            )
        else:
            lr_schedule = lr

        optimizer = optim.optax_to_numpyro(optax.chain(
            optax.clip(1.),
            optax.adam(lr_schedule),
        ))
        
        #=== loss ===
        if renyi_alpha != 1:
            loss = RenyiELBO(num_particles=num_particles, alpha=renyi_alpha)
            logging.warning(f'Using Renyi ELBO with alpha = {renyi_alpha}')
        else:
            loss = Trace_ELBO(num_particles=num_particles, vectorize_particles=True)

        #=== svi ===
        self.svi = SVI(self.model, self.guide, optimizer, loss)
        self.svi_results = run_svi_with_beta(
            self.svi,
            rng_key,
            n_steps,
            tempering_schedule=tempering_schedule,
            **model_static_kwargs
        )
        
        return self.svi_results
    
    
    def expand_samples(self, samples):
        new_samples = {}
        for k in samples.keys():
            if k in self.samples_expand_keys:
                for i in range(samples[k].shape[-1]):
                    new_samples[self.samples_expand_keys[k][i]] = samples[k][...,i]
            elif k in ['auto_shared_latent']:
                pass
            else:
                new_samples[k] = samples[k]
        return new_samples
        
        
    def get_svi_samples(self, rng_key=jax.random.PRNGKey(42), num_samples=50000, expand_samples=True):
        
        self.svi_samples = self.guide.sample_posterior(
            rng_key=rng_key,
            params=self.svi_results.params,
            sample_shape=(num_samples,)
        )
        if expand_samples:
            self.svi_samples = self.expand_samples(self.svi_samples)
            
        return self.svi_samples
    
    def get_neutra_model(self):
        """ Get model reparameterized via neural transport."""
        if self.svi_results is None:
            raise ValueError("Must run SVI before getting NeuTra model.")
        neutra = NeuTraReparam(self.guide, self.svi_results.params)
        self.model_neutra = neutra.reparam(self.model)
    
    def run_nuts(self, use_neutra=False, num_chains=4, num_warmup=500, num_samples=5000, step_size=0.1,
                 rng_key=jax.random.PRNGKey(0), **model_static_kwargs):
        
        if use_neutra:
            self.get_neutra_model()
            model = self.model_neutra
        else:
            model = self.model
        
        kernel = NUTS(model, max_tree_depth=4, dense_mass=False, step_size=step_size)
        self.nuts_mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, chain_method='vectorized')
        self.nuts_mcmc.run(rng_key, **model_static_kwargs)
        
        return self.nuts_mcmc
    
    
    def run_parallel_tempering_hmc(self, num_samples=5000, step_size_base=5e-2, num_leapfrog_steps=3, num_adaptation_steps=600, rng_key=jax.random.PRNGKey(0)):
        
        # Geometric temperatures decay
        inverse_temperatures = 0.5 ** jnp.arange(4.)

        # If everything was Normal, step_size should be ~ sqrt(temperature).
        step_size = step_size_base / jnp.sqrt(inverse_temperatures)[..., None]

        def make_kernel_fn(target_log_prob_fn):

            hmc = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size, num_leapfrog_steps=num_leapfrog_steps)

            adapted_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=hmc,
            num_adaptation_steps=num_adaptation_steps)

            return adapted_kernel
        
        self.get_neutra_model()
        
        kernel = ReplicaExchangeMC(self.model_neutra, inverse_temperatures=inverse_temperatures, make_kernel_fn=make_kernel_fn)
        self.mcmc = MCMC(kernel, num_warmup=num_adaptation_steps, num_samples=num_samples, num_chains=1, chain_method='vectorized')
        self.mcmc.run(rng_key, self.data)
        
        return self.mcmc
    

    def get_MAP_estimates(self, rng_key=jax.random.PRNGKey(42), lr=0.1, n_steps=30000, **model_static_kwargs):
        
        guide = autoguide.AutoDelta(self.model)
        optimizer = optim.optax_to_numpyro(optax.chain(optax.clip(1.), optax.adamw(lr)))
        svi = SVI(self.model, guide, optimizer, loss=Trace_ELBO())
        svi_results = svi.run(rng_key, n_steps, **model_static_kwargs)
        self.MAP_estimates = guide.median(svi_results.params)
        
        return svi_results


    def simulate(self, vd, modifiers=[], rng_seed=None):
        """ Simulate a map based on model templates.

        Args:
            vd (dict): Dictionary of truth parameters.
            modifiers (list of str): From ['deltapsf', 'flatexp', 'p6v11'].
                'deltapsf': Assumes PSF is trivial.
                'flatexp': Uses a flat exposure map.
                'p6v11': Uses the p6v11 diffuse template instead of the pib and ics templates.
            rng_seed (int): Random seed for deterministic testing. If None, no seeding is done.

        Return:
            np.ndarray: Simulated counts map.
        """

        if rng_seed is not None:
            np.random.seed(rng_seed)

        # poiss: nfw iso bub psc pib*3 ics*3 blg*5
        temp_nfw_poiss = self.nfw_temp_gen.get_NFW2_template(gamma=vd['gamma_poiss'])
        temp_nfw_poiss /= np.mean(temp_nfw_poiss[~self.nm])

        temps_poiss = [temp_nfw_poiss, self.temp_iso, self.temp_bub, self.temp_psc]
        theta = [vd['S_gce'] * (1 - vd['f_bulge_poiss']), vd['S_iso'], vd['S_bub'], vd['S_psc']]
        temps_poiss += list(self.blg_s)
        theta += list(vd['S_gce'] * vd['f_bulge_poiss'] * np.array(vd['theta_bulge_poiss']))

        if 'p6v11' in modifiers:
            temps_poiss.append(self.temp_p6v11)
            theta.append(vd['S_p6v11'])
        else:
            temps_poiss += list(self.pib)
            temps_poiss += list(self.ics)
            theta += list(vd['S_pib'] * np.array(vd['theta_pib']))
            theta += list(vd['S_ics'] * np.array(vd['theta_ics']))

        # ps: nfw+blg*5 dsk
        # temp_ps
        temp_nfw_ps = self.nfw_temp_gen.get_NFW2_template(gamma=vd['gamma_ps'])
        temp_nfw_ps /= np.mean(temp_nfw_ps[~self.nm])
        temp_blg_ps = np.einsum('i,ij->j', vd['theta_bulge_ps'], self.blg_s)
        temp_blg_ps /= np.mean(temp_blg_ps[~self.nm])
        temp_ps_gce = (1 - vd['f_bulge_ps']) * temp_nfw_ps + vd['f_bulge_ps'] * temp_blg_ps
        temp_ps_dsk = self.dsk_temp_gen.get_template(zs=vd['zs'], C=vd['C'])
        temp_ps_dsk /= np.mean(temp_ps_dsk[~self.nm])

        temps_ps = []
        if vd['Sps_gce'] > 0:
            temps_ps.append(np.array(temp_ps_gce))
            # theta[0] is the expected photon count per pixel in normalization mask region
            theta += [vd['Sps_gce'], vd['n1_gce'], vd['n2_gce'], vd['n3_gce'], vd['sb1_gce'], vd['lambdas_gce'] * vd['sb1_gce']]
        if vd['Sps_dsk'] > 0:
            temps_ps.append(np.array(temp_ps_dsk))
            theta += [vd['Sps_dsk'], vd['n1_dsk'], vd['n2_dsk'], vd['n3_dsk'], vd['sb1_dsk'], vd['lambdas_dsk'] * vd['sb1_dsk']]

        kp = KingPSF()

        if 'deltapsf' in modifiers:
            psf_scheme = 'true delta'
        else:
            psf_scheme = 'original'

        if 'flatexp' in modifiers:
            exp_map = np.ones_like(self.exposure) * np.mean(self.exposure)
        else:
            exp_map = self.exposure

        return simulator(
            theta, temps_poiss, temps_ps,
            mask_norm = self.nm,
            mask_sim = self.nm,
            psf_r_func = lambda r: kp.psf_fermi_r(r),
            exp_map = exp_map,
            psf_scheme = psf_scheme,
            sim1b = False
        )