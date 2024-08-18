import os
import numpyro
import healpy as hp
import numpy as np
import jax.numpy as jnp
import jax
from jax.example_libraries import stax

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

from models.scd import dnds
from models.templates import NFWTemplate, LorimerDiskTemplate
from models.bulge_models import BulgeTemplates
from likelihoods.npll_jax import log_like_np
from likelihoods.pll_jax import log_like_poisson
from utils.sph_harm import Ylm
from utils import create_mask as cm
from models.psf import KingPSF
from utils.psf_correction import PSFCorrection

import logging


wdir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(wdir, '../data')


class NPModel:
    """
    Parameters
    ----------
    ...
    non_poissonian : bool
        Whether to use non-poissonian template fitting.
    bulge_hybrid : bool
        If False, use the first template in bulge_template_names. If True, use a
        hybrid of the templates in bulge_template_names.
    ps_cat : {'3fgl', '4fgl'}
        Point source catalog to use for masks.
    band_mask_range : float
        |b| value [deg] below which the galactic plane is masked. Affects
        self.mask_roi .
        
    Attributes
    ----------
    ...
    normalization_mask: mask used to normalize templates.
    """
    def __init__(
        self, non_poissonian=True, l_max=2,
        dif_names=["ModelO", "ModelA", "ModelF"],
        bulge_hybrid=True,
        bulge_template_names=["mcdermott2022", "mcdermott2022_bbp", "mcdermott2022_x", "macias2019", "coleman2019"],
        vary_gamma=True,
        vary_disk=True,
        ps_cat="3fgl", r_outer=25, band_mask_range=2.,
        nside=128, n_exp=1,
        use_flat_exposure=False, # TMP
        debug_model=False,
        data=None,
    ):
        
        #========== General ==========
        self.nside = nside
        self.ps_cat = ps_cat
        self.non_poissonian = non_poissonian
        
        self.data_dir = f"{data_dir}/fermi_data_573w/fermi_data_{self.nside}"
        if data is None:
            self.data = jnp.array(np.load("{}/fermidata_counts.npy".format(self.data_dir)).astype(np.int32))
        else:
            self.data = data
        self.exposure_map = np.load("{}/fermidata_exposure.npy".format(self.data_dir))
        self.use_flat_exposure = use_flat_exposure
        if self.use_flat_exposure:
            self.exposure_map = np.full_like(self.exposure_map, np.mean(self.exposure_map))
    
        #========== Mask ==========
        if ps_cat == "3fgl":
            # mask_ps = np.load("{}/fermidata_pscmask_{}.npy".format(self.data_dir, self.ps_cat)) == 1
            mask_ps = hp.ud_grade(np.load(f"{data_dir}/mask_3fgl_0p8deg.npy"), nside_out=self.nside) > 0
        elif ps_cat == "4fgl":
            if non_poissonian:
                logging.warning('Using 4fgl with non-poissonian fit.')
            mask_ps = hp.ud_grade(np.load(f"{data_dir}/fermi_data_573w/fermi_data_{nside}/fermidata_pscmask_4fgl.npy"), nside_out=self.nside) > 0
        else:
            raise NotImplementedError("Other catalogs not supported at the moment.")
            
        self.mask_roi = cm.make_mask_total(nside=self.nside, band_mask=True, band_mask_range=band_mask_range, mask_ring=True, inner=0, outer=r_outer, custom_mask=mask_ps)
        self.mask_plane = cm.make_mask_total(nside=self.nside, band_mask=True, band_mask_range=2., mask_ring=True, inner=0, outer=25,)
        self.normalization_mask = self.mask_plane

        print(f'Number of pixels in ROI: {np.sum(~self.mask_roi)}')

        #========== Templates ==========
        self.vary_gamma = vary_gamma
        self.nfw_template = NFWTemplate(nside=self.nside)
        if self.non_poissonian:
            self.vary_disk = vary_disk
            self.disk_template = LorimerDiskTemplate(nside=self.nside)
        
        self.dif_names = dif_names
        
        # Load all bulge templates
        self.blg_names = bulge_template_names
        self.bulge_hybrid = bulge_hybrid
        self.bulge_templates = jnp.array([BulgeTemplates(template_name=template_name, nside_out=nside)() for template_name in bulge_template_names])
        if self.bulge_hybrid:
            self.n_bulge_templates = len(self.bulge_templates)
        else:
            self.n_bulge_templates = 1
        # Individually normalize the bulge templates
        self.bulge_templates = self.bulge_templates / jnp.mean(self.bulge_templates[:, ~self.normalization_mask], axis=-1)[:, None]
        
        self.load_templates()

        #========== Spherical harmonics ==========
        self.l_max = l_max
        self.get_sphharms()
        
        #========== NPTF ==========
        if self.non_poissonian:
            self.get_psf_correction()
            self.k_max = np.max(np.array(self.data)[~self.mask_roi])
            print("Max photon count is {}".format(self.k_max))
        
        self.get_exp_regions(n_exp)
        
        #========== sample expand keys ==========
        self.samples_expand_keys = {
            'theta_pib' : [f'theta_pib_{n}' for n in self.dif_names],
            'theta_ics' : [f'theta_ics_{n}' for n in self.dif_names],
            'theta_bulge_poiss' : [f'theta_poiss_{n}' for n in self.blg_names],
            'theta_bulge_ps' : [f'theta_ps_{n}' for n in self.blg_names],
        }
        
        #========== Other ==========
        self.debug_model = debug_model
        
        
    def get_psf_correction(self):

        #psf_tags = ['king', 'old']
        psf_tags = ['delta']
        print(f'Using PSF tags: {psf_tags}')

        if 'king' in psf_tags:
            kp = KingPSF()
            if 'old' in psf_tags:
                pc_inst = PSFCorrection(delay_compute=True, num_f_bins=15, nside=self.nside, f_trunc=0.01)
                print('!!! USING OLD F BIN !!!')
            elif 'new' in psf_tags:
                # new nonuniform f binning
                pc_inst = PSFCorrection(
                    delay_compute=True, num_f_bins='nonuni', nside=self.nside, f_trunc=0.00,
                    n_psf=100000
                )
                print('!!! USING NEW F BIN: NON-UNIFORM !!!')
            else:
                raise ValueError(psf_tags)
            pc_inst.psf_r_func = lambda r: kp.psf_fermi_r(r)
            pc_inst.sample_psf_max = 10.0 * kp.spe * (kp.score + kp.stail) / 2.0
            pc_inst.psf_samples = 10000
            pc_inst.psf_tag = f"Fermi_PSF_2GeV2_nside{self.nside}"
            pc_inst.make_or_load_psf_corr(force_recompute=True)

        elif 'delta' in psf_tags:
            pc_inst = PSFCorrection(
                delay_compute=True, num_f_bins='nonuni', nside=self.nside, f_trunc=0.00,
                psf_sigma_deg=1e-6,
                n_psf=100000
            )
            pc_inst.sample_psf_max = 1e-6
            pc_inst.psf_samples = 10000
            pc_inst.psf_tag = f"Delta_PSF_2GeV2_nside{self.nside}"
            pc_inst.make_or_load_psf_corr(force_recompute=True)

            if 'mod0count' in psf_tags:
                npix_before = np.sum(pc_inst.df_rho_div_f_ary[1:] * pc_inst.f_ary[1:])
                npix0 = 6839 - npix_before
                pc_inst.df_rho_div_f_ary[0] = npix0 / pc_inst.f_ary[0]
                pc_inst.df_rho_div_f_ary /= np.sum(pc_inst.df_rho_div_f_ary * pc_inst.f_ary**2)

        else:
            raise ValueError(psf_tags)

        self.f_ary = pc_inst.f_ary
        self.df_rho_div_f_ary = pc_inst.df_rho_div_f_ary

    def get_sphharms(self):
        
        npix = hp.nside2npix(self.nside)

        theta_ary, phi_ary = hp.pix2ang(self.nside, np.arange(npix))
        Ylm_list = [[np.real(Ylm(l, m, theta_ary, phi_ary)) for m in range(-l + 1, l + 1)] for l in range(1, self.l_max + 1)]
        self.Ylm_temps = np.array([item for sublist in Ylm_list for item in sublist])

    def load_templates(self):

        self.temp_psc = np.load("{}/template_psc_{}.npy".format(self.data_dir, self.ps_cat))
        self.temp_iso = np.load("{}/template_iso.npy".format(self.data_dir))
        self.temp_bub = np.load("{}/template_bub.npy".format(self.data_dir))
        self.temp_dsk = np.load("{}/template_dsk_z0p3.npy".format(self.data_dir))

        # Load Model O templates
        self.temp_mO_pib = np.load("{}/template_Opi.npy".format(self.data_dir))
        self.temp_mO_ics = np.load("{}/template_Oic.npy".format(self.data_dir))

        # Load Model A templates
        self.temp_mA_pib = np.load("{}/template_Api.npy".format(self.data_dir))
        self.temp_mA_ics = np.load("{}/template_Aic.npy".format(self.data_dir))

        # Load Model F templates
        self.temp_mF_pib = np.load("{}/template_Fpi.npy".format(self.data_dir))
        self.temp_mF_ics = np.load("{}/template_Fic.npy".format(self.data_dir))
                
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
        
        self.n_dif_templates = len(self.dif_names)
        
        self.svi = None
        self.svi_init_state = None
    
            
    def model(self, data=...):
        
        # Get mixed pib template
        theta_pib = numpyro.sample("theta_pib", dist.Dirichlet(jnp.ones((self.n_dif_templates,)) / self.n_dif_templates))
        temp_pib = jnp.sum(theta_pib[:, None] * self.pib, 0)

        # Get mixed ics template
        theta_ics = numpyro.sample("theta_ics", dist.Dirichlet(jnp.ones((self.n_dif_templates,)) / self.n_dif_templates))
        temp_ics = jnp.sum(theta_ics[:, None] * self.ics, 0)

        S_gce = numpyro.sample("S_gce", dist.Uniform(1e-5, 4.))
            
        temps = [self.temp_iso, self.temp_bub, self.temp_psc, temp_pib, temp_ics]
        temp_labels = ["iso", "bub", "psc", "pib", "ics"]
                
        mu = jnp.zeros_like(data)
        
        for temp, temp_label in zip(temps, temp_labels):
            
            if temp_label in ["pib", "ics"]:
                prior_lo, prior_hi = 1e-3, 14.
            else:
                prior_lo, prior_hi = 1e-3, 5.0
                
            prior_dist = dist.Uniform(prior_lo, prior_hi)
            S_temp = numpyro.sample("S_{}".format(temp_label), prior_dist)
            
            if temp_label in ["pib"]:
                
                temp_pib_mod = jnp.zeros_like(data)
                for ii in range(len(self.Ylm_temps)):
                    Alm = numpyro.sample("Alm_{}".format(ii), dist.Uniform(-0.05, 0.05))
                    temp_pib_mod += Alm * self.Ylm_temps[ii]
                
                temp_pib_mod = (1. + temp_pib_mod) * temp
                
                A_temp = S_temp / jnp.mean(temp_pib_mod[~self.normalization_mask])
                mu += A_temp * temp_pib_mod  
            else:
                A_temp = S_temp / jnp.mean(temp[~self.normalization_mask])
                mu += A_temp * temp     
                                            
        if self.vary_gamma:
            gamma_ps = numpyro.sample("gamma_ps", dist.Uniform(0.2, 2.)) if self.non_poissonian else None
            gamma_poiss = numpyro.sample("gamma_poiss", dist.Uniform(0.2, 2.))
        else:
            gamma_ps = 1.2 if self.non_poissonian else None
            gamma_poiss = 1.2

        temp_gce_nfw_ps = self.nfw_template.get_NFW2_template(gamma=gamma_ps) if self.non_poissonian else None
        temp_gce_nfw_poiss = self.nfw_template.get_NFW2_template(gamma=gamma_poiss)
            
        if self.non_poissonian:
            if self.vary_disk:
                zs = numpyro.sample("zs", dist.Uniform(0.1, 2.5))
                C = numpyro.sample("C", dist.Uniform(0.05, 8.))
                temp_dsk = self.disk_template.get_template(zs=zs, C=C)
            else:
                temp_dsk = self.temp_dsk
                
        if self.bulge_hybrid:
            f_bulge_ps = numpyro.sample("f_bulge_ps", dist.Uniform(0., 1.)) if self.non_poissonian else None
            f_bulge_poiss = numpyro.sample("f_bulge_poiss", dist.Uniform(0., 1.))
            
            theta_bulge_poiss = numpyro.sample("theta_bulge_poiss", dist.Dirichlet(jnp.ones((self.n_bulge_templates,)) / self.n_bulge_templates))
            temp_bulge = jnp.sum(theta_bulge_poiss[:, None] * self.bulge_templates, 0)
        else:
            f_bulge_ps = 0. if self.non_poissonian else None
            f_bulge_poiss = numpyro.sample("f_bulge_poiss", dist.Uniform(0., 1.))
            temp_bulge = self.bulge_templates[0]
        
        # Normalize to same mean
        A_gce_nfw = S_gce / jnp.mean(temp_gce_nfw_poiss[~self.normalization_mask])
        A_gce_bulge = S_gce / jnp.mean(temp_bulge[~self.normalization_mask])
        temp_gce_poiss = (1 - f_bulge_poiss) * A_gce_nfw * temp_gce_nfw_poiss \
                            + f_bulge_poiss * A_gce_bulge * temp_bulge
        
        A_gce = S_gce / jnp.mean(temp_gce_poiss[~self.normalization_mask])
        mu += A_gce * temp_gce_poiss
        
        if self.non_poissonian:
            # Get mixed bulge template
            theta_bulge_ps = numpyro.sample("theta_bulge_ps", dist.Dirichlet(jnp.ones((self.n_bulge_templates,)) / self.n_bulge_templates))
            temp_bulge = jnp.sum(theta_bulge_ps[:, None] * self.bulge_templates, 0)

            # Normalize to same mean
            A_gce_nfw = 1 / jnp.mean(temp_gce_nfw_ps[~self.normalization_mask])
            A_gce_bulge = 1 / jnp.mean(temp_bulge[~self.normalization_mask])

            # Get hybrid template
            temp_gce_ps = (1 - f_bulge_ps) * A_gce_nfw * temp_gce_nfw_ps + f_bulge_ps * A_gce_bulge * temp_bulge

            npt_compressed = jnp.array([temp_gce_ps, temp_dsk])

            theta = []    

            for ips, ps in enumerate(["gce", "dsk"]):

                Sps = numpyro.sample("Sps_{}".format(ps), dist.Uniform(1e-3, 4.))

                n1 = numpyro.sample("n1_{}".format(ps), dist.Uniform(4.0, 6.0))
                n2 = numpyro.sample("n2_{}".format(ps), dist.Uniform(0.5, 1.99))
                n3 = numpyro.sample("n3_{}".format(ps), dist.Uniform(-6., -5.))
                sb1 = numpyro.sample("sb1_{}".format(ps), dist.Uniform(5., 40.0))
                lambda_s = numpyro.sample("lambdas_{}".format(ps), dist.Uniform(0.1, 0.95))

                theta_tmp = jnp.array([1., n1, n2, n3, sb1, lambda_s * sb1])

                s_ary = jnp.logspace(-1., 2., 100)
                dnds_ary = dnds(s_ary, theta_tmp)

                A = Sps / jnp.mean(npt_compressed[ips][~self.normalization_mask] * jnp.trapz(s_ary * dnds_ary, s_ary))

                theta.append([A, n1, n2, n3, sb1, lambda_s * sb1])

            theta = jnp.array(theta)
                
        # Pad the last exposure region so that all are the same size
        exp_lens = [len(self.expreg_indices[i]) for i in range(len(self.expreg_indices))]
        n_pad = exp_lens[0] - exp_lens[-1]
        
        expreg_indices = jnp.zeros_like(self.expreg_indices)
        expreg_indices = expreg_indices.at[:-1].set(self.expreg_indices[:-1])
        expreg_indices = expreg_indices.at[-1].set(jnp.pad(self.expreg_indices[-1], (0, n_pad)))

        if self.non_poissonian:
            log_like_np_exp_vmapped = jax.vmap(log_like_np, in_axes=(0, 0, 1, 0, None, None, None, None))
        else:
            log_like_poisson_exp_vmapped = jax.vmap(log_like_poisson, in_axes=(0, 0))
                
        # Get relevant arrays for different exposure regions
        mu_batch = mu[~self.mask_roi][jnp.array(expreg_indices)]
        if self.non_poissonian:
            npt_compressed_batch = npt_compressed[:, ~self.mask_roi][:, jnp.array(expreg_indices)]
        data_batch = data[~self.mask_roi][jnp.array(expreg_indices)]
        
        exposure_multiplier = self.exposure_means_list / self.exposure_mean
        
        # Scale non-Poissonian parameters (norm divided by exposure ratio, breaks multiplied)
        if self.non_poissonian:
            theta = repeat(theta, "n_ps n_param -> n_exp n_ps n_param", n_exp=len(expreg_indices))
            theta = theta.at[:, :, 0].set(theta[:, :, 0] / exposure_multiplier[:, None])
            theta = theta.at[:, :, -1].set(theta[:, :, -1] * exposure_multiplier[:, None])
            theta = theta.at[:, :, -2].set(theta[:, :, -2] * exposure_multiplier[:, None])
        
        with numpyro.plate("data", size=len(mu[~self.mask_roi]), dim=-1):
            
            if self.non_poissonian:
                log_like_exp = log_like_np_exp_vmapped(theta, mu_batch, npt_compressed_batch, data_batch, self.f_ary, self.df_rho_div_f_ary, self.k_max, len(expreg_indices[0]))
            else:
                log_like_exp = log_like_poisson_exp_vmapped(mu_batch, data_batch)
            
            # Concatenate exposure regions
            loglike = jnp.concatenate(log_like_exp)[:len(mu[~self.mask_roi])]
                                
            with handlers.mask(mask=~jnp.logical_or(jnp.isinf(loglike), jnp.isnan(loglike))):
                return numpyro.factor('log-likelihood', loglike)
        

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
            
            
    def fit_svi(
        self, rng_key=jax.random.PRNGKey(42),
        guide='iaf', num_flows=5, hidden_dims=[128, 128],
        n_steps=7500, lr=1e-4, num_particles=8, vectorize_particles=True,
        **model_static_kwargs
    ):

        iaf_kwargs = dict(num_flows=num_flows, hidden_dims=hidden_dims, nonlinearity=stax.Tanh)

        if guide == "mvn":
            self.guide = autoguide.AutoMultivariateNormal(self.model)
            
        elif guide == "iaf":
            self.guide = autoguide.AutoIAFNormal(self.model, **iaf_kwargs)
            
        # elif guide == "iaf_mixture":
        #     class AutoIAFMixture(autoguide.AutoIAFNormal):
        #         def get_base_dist(self):
        #             C = num_base_mixture
        #             mixture = dist.MixtureSameFamily(
        #                 dist.Categorical(probs=jnp.ones(C) / C),
        #                 dist.Normal(jnp.arange(float(C)), 1.)
        #             )
        #             return mixture.expand([self.latent_dim]).to_event()
        #     self.guide = AutoIAFMixture(self.model, **iaf_kwargs)
            
        elif guide == "iaf_gaussians":
            class AutoIAFMultiGaussian(autoguide.AutoIAFNormal):
                def get_base_dist(self):
                    return dist.Normal(
                        jnp.array([-5, -2, -1, 0, 1, 2, 5, 20], dtype=jnp.float32),
                        1.,
                    )
            self.guide = AutoIAFMultiGaussian(self.model, **iaf_kwargs)
            
        else:
            raise NotImplementedError

        optimizer = optim.optax_to_numpyro(optax.chain(
            optax.clip(1.),
            optax.adam(lr),
        ))

        self.svi = SVI(self.model, self.guide, optimizer, Trace_ELBO(num_particles=num_particles, vectorize_particles=vectorize_particles))
        self.svi_results = self.svi.run(rng_key, n_steps, **model_static_kwargs)
        
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
        """ Get model reparameterized via neural transport """
        neutra = NeuTraReparam(self.guide, self.svi_results.params)
        self.model_neutra = neutra.reparam(self.model)
    
    def run_nuts(self, num_chains=4, num_warmup=500, num_samples=5000, step_size=0.1,
                 rng_key=jax.random.PRNGKey(0), use_neutra=True, **model_static_kwargs):
        
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
    
    
    def get_MAP_estimates(self, rng_key=jax.random.PRNGKey(42), lr=0.1, n_steps=30000):
        
        #optimizer = numpyro.optim.Adam(lr=lr)
        guide = autoguide.AutoDelta(self.model)
        optimizer = optim.optax_to_numpyro(optax.chain(optax.clip(1.), optax.adamw(lr)))
        svi = SVI(self.model, guide, optimizer, loss=Trace_ELBO())
        svi_results = svi.run(rng_key, n_steps, self.data)
        self.MAP_estimates = guide.median(svi_results.params)
        
        return svi_results
