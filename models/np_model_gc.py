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
from likelihoods.npll_jax_new import log_like_np
from likelihoods.pll_jax import log_like_poisson
from utils.sph_harm import Ylm
from utils import create_mask as cm
from models.psf import KingPSF
from utils.psf_correction import PSFCorrection

import logging


wdir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(wdir, '../data')


class NPModelGC11:
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
        # bulge_hybrid=True,
        bulge_template_names=["mcdermott2022", "mcdermott2022_bbp", "mcdermott2022_x", "macias2019", "coleman2019"],
        # vary_gamma=True,
        # vary_disk=True,
        ps_cat="3fgl", r_outer=25, band_mask_range=2.,
        nside=128, n_exp=1,
        use_flat_exposure=False,
        data=None,
        psf_tags=None,
    ):
        
        #========== General ==========
        self.nside = nside
        self.ps_cat = ps_cat
        self.non_poissonian = non_poissonian
        
        self.data_dir = f"{data_dir}/fermi_data_573w/fermi_data_{self.nside}"
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
        self.nm = self.normalization_mask
        self.npixROI = int(np.sum(~self.mask_roi))
        print(f'Number of pixels in ROI: {np.sum(~self.mask_roi)}')

        #========== Templates ==========
        self.nfw_template = NFWTemplate(nside=self.nside)
        # self.temp_p_nfw_fixed = self.nfw_template.get_NFW2_template(gamma=1.0)
        # self.temp_ps_nfw_fixed = self.nfw_template.get_NFW2_template(gamma=1.2)
        self.disk_template = LorimerDiskTemplate(nside=self.nside)
        # self.temp_p_dsk_fixed = self.disk_template.get_template(zs=0.6, C=6.)
        
        self.blg_names = bulge_template_names
        self.blg = jnp.array([BulgeTemplates(template_name=template_name, nside_out=nside)() for template_name in bulge_template_names])
        self.n_blg = len(self.blg)
        self.blg = self.blg / jnp.mean(self.blg[:, ~self.nm], axis=-1)[:, None]
        
        self.dif_names = dif_names
        self.load_templates()

        #========== Spherical harmonics ==========
        self.l_max = l_max
        self.get_sphharms()
        
        #========== NPTF ==========
        self.psf_tags = psf_tags
        self.get_psf_correction()
        self.k_max = np.max(np.array(self.data)[~self.mask_roi])
        print("Max photon count is {}".format(self.k_max))
        # self.get_exp_regions(n_exp)
        
        #========== sample expand keys ==========
        self.samples_expand_keys = {
            'theta_pib' : [f'theta_pib_{n}' for n in self.dif_names],
            'theta_ics' : [f'theta_ics_{n}' for n in self.dif_names],
            'theta_blg' : [f'theta_blg_{n}' for n in self.blg_names],
            'theta_blg_ps' : [f'theta_blg_ps_{n}' for n in self.blg_names],
        }
        self.svi = None
        self.svi_init_state = None
        
        
    def get_psf_correction(self):
        """
        ['king', 'old']
        ['king', 'new']
        ['delta']
        ['deltasimple']
        """

        psf_tags = self.psf_tags
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

            # if 'analytic' in psf_tags:
            #     pc_inst.f_ary[-1] = 1.
            #     pc_inst.df_rho_ary *= 0.
            #     pc_inst.df_rho_ary[-1] = 1./pc_inst.f_ary[-1]
            #     pc_inst.df_rho_ary[0] = 0. # does not matter

        elif 'deltasimple' in psf_tags:
            self.f_ary = jnp.array([0., 1.])
            self.df_rho_ary = jnp.array([0., 1.])
            return

        else:
            raise ValueError(psf_tags)

        self.f_ary = pc_inst.f_ary
        self.df_rho_ary = pc_inst.df_rho_ary

    def get_sphharms(self):
        
        npix = hp.nside2npix(self.nside)

        theta_ary, phi_ary = hp.pix2ang(self.nside, np.arange(npix))
        Ylm_list = [[np.real(Ylm(l, m, theta_ary, phi_ary)) for m in range(-l + 1, l + 1)] for l in range(1, self.l_max + 1)]
        self.Ylm_temps = np.array([item for sublist in Ylm_list for item in sublist])

    def load_templates(self):

        self.temp_psc = np.load("{}/template_psc_{}.npy".format(self.data_dir, self.ps_cat))
        self.temp_iso = np.load("{}/template_iso.npy".format(self.data_dir))
        self.temp_bub = np.load("{}/template_bub.npy".format(self.data_dir))
        self.temp_pcs = self.temp_psc / np.mean(self.temp_psc[~self.nm])
        self.temp_iso = self.temp_iso / np.mean(self.temp_iso[~self.nm])
        self.temp_bub = self.temp_bub / np.mean(self.temp_bub[~self.nm])
        # self.temp_dsk = np.load("{}/template_dsk_z0p3.npy".format(self.data_dir))

        self.temp_mO_pib = np.load("{}/template_Opi.npy".format(self.data_dir))
        self.temp_mO_ics = np.load("{}/template_Oic.npy".format(self.data_dir))
        self.temp_mA_pib = np.load("{}/template_Api.npy".format(self.data_dir))
        self.temp_mA_ics = np.load("{}/template_Aic.npy".format(self.data_dir))
        self.temp_mF_pib = np.load("{}/template_Fpi.npy".format(self.data_dir))
        self.temp_mF_ics = np.load("{}/template_Fic.npy".format(self.data_dir))
                
        self.pib = []
        self.ics = []
        
        if "ModelO" in self.dif_names:
            self.pib.append(self.temp_mO_pib / np.mean(self.temp_mO_pib[~self.nm]))
            self.ics.append(self.temp_mO_ics / np.mean(self.temp_mO_ics[~self.nm]))
        if "ModelA" in self.dif_names:
            self.pib.append(self.temp_mA_pib / np.mean(self.temp_mA_pib[~self.nm]))
            self.ics.append(self.temp_mA_ics / np.mean(self.temp_mA_ics[~self.nm]))
        if "ModelF" in self.dif_names:
            self.pib.append(self.temp_mF_pib / np.mean(self.temp_mF_pib[~self.nm]))
            self.ics.append(self.temp_mF_ics / np.mean(self.temp_mF_ics[~self.nm]))
            
        self.pib = jnp.array(self.pib)
        self.ics = jnp.array(self.ics)
        
        self.n_dif = len(self.dif_names)

    def model(self, data=...):
        
        mu = jnp.zeros_like(data)

        # poissonian
        S_pib = numpyro.sample("S_pib", dist.Uniform(1e-3, 14.))
        S_ics = numpyro.sample("S_ics", dist.Uniform(1e-3, 14.))
        S_bub = numpyro.sample("S_bub", dist.Uniform(1e-3, 5.))
        S_nfw = numpyro.sample("S_nfw", dist.Uniform(1e-3, 5.))
        S_dsk = numpyro.sample("S_dsk", dist.Uniform(1e-3, 5.))

        nm = self.normalization_mask
        mu += S_pib * self.pib[0] / jnp.mean(self.pib[0][~nm])
        mu += S_ics * self.ics[0] / jnp.mean(self.ics[0][~nm])
        mu += S_bub * self.temp_bub / jnp.mean(self.temp_bub[~nm])
        mu += S_dsk * self.temp_p_dsk_fixed / jnp.mean(self.temp_p_dsk_fixed[~nm])
        mu += S_nfw * self.temp_p_nfw_fixed / jnp.mean(self.temp_p_nfw_fixed[~nm])

        # non-poissonian
        temp_ps_nfw = self.temp_ps_nfw_fixed[~self.mask_roi] / jnp.mean(self.temp_ps_nfw_fixed[~nm])
        npt_compressed = jnp.array([temp_ps_nfw])

        theta = []
        for ps in ["nfw"]:
            Sps = numpyro.sample("Sps_{}".format(ps), dist.Uniform(1e-3, 4.))
            n1 = numpyro.sample("n1_{}".format(ps), dist.Uniform(4.0, 6.0))
            n2 = numpyro.sample("n2_{}".format(ps), dist.Uniform(0.5, 1.99))
            n3 = numpyro.sample("n3_{}".format(ps), dist.Uniform(-6., -5.))
            sb1 = numpyro.sample("sb1_{}".format(ps), dist.Uniform(5., 40.0))
            lambda_s = numpyro.sample("lambdas_{}".format(ps), dist.Uniform(0.1, 0.95))

            theta_tmp = jnp.array([1., n1, n2, n3, sb1, lambda_s * sb1])
            s_ary = jnp.logspace(-1., 2., 1000)
            dnds_ary = dnds(s_ary, theta_tmp)
            num_photon_for_unit_theta0 = jnp.trapz(s_ary * dnds_ary, s_ary)
            theta.append([Sps / num_photon_for_unit_theta0, n1, n2, n3, sb1, lambda_s * sb1])
        theta = jnp.array(theta)
        
        ll = log_like_np(
            theta=theta,
            pt_sum_compressed=mu[~self.mask_roi],
            npt_compressed=npt_compressed,
            data=data[~self.mask_roi],
            f_ary=self.f_ary,
            df_rho_div_f_ary=self.df_rho_div_f_ary,
            k_max=self.k_max,
            npixROI=self.npixROI,
        )
        with numpyro.plate('data', self.npixROI):
            return numpyro.factor('ll', ll)
        

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


class NPModelGC17 (NPModelGC11):

    def model(self, data=...):
        
        mu = jnp.zeros_like(data)

        # poissonian
        S_pib = numpyro.sample("S_pib", dist.Uniform(1e-3, 14.))
        S_ics = numpyro.sample("S_ics", dist.Uniform(1e-3, 14.))
        S_bub = numpyro.sample("S_bub", dist.Uniform(1e-3, 5.))
        S_nfw = numpyro.sample("S_nfw", dist.Uniform(1e-3, 5.))
        S_dsk = numpyro.sample("S_dsk", dist.Uniform(1e-3, 5.))

        nm = self.normalization_mask
        mu += S_pib * self.pib[0] / jnp.mean(self.pib[0][~nm])
        mu += S_ics * self.ics[0] / jnp.mean(self.ics[0][~nm])
        mu += S_bub * self.temp_bub / jnp.mean(self.temp_bub[~nm])
        mu += S_dsk * self.temp_p_dsk_fixed / jnp.mean(self.temp_p_dsk_fixed[~nm])
        mu += S_nfw * self.temp_p_nfw_fixed / jnp.mean(self.temp_p_nfw_fixed[~nm])

        # non-poissonian
        temp_ps_nfw = self.temp_ps_nfw_fixed[~self.mask_roi] / jnp.mean(self.temp_ps_nfw_fixed[~nm])
        temp_ps_dsk = self.temp_p_dsk_fixed[~self.mask_roi] / jnp.mean(self.temp_p_dsk_fixed[~nm]) # same template as poissonian
        npt_compressed = jnp.array([temp_ps_nfw, temp_ps_dsk])

        theta = []
        for ps in ["nfw", "dsk"]:
            Sps = numpyro.sample("Sps_{}".format(ps), dist.Uniform(1e-3, 4.))
            n1 = numpyro.sample("n1_{}".format(ps), dist.Uniform(4.0, 6.0))
            n2 = numpyro.sample("n2_{}".format(ps), dist.Uniform(0.5, 1.99))
            n3 = numpyro.sample("n3_{}".format(ps), dist.Uniform(-6., -5.))
            sb1 = numpyro.sample("sb1_{}".format(ps), dist.Uniform(5., 40.0))
            lambda_s = numpyro.sample("lambdas_{}".format(ps), dist.Uniform(0.1, 0.95))

            theta_tmp = jnp.array([1., n1, n2, n3, sb1, lambda_s * sb1])
            s_ary = jnp.logspace(-1., 2., 1000)
            dnds_ary = dnds(s_ary, theta_tmp)
            num_photon_for_unit_theta0 = jnp.trapz(s_ary * dnds_ary, s_ary)
            theta.append([Sps / num_photon_for_unit_theta0, n1, n2, n3, sb1, lambda_s * sb1])
        theta = jnp.array(theta)
        
        ll = log_like_np(
            theta=theta,
            pt_sum_compressed=mu[~self.mask_roi],
            npt_compressed=npt_compressed,
            data=data[~self.mask_roi],
            f_ary=self.f_ary,
            df_rho_ary=self.df_rho_ary,
            k_max=self.k_max,
            npixROI=self.npixROI,
        )
        with numpyro.plate('data', self.npixROI):
            return numpyro.factor('ll', ll)


class NPModelGC2 (NPModelGC11):

    def set_truth(self, truth_dict):
        self.truth_dict = truth_dict

    def model(self, data=...):
        
        mu = jnp.zeros_like(data)

        # poissonian
        S_pib = self.truth_dict['S_pib']
        S_ics = self.truth_dict['S_ics']
        S_bub = self.truth_dict['S_bub']
        S_nfw = self.truth_dict['S_nfw']
        S_dsk = self.truth_dict['S_dsk']

        nm = self.normalization_mask
        mu += S_pib * self.pib[0] / jnp.mean(self.pib[0][~nm])
        mu += S_ics * self.ics[0] / jnp.mean(self.ics[0][~nm])
        mu += S_bub * self.temp_bub / jnp.mean(self.temp_bub[~nm])
        mu += S_dsk * self.temp_p_dsk_fixed / jnp.mean(self.temp_p_dsk_fixed[~nm])
        mu += S_nfw * self.temp_p_nfw_fixed / jnp.mean(self.temp_p_nfw_fixed[~nm])

        # non-poissonian
        temp_ps_nfw = self.temp_ps_nfw_fixed[~self.mask_roi] / jnp.mean(self.temp_ps_nfw_fixed[~nm])
        temp_ps_dsk = self.temp_p_dsk_fixed[~self.mask_roi] / jnp.mean(self.temp_p_dsk_fixed[~nm]) # same template as poissonian
        npt_compressed = jnp.array([temp_ps_nfw, temp_ps_dsk])

        theta = []
        for ps in ["nfw", "dsk"]:
            Sps = numpyro.sample("Sps_{}".format(ps), dist.Uniform(1e-3, 4.))
            n1 = self.truth_dict[f'n1_{ps}']
            n2 = self.truth_dict[f'n2_{ps}']
            n3 = self.truth_dict[f'n3_{ps}']
            sb1 = self.truth_dict[f'sb1_{ps}']
            lambda_s = self.truth_dict[f'lambdas_{ps}']

            theta_tmp = jnp.array([1., n1, n2, n3, sb1, lambda_s * sb1])
            s_ary = jnp.logspace(-1., 2., 1000)
            dnds_ary = dnds(s_ary, theta_tmp)
            num_photon_for_unit_theta0 = jnp.trapz(s_ary * dnds_ary, s_ary)
            theta.append([Sps / num_photon_for_unit_theta0, n1, n2, n3, sb1, lambda_s * sb1])
        theta = jnp.array(theta)
        
        ll = log_like_np(
            theta=theta,
            pt_sum_compressed=mu[~self.mask_roi],
            npt_compressed=npt_compressed,
            data=data[~self.mask_roi],
            f_ary=self.f_ary,
            df_rho_ary=self.df_rho_ary,
            k_max=self.k_max,
            npixROI=self.npixROI,
        )
        with numpyro.plate('data', self.npixROI):
            return numpyro.factor('ll', ll)


class NPModelGC2SCF (NPModelGC11):

    def set_truth(self, truth_dict):
        self.truth_dict = truth_dict

    def model(self, data=...):
        
        mu = jnp.zeros_like(data)

        # poissonian
        S_pib = self.truth_dict['S_pib']
        S_ics = self.truth_dict['S_ics']
        S_bub = self.truth_dict['S_bub']
        S_nfw = self.truth_dict['S_nfw']
        S_dsk = self.truth_dict['S_dsk']

        nm = self.normalization_mask
        mu += S_pib * self.pib[0] / jnp.mean(self.pib[0][~nm])
        mu += S_ics * self.ics[0] / jnp.mean(self.ics[0][~nm])
        mu += S_bub * self.temp_bub / jnp.mean(self.temp_bub[~nm])
        mu += S_dsk * self.temp_p_dsk_fixed / jnp.mean(self.temp_p_dsk_fixed[~nm])
        mu += S_nfw * self.temp_p_nfw_fixed / jnp.mean(self.temp_p_nfw_fixed[~nm])

        # non-poissonian
        temp_ps_nfw = self.temp_ps_nfw_fixed[~self.mask_roi] / jnp.mean(self.temp_ps_nfw_fixed[~nm])
        temp_ps_dsk = self.temp_p_dsk_fixed[~self.mask_roi] / jnp.mean(self.temp_p_dsk_fixed[~nm]) # same template as poissonian
        npt_compressed = jnp.array([temp_ps_nfw, temp_ps_dsk])

        theta = []
        for ps in ["nfw", "dsk"]:
            Sps = numpyro.sample("Sps_{}".format(ps), dist.Uniform(1e-3, 4.))
            n1 = numpyro.sample("n1_{}".format(ps), dist.Uniform(4.0, 6.0))
            n2 = numpyro.sample("n2_{}".format(ps), dist.Uniform(0.5, 1.99))
            n3 = numpyro.sample("n3_{}".format(ps), dist.Uniform(-6., -5.))
            sb1 = numpyro.sample("sb1_{}".format(ps), dist.Uniform(5., 40.0))
            lambda_s = numpyro.sample("lambdas_{}".format(ps), dist.Uniform(0.1, 0.95))

            theta_tmp = jnp.array([1., n1, n2, n3, sb1, lambda_s * sb1])
            s_ary = jnp.logspace(-1., 2., 1000)
            dnds_ary = dnds(s_ary, theta_tmp)
            num_photon_for_unit_theta0 = jnp.trapz(s_ary * dnds_ary, s_ary)
            theta.append([Sps / num_photon_for_unit_theta0, n1, n2, n3, sb1, lambda_s * sb1])
        theta = jnp.array(theta)
        
        ll = log_like_np(
            theta=theta,
            pt_sum_compressed=mu[~self.mask_roi],
            npt_compressed=npt_compressed,
            data=data[~self.mask_roi],
            f_ary=self.f_ary,
            df_rho_ary=self.df_rho_ary,
            k_max=self.k_max,
            npixROI=self.npixROI,
        )
        with numpyro.plate('data', self.npixROI):
            return numpyro.factor('ll', ll)


class NPModelGC7 (NPModelGC11):

    def set_truth(self, truth_dict):
        self.truth_dict = truth_dict

    def model(self, data=...):
        
        mu = jnp.zeros_like(data)

        # poissonian
        S_pib = numpyro.sample("S_pib", dist.Uniform(1e-3, 14.))
        S_ics = numpyro.sample("S_ics", dist.Uniform(1e-3, 14.))
        S_bub = numpyro.sample("S_bub", dist.Uniform(1e-3, 5.))
        S_nfw = numpyro.sample("S_nfw", dist.Uniform(1e-3, 5.))
        S_dsk = numpyro.sample("S_dsk", dist.Uniform(1e-3, 5.))

        nm = self.normalization_mask
        mu += S_pib * self.pib[0] / jnp.mean(self.pib[0][~nm])
        mu += S_ics * self.ics[0] / jnp.mean(self.ics[0][~nm])
        mu += S_bub * self.temp_bub / jnp.mean(self.temp_bub[~nm])
        mu += S_dsk * self.temp_p_dsk_fixed / jnp.mean(self.temp_p_dsk_fixed[~nm])
        mu += S_nfw * self.temp_p_nfw_fixed / jnp.mean(self.temp_p_nfw_fixed[~nm])

        # non-poissonian
        temp_ps_nfw = self.temp_ps_nfw_fixed[~self.mask_roi] / jnp.mean(self.temp_ps_nfw_fixed[~nm])
        temp_ps_dsk = self.temp_p_dsk_fixed[~self.mask_roi] / jnp.mean(self.temp_p_dsk_fixed[~nm]) # same template as poissonian
        npt_compressed = jnp.array([temp_ps_nfw, temp_ps_dsk])

        theta = []
        for ps in ["nfw", "dsk"]:
            Sps = numpyro.sample("Sps_{}".format(ps), dist.Uniform(1e-3, 4.))
            n1 = self.truth_dict[f'n1_{ps}']
            n2 = self.truth_dict[f'n2_{ps}']
            n3 = self.truth_dict[f'n3_{ps}']
            sb1 = self.truth_dict[f'sb1_{ps}']
            lambda_s = self.truth_dict[f'lambdas_{ps}']

            theta_tmp = jnp.array([1., n1, n2, n3, sb1, lambda_s * sb1])
            s_ary = jnp.logspace(-1., 2., 1000)
            dnds_ary = dnds(s_ary, theta_tmp)
            num_photon_for_unit_theta0 = jnp.trapz(s_ary * dnds_ary, s_ary)
            theta.append([Sps / num_photon_for_unit_theta0, n1, n2, n3, sb1, lambda_s * sb1])
        theta = jnp.array(theta)
        
        ll = log_like_np(
            theta=theta,
            pt_sum_compressed=mu[~self.mask_roi],
            npt_compressed=npt_compressed,
            data=data[~self.mask_roi],
            f_ary=self.f_ary,
            df_rho_ary=self.df_rho_ary,
            k_max=self.k_max,
            npixROI=self.npixROI,
        )
        with numpyro.plate('data', self.npixROI):
            return numpyro.factor('ll', ll)


class NPModelGCFull (NPModelGC11):

    def __init__(self, Alm=False, **kwargs):
        super().__init__(**kwargs)
        self.Alm = Alm

    def set_truth(self, truth_dict):
        self.truth_dict = truth_dict

    def model(self, data=...):
        
        # self.k_max = jnp.max(self.data[~self.mask_roi])
        mu = jnp.zeros_like(data)
        nm = self.nm

        # poissonian fixed (14): pib*3 ics*3 iso bub psc blg*5
        S_pib = numpyro.sample("S_pib", dist.Uniform(1e-4, 14.))
        theta_pib = numpyro.sample("theta_pib", dist.Dirichlet(jnp.ones((self.n_dif,)) / self.n_dif))
        temp_pib = jnp.sum(theta_pib[:, None] * self.pib, 0)
        if self.Alm: # Alm (6)
            multiplier_pib = jnp.zeros_like(data)
            for ii in range(len(self.Ylm_temps)):
                Alm = numpyro.sample("Alm_{}".format(ii), dist.Uniform(-0.05, 0.05))
                multiplier_pib += Alm * self.Ylm_temps[ii]
            temp_pib = (1. + multiplier_pib) * temp_pib
        temp_pib = temp_pib / jnp.mean(temp_pib[~nm])
        mu += S_pib * temp_pib
        
        S_ics = numpyro.sample("S_ics", dist.Uniform(1e-4, 10.))
        theta_ics = numpyro.sample("theta_ics", dist.Dirichlet(jnp.ones((self.n_dif,)) / self.n_dif))
        mu += S_ics * jnp.sum(theta_ics[:, None] * self.ics, 0)
        
        S_iso = numpyro.sample("S_iso", dist.Uniform(1e-4, 4.))
        S_bub = numpyro.sample("S_bub", dist.Uniform(1e-4, 4.))
        S_psc = numpyro.sample("S_psc", dist.Uniform(1e-4, 4.))
        mu += S_iso * self.temp_iso / jnp.mean(self.temp_iso[~nm])
        mu += S_bub * self.temp_bub / jnp.mean(self.temp_bub[~nm])
        mu += S_psc * self.temp_psc / jnp.mean(self.temp_psc[~nm])

        S_blg = numpyro.sample("S_blg", dist.Uniform(1e-4, 4.))
        theta_blg = numpyro.sample("theta_blg", dist.Dirichlet(jnp.ones((self.n_blg,)) / self.n_blg))
        mu += S_blg * jnp.sum(theta_blg[:, None] * self.blg, 0)

        # poissonian variable (2): nfw
        S_nfw = numpyro.sample("S_nfw", dist.Uniform(1e-4, 5.))
        gamma_poiss = numpyro.sample("gamma_poiss", dist.Uniform(0.2, 2.))
        temp_nfw_poiss = self.nfw_template.get_NFW2_template(gamma=gamma_poiss)
        mu += S_nfw * temp_nfw_poiss / jnp.mean(temp_nfw_poiss[~nm])

        # non-poissonian (20): gce 7+5 dsk 3+5
        Sps_nfw = numpyro.sample("Sps_nfw", dist.Uniform(1e-4, 4.))
        gamma_ps = numpyro.sample("gamma_ps", dist.Uniform(0.2, 2.))
        temp_nfw_ps = self.nfw_template.get_NFW2_template(gamma=gamma_ps)
        temp_gce_ps = Sps_nfw * temp_nfw_ps / jnp.mean(temp_nfw_ps[~nm])

        Sps_blg = numpyro.sample("Sps_blg", dist.Uniform(1e-4, 4.))
        theta_blg_ps = numpyro.sample("theta_blg_ps", dist.Dirichlet(jnp.ones((self.n_blg,)) / self.n_blg))
        temp_gce_ps += Sps_blg * jnp.sum(theta_blg_ps[:, None] * self.blg, 0)
        Sps_gce = jnp.mean(temp_gce_ps[~nm]) # definition of Sps_gce
        temp_gce_ps = temp_gce_ps / Sps_gce

        Sps_dsk = numpyro.sample("Sps_dsk", dist.Uniform(1e-4, 4.))
        zs = numpyro.sample("zs", dist.Uniform(0.1, 2.5))
        C = numpyro.sample("C", dist.Uniform(0.05, 8.))
        temp_dsk_ps = self.disk_template.get_template(zs=zs, C=C)
        temp_dsk_ps = temp_dsk_ps / jnp.mean(temp_dsk_ps[~nm])

        npt_compressed = jnp.array([temp_gce_ps[~self.mask_roi], temp_dsk_ps[~self.mask_roi]])

        theta = []
        for ps in ["gce", "dsk"]:
            Sps = Sps_gce if ps == "gce" else Sps_dsk
            n1 = numpyro.sample(f"n1_{ps}", dist.Uniform(4.0, 6.0))
            n2 = numpyro.sample(f"n2_{ps}", dist.Uniform(0.5, 1.99))
            n3 = numpyro.sample(f"n3_{ps}", dist.Uniform(-6., -5.))
            sb1 = numpyro.sample(f"sb1_{ps}", dist.Uniform(5., 40.0))
            lambda_s = numpyro.sample(f"lambdas_{ps}", dist.Uniform(0.1, 0.95))

            theta_tmp = jnp.array([1., n1, n2, n3, sb1, lambda_s * sb1])
            s_ary = jnp.logspace(-1., 2., 1000)
            dnds_ary = dnds(s_ary, theta_tmp)
            num_photon_for_unit_theta0 = jnp.trapz(s_ary * dnds_ary, s_ary)
            theta.append([Sps / num_photon_for_unit_theta0, n1, n2, n3, sb1, lambda_s * sb1])
        theta = jnp.array(theta)
        
        ll = log_like_np(
            theta=theta,
            pt_sum_compressed=mu[~self.mask_roi],
            npt_compressed=npt_compressed,
            data=data[~self.mask_roi],
            f_ary=self.f_ary,
            df_rho_ary=self.df_rho_ary,
            k_max=self.k_max,
            npixROI=self.npixROI,
        )
        ll = jnp.where(jnp.isnan(ll), -jnp.inf, ll)
        ll = jnp.where(jnp.isinf(ll), -100., ll)
        # ll = jnp.nan_to_num(ll)
        with numpyro.plate('data', self.npixROI):
            # with handlers.mask(mask=~jnp.logical_or(jnp.isinf(ll), jnp.isnan(ll))):
            return numpyro.factor('ll', ll)