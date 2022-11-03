import numpyro
import healpy as hp
import numpy as np
import jax.numpy as jnp
import jax
import numpyro.distributions as dist

from numpyro.infer import SVI, Predictive, Trace_ELBO, autoguide
from numpyro import optim
import optax
import arviz as az

from models.scd import dnds
from models.templates import NFWTemplate, LorimerDiskTemplate
from models.bulge_models import BulgeTemplates
from likelihoods.npll_jax import log_like_np
from utils.sph_harm import Ylm
from utils import create_mask as cm
from models.psf import KingPSF
from utils.psf_correction import PSFCorrection


class NPModel:
    def __init__(self, r_outer=25, l_max=0, dif="ModelO", vary_disk=True, vary_gamma=True, bulge_hybrid=True, bulge_template_name="macias2019", ps_cat="3fgl", nside=128):
        
        self.nside = nside
        self.ps_cat = ps_cat
        
        self.data_dir = "../data/fermi_data_573w/fermi_data_{}".format(self.nside)

        self.data = jnp.array(np.load("{}/fermidata_counts.npy".format(self.data_dir)).astype(np.int32))

        # mask_ps = np.load("{}/fermidata_pscmask_{}.npy".format(self.data_dir, self.ps_cat)) == 1
        mask_ps = hp.ud_grade(np.load("../data/mask_3fgl_0p8deg.npy"), nside_out=self.nside) > 0
        
        if ps_cat != "3fgl":
            raise NotImplementedError("Catalogs other than 3FGL not supported at the moment.")
        
        self.mask_roi = cm.make_mask_total(nside=self.nside, band_mask=True, band_mask_range=2., mask_ring=True, inner=0, outer=r_outer, custom_mask=mask_ps)
        self.mask_plane = cm.make_mask_total(nside=self.nside, band_mask=True, band_mask_range=2., mask_ring=True, inner=0, outer=25,)

        self.nfw_template = NFWTemplate(nside=self.nside)
        self.disk_template = LorimerDiskTemplate(nside=self.nside)

        self.bulge_template = BulgeTemplates(template_name=bulge_template_name, nside_out=self.nside)()

        self.l_max = l_max
        self.dif = dif

        self.load_templates()
        self.get_sphharms()
        self.get_psf_correction()
        
        self.bulge_hybrid = bulge_hybrid
        self.vary_gamma = vary_gamma
        self.vary_disk = vary_disk

        self.k_max = np.max(np.array(self.data)[~self.mask_roi])
        print("Max photon count is {}".format(self.k_max))

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
        self.temp_mO_pibrem = np.load("{}/template_Opi.npy".format(self.data_dir))
        self.temp_mO_ics = np.load("{}/template_Oic.npy".format(self.data_dir))

        # Load Model A templates
        self.temp_mA_pibrem = np.load("{}/template_Api.npy".format(self.data_dir))
        self.temp_mA_ics = np.load("{}/template_Aic.npy".format(self.data_dir))

        # Load Model F templates
        self.temp_mF_pibrem = np.load("{}/template_Fpi.npy".format(self.data_dir))
        self.temp_mF_ics = np.load("{}/template_Fic.npy".format(self.data_dir))

        if self.dif == "ModelO":
            self.pibrem = self.temp_mO_pibrem
            self.ics = self.temp_mO_ics
        elif self.dif == "ModelA":
            self.pibrem = self.temp_mA_pibrem
            self.ics = self.temp_mA_ics
        elif self.dif == "ModelF":
            self.pibrem = self.temp_mF_pibrem
            self.ics = self.temp_mF_ics

    def model(self, data, subsample_frac=0.8):

        S_gce = numpyro.sample("S_gce", dist.Uniform(1e-5, 3.))
            
        temps = [self.temp_iso, self.temp_bub, self.temp_psc, self.pibrem, self.ics]
        temp_labels = ["iso", "bub", "psc", "dif", "ics"]
        
        temp_bulge = self.bulge_template
        
        mu = jnp.zeros_like(data)
        
        for temp, temp_label in zip(temps, temp_labels):
            
            if temp_label in ["dif"]:
                prior_dist = dist.Uniform(1e-5, 20.0)
            else:
                prior_dist = dist.Uniform(1e-5, 5.0)
    
            S_temp = numpyro.sample("S_{}".format(temp_label), prior_dist)
            
            if temp_label in ["dif"]:
                
                temp_dif_mod = jnp.zeros_like(data)
                for ii in range(len(self.Ylm_temps)):
                    Alm = numpyro.sample("Alm_{}".format(ii), dist.Normal(0., 0.15))
                    temp_dif_mod += Alm * self.Ylm_temps[ii]
                
                temp_dif_mod = (1. + temp_dif_mod) * temp
                
                A_temp = S_temp / jnp.mean(temp_dif_mod[~self.mask_plane])
                mu += A_temp * temp_dif_mod  
            else:
                A_temp = S_temp / jnp.mean(temp[~self.mask_plane])
                mu += A_temp * temp     
                                            
        if self.vary_gamma:
            gamma_ps = numpyro.sample("gamma_ps", dist.Uniform(0.2, 2.))
            gamma_poiss = numpyro.sample("gamma_poiss", dist.Uniform(0.2, 2.))
        else:
            gamma_poiss = gamma_ps = 1.2

        temp_gce_nfw_ps = self.nfw_template.get_NFW2_template(gamma=gamma_ps)
        temp_gce_nfw_poiss = self.nfw_template.get_NFW2_template(gamma=gamma_poiss)
            
        if self.vary_disk:
            zs = numpyro.sample("zs", dist.Uniform(0.1, 2.5))
            C = numpyro.sample("C", dist.Uniform(0.05, 15.))
            temp_dsk = self.disk_template.get_template(zs=zs, C=C)
        else:
            temp_dsk = self.temp_dsk
                
        if self.bulge_hybrid:
            f_bulge_poiss = numpyro.sample("f_bulge_poiss", dist.Uniform(0., 1.))
            f_bulge_ps = numpyro.sample("f_bulge_ps", dist.Uniform(0., 1.))
        else:
            f_bulge_poiss = f_bulge_ps = 0.
            
        # Normalize to same mean
        A_gce_nfw = S_gce / jnp.mean(temp_gce_nfw_poiss[~self.mask_plane])
        A_gce_bulge = S_gce / jnp.mean(temp_bulge[~self.mask_plane])
        
        # Get hybrid template
        temp_gce_poiss = (1 - f_bulge_poiss) * A_gce_nfw * temp_gce_nfw_poiss + f_bulge_ps * A_gce_bulge * temp_bulge
        
        A_gce = S_gce / jnp.mean(temp_gce_poiss[~self.mask_plane])
        mu += A_gce * temp_gce_poiss
        
        # Normalize to same mean
        A_gce_nfw = 1 / jnp.mean(temp_gce_nfw_ps[~self.mask_plane])
        A_gce_bulge = 1 / jnp.mean(temp_bulge[~self.mask_plane])
        
        # Get hybrid template
        temp_gce_ps = (1 - f_bulge_ps) * A_gce_nfw * temp_gce_nfw_ps + f_bulge_ps * A_gce_bulge * temp_bulge
            
        npt_compressed = jnp.array([temp_gce_ps, temp_dsk])

        theta = []    
        
        for ips, ps in enumerate(["gce", "dsk"]):
            
            Sps = numpyro.sample("Sps_{}".format(ps), dist.Uniform(1e-5, 3.))
            
            n1 = numpyro.sample("n1_{}".format(ps), dist.Uniform(3.0, 10.0))
            n2 = numpyro.sample("n2_{}".format(ps), dist.Uniform(0.5, 2.))
            n3 = numpyro.sample("n3_{}".format(ps), dist.Uniform(-5., -4.))
            sb1 = numpyro.sample("sb1_{}".format(ps), dist.Uniform(5., 40.0))
            lambda_s = numpyro.sample("lambdas_{}".format(ps), dist.Uniform(0.1, 0.99))
            
            theta_tmp = jnp.array([1., n1, n2, n3, sb1, lambda_s * sb1])
            
            s_ary = jnp.logspace(0., 2, 1000)
            dnds_ary = dnds(s_ary, theta_tmp)
                    
            A = Sps / jnp.mean(npt_compressed[ips][~self.mask_plane] * jnp.trapz(s_ary * dnds_ary, s_ary))
                                        
            theta.append([A, n1, n2, n3, sb1, lambda_s * sb1])
            
        theta = jnp.array(theta)
        
        subsample_size = int(subsample_frac * len(data[~self.mask_roi]))
        with numpyro.plate("data", len(mu[~self.mask_roi]), subsample_size=subsample_size) as ind:
                
            loglike = log_like_np(theta, mu[~self.mask_roi][ind], npt_compressed[:, ~self.mask_roi][:, ind], self.data[~self.mask_roi][ind], self.f_ary, self.df_rho_div_f_ary, self.k_max, subsample_size)
            
            return numpyro.factor('log-likelihood', loglike)

    def fit_svi(self, rng_key=jax.random.PRNGKey(1), n_steps=5000, lr=5e-3, num_particles=2, subsample_frac=0.8):

        self.guide = autoguide.AutoMultivariateNormal(self.model)
        optimizer = optim.optax_to_numpyro(optax.chain(optax.clip(10.0), optax.adam(lr)))
        
        svi = SVI(self.model, self.guide, optimizer, Trace_ELBO(num_particles=num_particles))
        self.svi_results = svi.run(rng_key, n_steps, self.data, subsample_frac=subsample_frac)
        
        return self.svi_results
        
    def get_posterior_samples(self, rng_key=jax.random.PRNGKey(1), num_samples=50000):
        
        rng_key, key = jax.random.split(rng_key)
        posterior = self.guide.sample_posterior(rng_key=rng_key, params=self.svi_results.params, sample_shape=(num_samples,))
        arviz_post = az.from_dict(posterior)
        
        return arviz_post