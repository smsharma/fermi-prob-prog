import sys

sys.path.append("..")

import numpyro
import healpy as hp
import numpy as np
import jax.numpy as jnp
import jax
from jax import jit

import numpyro.distributions as dist
from numpyro import handlers

from models.np_model import NPModel

@jit
def log_like_poisson(pt_sum_compressed, data):
    return dist.discrete.Poisson(pt_sum_compressed).log_prob(data)


class PoissonModel (NPModel):
    
    def __init__(self, **kwargs):
        
        print('PoissonModel: calling super().__init__()')
        super().__init__(**kwargs)
        
        del self.k_max
        del self.f_ary
        del self.df_rho_div_f_ary
    
    
    def model(self, data):
        # all commented code are in NPModel.model
        
        theta_pibrem = numpyro.sample("theta_pibrem", dist.Dirichlet(jnp.ones((self.n_dif_templates,)) / self.n_dif_templates))
        temp_pibrem = jnp.sum(theta_pibrem[:, None] * self.pibrem, 0)

        theta_ics = numpyro.sample("theta_ics", dist.Dirichlet(jnp.ones((self.n_dif_templates,)) / self.n_dif_templates))
        temp_ics = jnp.sum(theta_ics[:, None] * self.ics, 0)

        S_gce = numpyro.sample("S_gce", dist.Uniform(1e-5, 2.))
            
        temps = [self.temp_iso, self.temp_bub, self.temp_psc, temp_pibrem, temp_ics]
        temp_labels = ["iso", "bub", "psc", "dif", "ics"]
                
        mu = jnp.zeros_like(data)
        
        for temp, temp_label in zip(temps, temp_labels):
            
            if temp_label in ["dif"]:
                prior_lo, prior_hi = 1e-3, 14.
            else:
                prior_lo, prior_hi = 1e-3, 5.0
                
            prior_dist = dist.Uniform(prior_lo, prior_hi)
            S_temp = numpyro.sample("S_{}".format(temp_label), prior_dist)
            
            if temp_label in ["dif"]:
                
                temp_dif_mod = jnp.zeros_like(data)
                for ii in range(len(self.Ylm_temps)):
                    Alm = numpyro.sample("Alm_{}".format(ii), dist.Uniform(-0.05, 0.05))
                    temp_dif_mod += Alm * self.Ylm_temps[ii]
                
                temp_dif_mod = (1. + temp_dif_mod) * temp
                
                A_temp = S_temp / jnp.mean(temp_dif_mod[~self.mask_plane])
                mu += A_temp * temp_dif_mod  
            else:
                A_temp = S_temp / jnp.mean(temp[~self.mask_plane])
                mu += A_temp * temp     
                                            
        if self.vary_gamma:
            #gamma_ps = numpyro.sample("gamma_ps", dist.Uniform(0.2, 2.))
            gamma_poiss = numpyro.sample("gamma_poiss", dist.Uniform(0.2, 2.))
        else:
            #gamma_ps = 1.2
            gamma_poiss = 1.2

        #temp_gce_nfw_ps = self.nfw_template.get_NFW2_template(gamma=gamma_ps)
        temp_gce_nfw_poiss = self.nfw_template.get_NFW2_template(gamma=gamma_poiss)
            
        # if self.vary_disk:
        #     zs = numpyro.sample("zs", dist.Uniform(0.1, 2.5))
        #     C = numpyro.sample("C", dist.Uniform(0.05, 15.))
        #     temp_dsk = self.disk_template.get_template(zs=zs, C=C)
        # else:
        #     temp_dsk = self.temp_dsk
                
        if self.bulge_hybrid:
            f_bulge_poiss = numpyro.sample("f_bulge_poiss", dist.Uniform(0., 1.))
            #f_bulge_ps = numpyro.sample("f_bulge_ps", dist.Uniform(0., 1.))
        else:
            f_bulge_poiss = 0.
            #f_bulge_ps = 0.
            
        theta_bulge_poiss = numpyro.sample("theta_bulge_poiss", dist.Dirichlet(jnp.ones((self.n_bulge_templates,)) / self.n_bulge_templates))
        temp_bulge = jnp.sum(theta_bulge_poiss[:, None] * self.bulge_templates, 0)
        
        A_gce_nfw = S_gce / jnp.mean(temp_gce_nfw_poiss[~self.mask_plane])
        A_gce_bulge = S_gce / jnp.mean(temp_bulge[~self.mask_plane])
        
        
        temp_gce_poiss = (1 - f_bulge_poiss) * A_gce_nfw * temp_gce_nfw_poiss + f_bulge_poiss * A_gce_bulge * temp_bulge
        
        A_gce = S_gce / jnp.mean(temp_gce_poiss[~self.mask_plane])
        mu += A_gce * temp_gce_poiss
        
        #theta_bulge_ps = numpyro.sample("theta_bulge_ps", dist.Dirichlet(jnp.ones((self.n_bulge_templates,)) / self.n_bulge_templates))
        #temp_bulge = jnp.sum(theta_bulge_ps[:, None] * self.bulge_templates, 0)

        #A_gce_nfw = 1 / jnp.mean(temp_gce_nfw_ps[~self.mask_plane])
        #A_gce_bulge = 1 / jnp.mean(temp_bulge[~self.mask_plane])
        
        #temp_gce_ps = (1 - f_bulge_ps) * A_gce_nfw * temp_gce_nfw_ps + f_bulge_ps * A_gce_bulge * temp_bulge
            
        #npt_compressed = jnp.array([temp_gce_ps, temp_dsk])

#         theta = []    
        
#         for ips, ps in enumerate(["gce", "dsk"]):
            
#             Sps = numpyro.sample("Sps_{}".format(ps), dist.Uniform(1e-5, 2.))
            
#             n1 = numpyro.sample("n1_{}".format(ps), dist.Uniform(4.0, 6.0))
#             n2 = numpyro.sample("n2_{}".format(ps), dist.Uniform(0.5, 1.99))
#             n3 = numpyro.sample("n3_{}".format(ps), dist.Uniform(-6., -5.))
#             sb1 = numpyro.sample("sb1_{}".format(ps), dist.Uniform(5., 40.0))
#             lambda_s = numpyro.sample("lambdas_{}".format(ps), dist.Uniform(0.1, 0.95))
            
#             theta_tmp = jnp.array([1., n1, n2, n3, sb1, lambda_s * sb1])
            
#             s_ary = jnp.logspace(0., 2, 100)
#             dnds_ary = dnds(s_ary, theta_tmp)
                    
#             A = Sps / jnp.mean(npt_compressed[ips][~self.mask_plane] * jnp.trapz(s_ary * dnds_ary, s_ary))
                                        
#             theta.append([A, n1, n2, n3, sb1, lambda_s * sb1])
            
#         theta = jnp.array(theta)
                
        # Pad the last exposure region so that all are the same size
        exp_lens = [len(self.expreg_indices[i]) for i in range(len(self.expreg_indices))]
        n_pad = exp_lens[0] - exp_lens[-1]
        
        expreg_indices = jnp.zeros_like(self.expreg_indices)
        expreg_indices = expreg_indices.at[:-1].set(self.expreg_indices[:-1])
        expreg_indices = expreg_indices.at[-1].set(jnp.pad(self.expreg_indices[-1], (0, n_pad)))

        #log_like_np_exp_vmapped = jax.vmap(log_like_np, in_axes=(0, 0, 1, 0, None, None, None, None))
        log_like_poisson_exp_vmapped = jax.vmap(log_like_poisson, in_axes=(0, 0))
                
        # Get relevant arrays for different exposure regions
        mu_batch = mu[~self.mask_roi][jnp.array(expreg_indices)]
        #npt_compressed_batch = npt_compressed[:, ~self.mask_roi][:, jnp.array(expreg_indices)]
        data_batch = self.data[~self.mask_roi][jnp.array(expreg_indices)]
        
        exposure_multiplier = self.exposure_means_list / self.exposure_mean
        
        # Scale non-Poissonian parameters (norm divided by exposure ratio, breaks multiplied)
        # theta = repeat(theta, "n_ps n_param -> n_exp n_ps n_param", n_exp=len(expreg_indices))
        # theta = theta.at[:, :, 0].set(theta[:, :, 0] / exposure_multiplier[:, None])
        # theta = theta.at[:, :, -1].set(theta[:, :, -1] * exposure_multiplier[:, None])
        # theta = theta.at[:, :, -2].set(theta[:, :, -2] * exposure_multiplier[:, None])
        
        if self.debug_model:
            print('mu.shape', mu.shape)
            print('mu_batch.shape', mu_batch.shape)
            print('data_batch.shape', data_batch.shape)
            print('expreg_indices.shape', expreg_indices.shape)
        
        with numpyro.plate("data", size=len(mu[~self.mask_roi]), dim=-1):            
                
            #log_like_exp = log_like_np_exp_vmapped(theta, mu_batch, npt_compressed_batch, data_batch, self.f_ary, self.df_rho_div_f_ary, self.k_max, len(expreg_indices[0]))
            log_like_exp = log_like_poisson_exp_vmapped(mu_batch, data_batch)
            
            # Concatenate exposure regions
            loglike = jnp.concatenate(log_like_exp)[:len(mu[~self.mask_roi])]
            
            if self.debug_model:
                print('log_like_exp.shape', log_like_exp.shape)
                print('loglike.shape', loglike.shape)
                                
            with handlers.mask(mask=~jnp.logical_or(jnp.isinf(loglike), jnp.isnan(loglike))):
                return numpyro.factor('log-likelihood', loglike)
    
    
    def get_neutra_model(self):
        raise NotImplementedError
        
    def run_nuts(self):
        raise NotImplementedError
    
    def run_parallel_tempering_hmc(self):
        raise NotImplementedError