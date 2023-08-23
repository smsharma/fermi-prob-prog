import os
import sys

import numpy as np
import healpy as hp

import jax
import jax.numpy as jnp
from jax.example_libraries import stax
import numpyro
import numpyro.distributions as dist
from numpyro import optim
from numpyro.infer import SVI, Trace_ELBO, autoguide
from numpyro.infer.reparam import NeuTraReparam
from numpyro.infer import MCMC, NUTS
from numpyro.contrib.tfp.mcmc import ReplicaExchangeMC
import optax
from tensorflow_probability.substrates import jax as tfp

sys.path.append("..")
from utils import create_mask as cm
from likelihoods.pll_jax import log_like_poisson


os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"


class RandomModel:

    def __init__(self, rng_key, size, n_temp):
        self.size = size
        self.n_temp = n_temp

        temps = []
        for _ in range(n_temp):
            rng_key, key = jax.random.split(rng_key)
            temp = jax.random.uniform(rng_key, shape=(self.size,))
            temp /= jnp.mean(temp)
            temps.append(temp)
        self.temps = jnp.array(temps)

    def set_truth(self, truth_arr):
        self.truth_arr = truth_arr
        self.mu = jnp.einsum('i,ij->j', truth_arr, self.temps)

    def generate_counts(self, rng_key):
        return jax.random.poisson(rng_key, self.mu)
    
    def model(self):
        mu = jnp.zeros((self.size,))
        for i in range(self.n_temp):
            mu += self.temps[i] * numpyro.sample(f'temp_{i}', dist.Uniform(0, 10))
        return mu

    def conditioned_model(self, counts=...):
        mu = self.model()
        with numpyro.plate("data", size=len(mu), dim=-1):
            ll = log_like_poisson(mu, counts)
            return numpyro.factor('log-likelihood', ll)
        
    def fit_svi(
        self, rng_key=jax.random.PRNGKey(42),
        guide='iaf', num_flows=5, hidden_dims=[256, 256],
        n_steps=7500, lr=5e-5, num_particles=8,
        **model_static_kwargs,
    ):
        if guide == 'iaf':
            self.guide = autoguide.AutoIAFNormal(
                self.conditioned_model,
                num_flows=num_flows,
                hidden_dims=hidden_dims,
                nonlinearity=stax.Tanh
            )
        elif guide == 'mvn':
            self.guide = autoguide.AutoMultivariateNormal(self.conditioned_model)
        else:
            raise NotImplementedError
        
        optimizer = optim.optax_to_numpyro(
            optax.chain(
                optax.clip(1.),
                optax.adamw(lr),
            )
        )
        self.svi_model_static_kwargs = model_static_kwargs
        svi = SVI(
            self.conditioned_model, self.guide, optimizer,
            Trace_ELBO(num_particles=num_particles),
            **self.svi_model_static_kwargs,
        )
        self.svi_results = svi.run(rng_key, n_steps)
        
        return self.svi_results
    
    def get_svi_samples(self, num_samples, rng_key=jax.random.PRNGKey(42)):
        self.svi_samples = self.guide.sample_posterior(
            rng_key=rng_key,
            params=self.svi_results.params,
            sample_shape=(num_samples,)
        )
        return self.svi_samples
    
    def get_neutra_model(self):
        neutra = NeuTraReparam(self.guide, self.svi_results.params)
        self.model_neutra = neutra.reparam(self.conditioned_model)
        
    def run_nuts(self, num_chains=4, num_warmup=500, num_samples=5000, step_size=0.1,
                 rng_key=jax.random.PRNGKey(0), use_neutra=True, **model_static_kwargs):
        
        if use_neutra:
            self.get_neutra_model()
            model = self.model_neutra
        else:
            model = self.conditioned_model
        
        kernel = NUTS(model, max_tree_depth=4, dense_mass=False, step_size=step_size)
        self.nuts_mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, chain_method='vectorized')
        self.nuts_mcmc.run(rng_key, **model_static_kwargs)
        
        return self.nuts_mcmc
    
    def run_parallel_tempering_hmc(self, num_samples=5000, step_size_base=5e-2, num_leapfrog_steps=3, num_adaptation_steps=600, rng_key=jax.random.PRNGKey(0), use_neutra=True):
        
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
        
        if use_neutra:
            self.get_neutra_model()
            model = self.model_neutra
        else:
            model = lambda x: self.conditioned_model(**self.svi_model_static_kwargs)
        
        kernel = ReplicaExchangeMC(model, inverse_temperatures=inverse_temperatures, make_kernel_fn=make_kernel_fn)
        self.pt_mcmc = MCMC(kernel, num_warmup=num_adaptation_steps, num_samples=num_samples, num_chains=1, chain_method='vectorized')
        self.pt_mcmc.run(rng_key, None)
        
        return self.pt_mcmc
    
    def fit_MAP(
        self, rng_key=jax.random.PRNGKey(42),
        lr=0.1, n_steps=10000, num_particles=8,
        **model_static_kwargs,
    ):
        guide = autoguide.AutoDelta(self.conditioned_model)
        optimizer = optim.optax_to_numpyro(optax.chain(optax.clip(1.), optax.adamw(lr)))
        svi = SVI(
            self.model, guide, optimizer,
            loss=Trace_ELBO(num_particles=num_particles),
            **model_static_kwargs,
        )
        svi_results = svi.run(rng_key, n_steps)
        self.MAP_estimates = guide.median(svi_results.params)
        
        return svi_results
    

class PoissonModel (RandomModel):

    def __init__(self):

        current_file_path = os.path.abspath(os.path.dirname(__file__))
        data_dir = os.path.join(current_file_path, f"../data/fermi_data_573w/fermi_data_128")

        self.labels = ['psc', 'iso', 'bub', 'dsk', 'Opi', 'Oic', 'Api', 'Aic', 'Fpi', 'Fic', 'nfw']
        self.temps_unmasked = [
            np.load(f"{data_dir}/template_psc_3fgl.npy"),
            np.load(f"{data_dir}/template_iso.npy"),
            np.load(f"{data_dir}/template_bub.npy"),
            np.load(f"{data_dir}/template_dsk_z0p3.npy"),
            np.load(f"{data_dir}/template_Opi.npy"),
            np.load(f"{data_dir}/template_Oic.npy"),
            np.load(f"{data_dir}/template_Api.npy"),
            np.load(f"{data_dir}/template_Aic.npy"),
            np.load(f"{data_dir}/template_Fpi.npy"),
            np.load(f"{data_dir}/template_Fic.npy"),
            np.load(f"{data_dir}/template_nfw_g1p2.npy"),
        ]
        mask_ps = hp.ud_grade(np.load(f"{data_dir}/../../mask_3fgl_0p8deg.npy"), nside_out=128) > 0
        self.mask_roi = cm.make_mask_total(
            nside=128, band_mask=True, band_mask_range=2,
            mask_ring=True, inner=0, outer=25, custom_mask=mask_ps
        )
        self.temps = [temp[~self.mask_roi] for temp in self.temps_unmasked]
        self.temps = [temp / np.mean(temp) for temp in self.temps]
        self.size = len(self.temps[0])
        self.n_temp = len(self.temps)