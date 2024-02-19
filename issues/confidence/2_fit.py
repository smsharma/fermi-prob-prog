import argparse
import pickle

import numpy as np
import jax
import jax.numpy as jnp
from jax.example_libraries import stax
import optax
from numpyro import optim
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer import autoguide
from numpyro.infer.reparam import NeuTraReparam
from numpyro.infer import MCMC, NUTS

from simplemodel import model

from common import *

def fit_svi(
    model,
    rng_key=jax.random.PRNGKey(42),
    num_flows=5, hidden_dims=[128, 128],
    n_steps=5000, lr=5e-4, num_particles=8, vectorize_particles=True,
    **model_static_kwargs
):
    iaf_kwargs = dict(num_flows=num_flows, hidden_dims=hidden_dims, nonlinearity=stax.Tanh)
    guide = autoguide.AutoIAFNormal(model, **iaf_kwargs)
    optimizer = optim.optax_to_numpyro(optax.chain(optax.clip(1.), optax.adam(lr)))
    svi = SVI(model, guide, optimizer, Trace_ELBO(num_particles=num_particles, vectorize_particles=vectorize_particles))
    svi_results = svi.run(rng_key, n_steps, **model_static_kwargs)
    
    return svi_results, guide

def fit_hmc(
    model,
    num_chains=4, num_warmup=500, num_samples=5000, step_size=0.1,
    rng_key=jax.random.PRNGKey(0),
    **model_static_kwargs
):
    kernel = NUTS(model, max_tree_depth=4, dense_mass=False, step_size=step_size)
    nuts_mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, chain_method='vectorized')
    nuts_mcmc.run(rng_key, **model_static_kwargs)
    
    return nuts_mcmc

def fit_hmc_neutra(
    model, guide, svi_results,
    num_chains=4, num_warmup=500, num_samples=5000, step_size=0.1,
    rng_key=jax.random.PRNGKey(0),
    **model_static_kwargs
):
    neutra = NeuTraReparam(guide, svi_results.params)
    model_neutra = neutra.reparam(model)

    kernel = NUTS(model_neutra, max_tree_depth=4, dense_mass=False, step_size=step_size)
    nuts_mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, chain_method='vectorized')
    nuts_mcmc.run(rng_key, **model_static_kwargs)

    return nuts_mcmc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=int)
    parser.add_argument('--wdir', type=str, default='base')
    parser.add_argument('--mode', type=str)
    args = parser.parse_args()

    wdir = args.wdir
    fit_mode = args.mode

    # data
    data_masked = np.load(f"{wdir}/counts_{args.i}.npy")
    data_in = np.zeros(hp.nside2npix(nside), dtype=np.int32)
    data_in[~mask_plane] = data_masked
    data_in = jnp.asarray(data_in, dtype=jnp.int32)
    k_max = int(np.max(data_in))
    npixROI = int(np.sum(~mask_plane))

    # common kwargs
    kwargs = dict(
        data=data_in,
        k_max=k_max,
        npixROI=npixROI,
        deltapsf=True
    )

    # fit
    rng_key = jax.random.PRNGKey(42 + 4242 * args.i)
    if fit_mode == 'svi':
        key, subkey = jax.random.split(rng_key)
        svi_results, guide = fit_svi(model, rng_key=subkey, **kwargs)

        svi_samples = guide.sample_posterior(
            rng_key=key,
            params=svi_results.params,
            sample_shape=(50000,)
        )
        pickle.dump(svi_samples, open(f"{wdir}/svi_samples_{args.i}.p", "wb"))
        pickle.dump(svi_results, open(f"{wdir}/svi_results_{args.i}.p", "wb"))
        print('svi samples generated', flush=True)

    elif fit_mode == 'hmc':
        mcmc = fit_hmc(model, rng_key=rng_key, **kwargs)
        hmc_samples = mcmc.get_samples()
        pickle.dump(hmc_samples, open(f"{wdir}/hmc_samples_{args.i}.p", "wb"))
        print('hmc samples generated', flush=True)

    elif fit_mode == 'hmc-neutra':
        raise NotImplementedError