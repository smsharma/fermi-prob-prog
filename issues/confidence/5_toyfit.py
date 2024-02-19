import numpy as np
import pickle
import argparse
import jax
from numpyro.infer import MCMC, NUTS
import numpyro
import numpyro.distributions as dist

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=int)
    args = parser.parse_args()

    key = jax.random.PRNGKey(42 + args.i*42)

    data_s = np.load(f"toysim_t01/data_s.npy")
    t0 = np.load(f"toytemps/toytemp_0.npy")
    t1 = np.load(f"toytemps/toytemp_1.npy")

    def model(data=...):
        S0 = numpyro.sample('S0', dist.Uniform(0, 10))
        S1 = numpyro.sample('S1', dist.Uniform(0, 10))
        mu = S0 * t0 + S1 * t1
        ll = dist.discrete.Poisson(mu).log_prob(data)
        return numpyro.factor('log-likelihood', ll)

    data = data_s[args.i]
    key, subkey = jax.random.split(key)
    mcmc = fit_hmc(model, data=data, rng_key=subkey)
    samples = mcmc.get_samples()
    pickle.dump(samples, open(f"toysim_t01/samples_{args.i}.p", 'wb'))