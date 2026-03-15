import os
import dill as pickle
import argparse

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from fpp.models.np_model import NPModel


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fit', type=str, choices=['svi', 'hmc'], required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    subname = f'{args.fit}-seed{args.seed}'
    print(f"Running {subname} ...")

    wdir = os.path.dirname(os.path.abspath(__file__))
    save_dir = f"{wdir}/../outputs/production/fits/fermi"
    os.makedirs(save_dir, exist_ok=True)

    data = jnp.array(np.load(f"{wdir}/../data/fermi_data_573w/fermi_data_128/fermidata_counts.npy"), dtype=jnp.int32)
    
    m = NPModel(
        data=data,
        psf_tag='king',
        n_exp=7,
        diffuse_names=["ModelO", "ModelA", "ModelF"],
    )

    if args.fit == 'svi':
        m.fit_svi(
            data=data, rng_key=jax.random.PRNGKey(args.seed),
            n_steps=10000, lr=3e-4,
            guide='iaf', num_flows=5, hidden_dims=[128, 128],
            num_particles=16,
        )
        samples = m.get_svi_samples(num_samples=50000)

    elif args.fit == 'hmc':
        m.run_nuts(
            data=data, rng_key=jax.random.PRNGKey(args.seed),
            num_chains=4, num_warmup=1000, num_samples=30000//4, step_size=0.05,
        )
        samples = m.nuts_mcmc.get_samples()
    
    pickle.dump(samples, open(f"{save_dir}/{subname}.p", 'wb'))
