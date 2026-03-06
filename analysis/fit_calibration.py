import os
import sys
import dill as pickle
import argparse

import numpy as np
import healpy as hp

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

wdir = "/n/home07/yitians/fermi/fermi-prob-prog/analysis"
from fpp.models.np_model import NPModel


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=int) # 0-29
    parser.add_argument('--fit', type=str) # svi hmc
    parser.add_argument('--truth', type=str) # old new
    parser.add_argument('--psf', type=str) # delta king
    args = parser.parse_args()


    subname = f"{args.fit}-{args.truth}-{args.psf}"
    print(f"Running {subname} # {args.i} ...")

    save_dir = f"{wdir}/../outputs/production/fits/calibration/{subname}"
    os.makedirs(save_dir, exist_ok=True)

    data_name = ('nmold' if args.truth == 'old' else 'nmnew')
    if args.psf == 'delta':
        data_name += '_deltapsf'
    print(f"Data name: {data_name}")

    data = jnp.array(np.load(f"../outputs/production/simulations/{data_name}.npy")[0], dtype=jnp.int32)
    
    m = NPModel(
        data=data,
        psf_tag=args.psf,
        n_exp=7,
    )

    if args.fit == 'svi':
        m.fit_svi(
            data=data, rng_key=jax.random.PRNGKey(42),
            n_steps=10000, lr=3e-4,
            guide='iaf', num_flows=5, hidden_dims=[128, 128],
            num_particles=16,
        )
        samples = m.get_svi_samples(num_samples=50000)

    elif args.fit == 'hmc':
        m.run_nuts(
            data=data, rng_key=jax.random.PRNGKey(42),
            num_chains=4, num_warmup=1000, num_samples=10000//4, step_size=0.05,
        )
        samples = m.nuts_mcmc.get_samples()
    
    pickle.dump(samples, open(f"{save_dir}/{args.i}.p", 'wb'))
