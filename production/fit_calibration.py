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
    parser.add_argument('-i', type=int) # 0-99
    parser.add_argument('--fit', type=str) # svi hmc hmctd10 pthmc
    parser.add_argument('--truth', type=str) # old new fullprior
    parser.add_argument('--psf', type=str) # delta king
    args = parser.parse_args()

    #=== dir ===
    subname = f"{args.fit}-{args.truth}-{args.psf}"
    print(f"Running {subname} # {args.i} ...")

    wdir_base = os.environ['MYSTORE'] + f"/fermi/fermi-prob-prog/outputs/production"
    os.makedirs(wdir_base + f"/fits/calibration/{subname}", exist_ok=True)

    #=== data ===
    data_name_dict = {
        'old': 'nmold',
        'new': 'nmnew',
        'fullprior': 'fullprior42',
    }
    data_name = data_name_dict[args.truth]
    if args.psf == 'delta':
        data_name += '-deltapsf'
    # data_name += "-2"
    print(f"Data name: {data_name}")

    data = jnp.array(np.load(wdir_base + f"/simulations/{data_name}.npy")[args.i], dtype=jnp.int32)
    
    #=== model ===
    m = NPModel(
        data=data,
        psf_tag=args.psf,
        n_exp=7,
    )

    #=== fit ===
    if args.fit == 'svi':
        m.fit_svi(
            data=data, rng_key=jax.random.PRNGKey(42),
            n_steps=10000, lr=3e-4,
            guide='iaf', num_flows=5, hidden_dims=[128, 128],
            num_particles=16,
        )
        samples = m.get_svi_samples(num_samples=50000)

    elif args.fit.startswith('hmc'):
        if args.fit == 'hmctd10':
            max_tree_depth = 10
        else:
            max_tree_depth = 4
        m.run_nuts(
            data=data, rng_key=jax.random.PRNGKey(42),
            num_chains=4, num_warmup=500, num_samples=10000//4,
            max_tree_depth=max_tree_depth, step_size=0.05,
        )
        samples = m.nuts_mcmc.get_samples()

    elif args.fit == 'pthmc':
        m.run_parallel_tempering_hmc(num_samples=10000)
        samples = m.mcmc.get_samples()
    
    pickle.dump(samples, open(wdir_base + f"/fits/calibration/{subname}/{args.i}.p", 'wb'))
