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
from fpp.models.np_model_cmp import NPModelCMP


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--fit_type', type=str)
    parser.add_argument('-n', type=int)
    parser.add_argument('--n_step', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args()

    run_name = args.run_name
    print('run_name:', run_name)

    save_dir = f"{wdir}/../outputs/fits_cmp/{run_name}"
    os.makedirs(save_dir, exist_ok=True)

    #===== model =====
    m = NPModelCMP()

    #===== fit =====
    if args.fit_type == 'svi':
        m.fit_svi(n_steps=args.n_step, lr=args.lr)
        samples = m.get_svi_samples(num_samples=args.n)

    elif args.fit_type == 'hmc':
        mcmc = m.run_nuts(num_chains=4, num_warmup=1000, num_samples=args.n//4, step_size=0.05,)
        samples = mcmc.get_samples()

    else:
        raise NotImplementedError(args.fit_type)
    
    pickle.dump(samples, open(f"{save_dir}/{args.fit_type}_samples.p", 'wb'))
    if args.fit_type == 'svi':
        pickle.dump(m.svi_results.losses, open(f"{save_dir}/losses.p", 'wb'))