import os
import sys
import dill as pickle
import argparse
import time

import numpy as np
import healpy as hp

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

wdir = "/n/home07/yitians/fermi/fermi-prob-prog/analysis"
from fpp.models.np_model_cmp import NPModelCMP


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=int)
    parser.add_argument('--fit_type', type=str)
    args = parser.parse_args()

    dim_s = [10, 14, 18, 22]
    dim = dim_s[args.i]

    run_name = f'{args.fit_type}_d{dim}'
    print('run_name:', run_name)
    print('fit_type:', args.fit_type)
    print('dim:', dim)
    save_dir = f"{wdir}/../outputs/fits_cmp/{run_name}"
    os.makedirs(save_dir, exist_ok=True)

    #===== data & model =====
    data = jnp.array(np.load(f'/n/home07/yitians/fermi/fermi-prob-prog/outputs/simulations/fermi.npy').astype(np.int32)[0])
    m = NPModelCMP(data=data, dim=dim)

    #===== fit =====
    timer_start = time.time()
    if args.fit_type == 'svi':
        m.fit_svi(model_name='model', n_steps=7500, lr=3e-4)
        samples = m.get_svi_samples(num_samples=50000)

    elif args.fit_type == 'hmc':
        mcmc = m.run_nuts(model_name='model', num_chains=4, num_warmup=1000, num_samples=20000//4, step_size=0.05,)
        samples = mcmc.get_samples()

    else:
        raise NotImplementedError(args.fit_type)
    timer_end = time.time()
    print(f"Time taken for {run_name}: {timer_end - timer_start:.2f} seconds")
    
    pickle.dump(samples, open(f"{save_dir}/{args.fit_type}_samples.p", 'wb'))
    if args.fit_type == 'svi':
        pickle.dump(m.svi_results.losses, open(f"{save_dir}/losses.p", 'wb'))