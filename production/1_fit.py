import os
import sys
import dill as pickle
import argparse

import numpy as np
import healpy as hp

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

wdir = "/n/home07/yitians/fermi/fermi-prob-prog/production"
sys.path.append(f"{wdir}/..")
from models.np_model import NPModel


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=int)
    parser.add_argument('-n', type=int)
    parser.add_argument('--data', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--n_step', type=int, default=0)
    parser.add_argument('--fit_type', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--comment', type=str, default='')
    args = parser.parse_args()

    wdir = "/n/home07/yitians/fermi/fermi-prob-prog/production"
    comment_str = '' if args.comment == '' else f"_{args.comment}"
    run_name = f"{args.fit_type}_D{args.data}_M{args.model}" + comment_str
    print('run_name:', run_name)

    save_dir = f"{wdir}/../outputs/fit/{run_name}"
    os.makedirs(save_dir, exist_ok=True)

    mask_roi = jnp.load(f"{wdir}/mask_roi.npy")
    mask_norm = jnp.load(f"{wdir}/mask_norm.npy")

    data = np.load(f"../outputs/sims/{args.data}_n100.npy")[args.i]
    if len(data) < hp.nside2npix(128):
        data_full = np.zeros(hp.nside2npix(128))
        data_full[~mask_norm] = data
        data_in = jnp.array(data_full, dtype=jnp.int32)
    else:
        data_in = jnp.array(data, dtype=jnp.int32)

    if 'deltapsf' in args.model:
        m = NPModel(data=data_in, psf_tag='delta')
        print('USING DELTA PSF')
    else:
        m = NPModel(data=data_in, psf_tag='king')
        print('USING KING PSF')

    if args.fit_type == 'svi':
        m.fit_svi(
            n_steps=args.n_step, data=data_in, lr=1e-4,
            rng_key=jax.random.PRNGKey(args.seed)
        )
        samples = m.get_svi_samples(num_samples=args.n)

    elif args.fit_type in ['hmc', 'hmcnt']:

        if args.fit_type == 'hmcnt':
            m.fit_svi(
                n_steps=args.n_step, data=data_in, lr=1e-4,
                rng_key=jax.random.PRNGKey(args.seed)
            )
            
        mcmc = m.run_nuts(
            num_chains=4, num_warmup=1000, num_samples=args.n//4, step_size=0.05,
            use_neutra=(args.fit_type=='hmcnt'), data=data_in,
            rng_key=jax.random.PRNGKey(args.seed)
        )
        samples = mcmc.get_samples()

    elif args.fit_type in ['pthmc']:
        m.fit_svi(n_steps=args.n_step, data=data_in, lr=1e-4)
        mcmc = m.run_parallel_tempering_hmc(
            num_samples=args.n,
            step_size_base=5e-2,
            num_leapfrog_steps=3,
            num_adaptation_steps=600,
        )
        samples = mcmc.get_samples()

    elif args.fit_type == 'testhmc':
        mcmc = m.run_nuts(
            num_chains=8, num_warmup=10, num_samples=10, step_size=0.1,
            use_neutra=False, data=data_in
        )
        samples = mcmc.get_samples()

    else:
        raise NotImplementedError(args.fit_type)
    
    pickle.dump(samples, open(f"{save_dir}/i{args.i}_n{args.n}_ns{args.n_step}.p", 'wb'))
