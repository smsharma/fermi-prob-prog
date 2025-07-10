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
from models.np_model_1b import NPModel1B


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=int)
    parser.add_argument('-n', type=int)
    parser.add_argument('--data', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--n_exp', type=int)
    parser.add_argument('--n_step', type=int, default=0)
    parser.add_argument('--fit_type', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--guide', type=str, default='iaf')
    parser.add_argument('--n_par', type=int, default=8)
    parser.add_argument('--renyi_alpha', type=float, default=1)
    parser.add_argument('--num_flows', type=int, default=4)
    parser.add_argument('--hidden_dim_n', type=int, default=64)
    parser.add_argument('--comment', type=str, default='')
    args = parser.parse_args()

    wdir = "/n/home07/yitians/fermi/fermi-prob-prog/production"
    comment_str = '' if args.comment == '' else f"_{args.comment}"
    run_name = f"{args.fit_type}_D{args.data}_M{args.model}" + comment_str
    print('run_name:', run_name)

    save_dir = f"{wdir}/../outputs/fit/{run_name}"
    os.makedirs(save_dir, exist_ok=True)

    mask_roi = np.load(f"{wdir}/mask_roi.npy")
    mask_norm = jnp.load(f"{wdir}/mask_norm.npy")

    # Ensure mask_roi's length is divisible by args.n_exp
    n_pix_remainder = int(np.sum(~mask_roi)) % args.n_exp
    print(f'Pixel number: {int(np.sum(~mask_roi))}', end=' ')
    if n_pix_remainder != 0:
        unmasked_indices = np.where(mask_roi == 0)[0]
        mask_roi[unmasked_indices[-n_pix_remainder:]] = 1
    print(f'-> {int(np.sum(~mask_roi))} = {args.n_exp} * {int(np.sum(~mask_roi) / args.n_exp)}')

    data = np.load(f"../outputs/sims/{args.data}.npy")[args.i]
    if len(data) < hp.nside2npix(128):
        data_full = np.zeros(hp.nside2npix(128))
        data_full[~mask_norm] = data
        data_in = jnp.array(data_full, dtype=jnp.int32)
    else:
        data_in = jnp.array(data, dtype=jnp.int32)

    psf_tag = 'delta' if 'deltapsf' in args.model else 'king'
    print('PSF:', psf_tag)
    if '1b' in args.model:
        Model = NPModel1B
        print('Using NPModel1B model')
    else:
        Model = NPModel
        print('Using NPModel model')
    m = Model(data=data_in, psf_tag=psf_tag, n_exp=args.n_exp, custom_mask_roi=mask_roi)
    # m.debug_exaggerate_exposure(5)

    if 'pois' in args.model:
        model_type = 'pois'
    else:
        model_type = 'np'

    if args.fit_type == 'svi':
        m.fit_svi(
            model_name=model_type, n_steps=args.n_step, data=data_in, lr=args.lr,
            rng_key=jax.random.PRNGKey(args.seed),
            guide=args.guide, num_flows=args.num_flows, hidden_dims=[args.hidden_dim_n, args.hidden_dim_n],
            num_particles=args.n_par, vectorize_particles=True,
            renyi_alpha=args.renyi_alpha, lr_exp_decay=False,
        )
        samples = m.get_svi_samples(num_samples=args.n)

    elif args.fit_type in ['hmc', 'hmcnt']:

        if args.fit_type == 'hmcnt':
            model_type = 'neutra'
            m.fit_svi(
                model_name=model_type, n_steps=args.n_step, data=data_in, lr=1e-4,
                rng_key=jax.random.PRNGKey(args.seed)
            )
            
        mcmc = m.run_nuts(
            model_name=model_type, num_chains=4, num_warmup=1000, num_samples=args.n//4, step_size=0.05,
            data=data_in,
            rng_key=jax.random.PRNGKey(args.seed)
        )
        samples = mcmc.get_samples()

    elif args.fit_type in ['pthmc']:
        m.fit_svi(model_name=model_type, n_steps=args.n_step, data=data_in, lr=1e-4)
        mcmc = m.run_parallel_tempering_hmc(
            model_name=model_type,
            num_samples=args.n,
            step_size_base=5e-2,
            num_leapfrog_steps=3,
            num_adaptation_steps=600,
        )
        samples = mcmc.get_samples()

    elif args.fit_type == 'testhmc':
        mcmc = m.run_nuts(
            model_name=model_type, num_chains=8, num_warmup=10, num_samples=10, step_size=0.1,
            use_neutra=False, data=data_in
        )
        samples = mcmc.get_samples()

    else:
        raise NotImplementedError(args.fit_type)
    
    pickle.dump(samples, open(f"{save_dir}/i{args.i}_n{args.n}_ns{args.n_step}.p", 'wb'))
    if args.fit_type == 'svi':
        pickle.dump(m.svi_results.losses, open(f"{save_dir}/i{args.i}_n{args.n}_ns{args.n_step}_losses.p", 'wb'))
