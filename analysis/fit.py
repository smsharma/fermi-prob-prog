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
    parser.add_argument('-i', type=int)
    parser.add_argument('-n', type=int)
    parser.add_argument('--data', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--n_exp', type=int)
    parser.add_argument('--n_step', type=int, default=0)
    parser.add_argument('--fit_type', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lrexpdecay', action='store_true')
    parser.add_argument('--guide', type=str, default='iaf')
    parser.add_argument('--n_par', type=int, default=8)
    parser.add_argument('--renyi_alpha', type=float, default=1)
    parser.add_argument('--tempering_schedule', type=str, default='none')
    parser.add_argument('--num_flows', type=int, default=4)
    parser.add_argument('--hidden_dim_n', type=int, default=64)
    parser.add_argument('--comment', type=str, default='')
    args = parser.parse_args()

    comment_str = '' if args.comment == '' else f"_{args.comment}"
    run_name = f"{args.fit_type}_D{args.data}_M{args.model}" + comment_str
    print('run_name:', run_name)

    save_dir = f"{wdir}/../outputs/fits/{run_name}"
    os.makedirs(save_dir, exist_ok=True)

    #===== masks =====
    mask_roi = np.load(f"{wdir}/mask_roi.npy")
    mask_norm = jnp.load(f"{wdir}/mask_norm.npy")

    # Ensure mask_roi's length is divisible by args.n_exp
    n_pix_remainder = int(np.sum(~mask_roi)) % args.n_exp
    print(f'Pixel number: {int(np.sum(~mask_roi))}', end=' ')
    if n_pix_remainder != 0:
        unmasked_indices = np.where(mask_roi == 0)[0]
        mask_roi[unmasked_indices[-n_pix_remainder:]] = 1
    print(f'-> {int(np.sum(~mask_roi))} = {args.n_exp} * {int(np.sum(~mask_roi) / args.n_exp)}')

    #===== data =====
    data = np.load(f"../outputs/simulations/{args.data}.npy")[args.i]
    if len(data) < hp.nside2npix(128):
        data_full = np.zeros(hp.nside2npix(128))
        data_full[~mask_norm] = data
        data_in = jnp.array(data_full, dtype=jnp.int32)
    else:
        data_in = jnp.array(data, dtype=jnp.int32)

    #===== model =====
    if 'A6' in args.model:
        sphold = True
    else:
        sphold = False
    psf_tag = 'delta' if 'deltapsf' in args.model else 'king'
    print('PSF:', psf_tag)
    if args.model == 'base23fixO':
        dif_names = ["ModelO"]
    elif args.model == 'base23fixA':
        dif_names = ["ModelA"]
    elif args.model == 'base23fixF':
        dif_names = ["ModelF"]
    else:
        dif_names = ["ModelO", "ModelA", "ModelF"]
    m = NPModel(data=data_in, psf_tag=psf_tag, n_exp=args.n_exp, diffuse_names=dif_names, debug_old_sphh=sphold)
    # m.debug_exaggerate_exposure(5)

    #===== fit =====
    if args.fit_type == 'svi':
        m.fit_svi(
            n_steps=args.n_step, data=data_in, lr=args.lr,
            rng_key=jax.random.PRNGKey(args.seed),
            guide=args.guide, num_flows=args.num_flows, hidden_dims=[args.hidden_dim_n, args.hidden_dim_n],
            num_particles=args.n_par,
            renyi_alpha=args.renyi_alpha, lr_exp_decay=args.lrexpdecay,
            tempering_schedule=args.tempering_schedule
        )
        samples = m.get_svi_samples(num_samples=args.n)

    elif args.fit_type in ['hmc', 'hmcnt']:

        if args.fit_type == 'hmcnt':
            use_neutra = True
            m.fit_svi(n_steps=args.n_step, data=data_in, lr=1e-4, rng_key=jax.random.PRNGKey(args.seed))
        else:
            use_neutra = False
            
        mcmc = m.run_nuts(
            use_neutra=use_neutra, num_chains=4, num_warmup=1000, num_samples=args.n//4, step_size=0.05,
            data=data_in,
            rng_key=jax.random.PRNGKey(args.seed)
        )
        samples = mcmc.get_samples()

    elif args.fit_type in ['pthmc']:
        m.fit_svi(n_steps=args.n_step, data=data_in, lr=1e-4, rng_key=jax.random.PRNGKey(args.seed))
        mcmc = m.run_parallel_tempering_hmc(
            num_samples=args.n,
            step_size_base=5e-2,
            num_leapfrog_steps=3,
            num_adaptation_steps=600,
        )
        samples = mcmc.get_samples()

    else:
        raise NotImplementedError(args.fit_type)
    
    pickle.dump(samples, open(f"{save_dir}/i{args.i}_n{args.n}_ns{args.n_step}.p", 'wb'))
    if args.fit_type == 'svi':
        pickle.dump(m.svi_results.losses, open(f"{save_dir}/i{args.i}_n{args.n}_ns{args.n_step}_losses.p", 'wb'))
