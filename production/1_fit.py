import os
import sys
import dill as pickle
import argparse

import numpy as np
import healpy as hp

import jax.numpy as jnp

wdir = "/n/home07/yitians/fermi/fermi-prob-prog/production"
sys.path.append(f"{wdir}/..")
from models.np_model_single import NPModelSingle


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=int, default=0)
    parser.add_argument('-n', type=int, default=10000)
    parser.add_argument('--n_step', type=int, default=2000)
    parser.add_argument('--fit_type', type=str, default='test')
    args = parser.parse_args()

    wdir = "/n/home07/yitians/fermi/fermi-prob-prog/production"
    #data_dir = f"{wdir}/../data/fermi_data_573w/fermi_data_128"
    data_dir = f"{wdir}/../outputs/simulations"
    save_dir = f"{wdir}/../outputs/fit/svi_240818.2_kingpsf"
    os.makedirs(save_dir, exist_ok=True)

    mask_roi = jnp.load(f"{wdir}/mask_roi.npy")
    mask_norm = jnp.load(f"{wdir}/mask_norm.npy")

    data = np.load(f"{data_dir}/sim_Spsgce_n100.npy")[args.i]
    data_full = np.zeros(hp.nside2npix(128))
    data_full[~mask_norm] = data
    data_in = jnp.array(data_full, dtype=jnp.int32)


    m = NPModelSingle()

    if args.fit_type == 'svi':
        m.fit_svi(n_steps=args.n_step, data=data_in)
        samples = m.get_svi_samples(num_samples=args.n)

    elif args.fit_type in ['hmc', 'hmcnt']:
        if args.fit_type == 'hmcnt':
            m.fit_svi(n_steps=args.n_step, data=data_in)
        mcmc = m.run_nuts(
            num_chains=4, num_warmup=500, num_samples=args.n//4,
            use_neutra=(args.fit_type=='hmcnt'), data=data_in
        )
        samples = mcmc.get_samples()

    elif args.fit_type == 'test':
        mcmc = m.run_nuts(
            num_chains=4, num_warmup=10, num_samples=10,
            use_neutra=False, data=data_in
        )
        samples = mcmc.get_samples()

    else:
        raise NotImplementedError(args.fit_type)
    
    pickle.dump(samples, open(f"{save_dir}/{args.fit_type}_samples_i{args.i}_n{args.n}_ns{args.n_step}.p", 'wb'))
