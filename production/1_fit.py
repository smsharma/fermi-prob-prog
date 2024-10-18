import sys
import dill as pickle
import argparse

import numpy as np
import healpy as hp

import jax.numpy as jnp

wdir = "/n/home07/yitians/fermi/fermi-prob-prog/production"
sys.path.append(f"{wdir}/..")
from models.np_model import NPModel


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=int, default=0)
    parser.add_argument('-n', type=int)
    parser.add_argument('--data', type=str)
    parser.add_argument('--n_step', type=int)
    parser.add_argument('--fit_type', type=str)
    args = parser.parse_args()

    wdir = "/n/home07/yitians/fermi/fermi-prob-prog/production"
    save_dir = f"{wdir}/../outputs/fit/{args.fit_type}_{args.data}_ns{args.n_step}"

    mask_roi = jnp.load(f"{wdir}/mask_roi.npy")
    mask_norm = jnp.load(f"{wdir}/mask_norm.npy")

    data = np.load(f"../outputs/sims/sim_{args.data}_n30.npy")[args.i]
    data_full = np.zeros(hp.nside2npix(128))
    data_full[~mask_norm] = data
    data_in = jnp.array(data_full, dtype=jnp.int32)


    m = NPModel()

    if args.fit_type == 'svi':
        m.fit_svi(n_steps=args.n_step, data=data_in)
        samples = m.get_svi_samples(num_samples=args.n)

    elif args.fit_type in ['hmc', 'hmcnt']:
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
    
    pickle.dump(samples, open(f"{save_dir}/i{args.i}_n{args.n}.p", 'wb'))
