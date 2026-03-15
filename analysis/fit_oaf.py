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
    parser.add_argument('-i', type=int) # 0-7
    parser.add_argument('--i_data', type=int)
    args = parser.parse_args()

    fit_list = ['hmc']
    dif_list = [["ModelO"], ["ModelO", "ModelA", "ModelF"]]
    i_fit, i_dif = np.unravel_index(args.i, (len(fit_list), len(dif_list)))
    fit = fit_list[i_fit]
    dif = dif_list[i_dif]
    subname = fit + '-' + ['O', 'A', 'F', 'OAF'][i_dif]
    print(f"Running {subname} ...")

    save_dir = f"{wdir}/../outputs/production/fits/oaf"
    os.makedirs(save_dir, exist_ok=True)

    data = jnp.array(np.load(f"../outputs/production/simulations/23new_p6v11.npy")[args.i_data], dtype=jnp.int32)
    
    m = NPModel(
        data=data,
        psf_tag='king',
        n_exp=7,
        diffuse_names=dif,
    )

    if fit == 'svi':
        m.fit_svi(
            data=data, rng_key=jax.random.PRNGKey(42),
            n_steps=10000, lr=3e-4,
            guide='iaf', num_flows=5, hidden_dims=[128, 128],
            num_particles=16,
        )
        samples = m.get_svi_samples(num_samples=50000)

    elif fit == 'hmc':
        m.run_nuts(
            data=data, rng_key=jax.random.PRNGKey(42),
            num_chains=4, num_warmup=1000, num_samples=30000//4, step_size=0.05,
        )
        samples = m.nuts_mcmc.get_samples()
    
    pickle.dump(samples, open(f"{save_dir}/{subname}-{args.i_data}.p", 'wb'))
