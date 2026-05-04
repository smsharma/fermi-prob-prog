import os
import dill as pickle
import argparse

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from fpp.models.np_model import NPModel


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fit', type=str, choices=['svi', 'hmc'], required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--init', type=str, default='none') # none map
    parser.add_argument('--comment', type=str, default='')
    args = parser.parse_args()

    subname = f'{args.fit}'
    if args.comment != '':
        subname += f"-{args.comment}"
    print(f"Running {subname} ...")

    wdir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.environ['MYSTORE'] + f"/fermi/fermi-prob-prog/outputs/production/fits/fermi"
    os.makedirs(save_dir, exist_ok=True)

    data = jnp.array(np.load(f"{wdir}/../data/fermi_data_573w/fermi_data_128/fermidata_counts.npy"), dtype=jnp.int32)
    
    m = NPModel(data=data)

    if args.fit == 'svi':
        m.fit_svi(
            data=data, rng_key=jax.random.PRNGKey(args.seed),
            n_steps=10000, lr=3e-4,
            guide='iaf', num_flows=5, hidden_dims=[128, 128],
            num_particles=16,
        )
        samples = m.get_svi_samples(num_samples=50000)

    elif args.fit == 'hmc':
        if args.init == 'map':
            print("Initializing from MAP estimate...")
            m.get_map_estimate(data=data)
            init_params = m.map_estimate
            print("MAP estimate:", init_params)
        else:
            init_params = None

        m.run_nuts(
            data=data, rng_key=jax.random.PRNGKey(args.seed),
            num_chains=4, num_warmup=1000, num_samples=30000//4, step_size=0.05, max_tree_depth=4,
            init_params=init_params
        )
        samples = m.nuts_mcmc.get_samples()
    
    pickle.dump(samples, open(f"{save_dir}/{subname}.p", 'wb'))
