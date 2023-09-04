import os
import sys
import pickle
import argparse

import numpy as np
import jax
import jax.numpy as jnp

sys.path.append("../..")
from models.np_model import NPModel

os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"
print(jax.devices())


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', type=int)
    # args = parser.parse_args()
    # print(f'Running i={args.i}...')

    # config
    run_dir = "/n/home07/yitians/fermi/fermi-prob-prog/outputs/np_p_230827"

    model = NPModel(
        non_poissonian=True, l_max=2,
        dif_names=["ModelO", "ModelA", "ModelF"],
        bulge_hybrid=True,
        bulge_template_names=["mcdermott2022", "mcdermott2022_bbp", "mcdermott2022_x", "macias2019", "coleman2019"],
        vary_gamma=True,
        vary_disk=True,
        ps_cat="3fgl", r_outer=25, band_mask_range=2.,
        nside=128, n_exp=1, debug_model=False,
    )

    for i in range(1, 10):
        print(f"Running i={i}...")

        counts = jnp.array(np.load(f"{run_dir}/counts_{i}.npy"), dtype=jnp.int32)
        model.data = counts

        # svi
        rng_key = jax.random.PRNGKey(42 * i)
        rng_key, key = jax.random.split(rng_key)
        svi_results = model.fit_svi(
            rng_key,
            guide='iaf', num_flows=5, hidden_dims=[256, 256],
            n_steps=10000, lr=5e-5, num_particles=16, data=counts
        )
        rng_key, key = jax.random.split(rng_key)
        svi_samples = model.get_svi_samples(key, num_samples=50000)
        pickle.dump(svi_results, open(f"{run_dir}/svi_results_{i}.p", "wb"))
        pickle.dump(svi_samples, open(f"{run_dir}/svi_samples_{i}.p", "wb"))

    # hmc
    # rng_key, key = jax.random.split(rng_key)
    # mcmc = model.run_nuts(
    #     num_chains=4, num_warmup=500, num_samples=200, step_size=0.01,
    #     rng_key=key, use_neutra=True, data=counts
    # )
    # rng_key, key = jax.random.split(rng_key)
    # hmc_samples = model.expand_samples(mcmc.get_samples())
    # pickle.dump(hmc_samples, open(f"{run_dir}/hmc_samples_{args.i}.p", "wb"))