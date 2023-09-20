import os
import sys
import pickle
import argparse

import numpy as np
import jax
import jax.numpy as jnp

sys.path.append("../..")
from models.np_model_debug import NPModelDebug

os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"
print(jax.devices())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=int)
    args = parser.parse_args()

    print(f'running i={args.i}...', flush=True)

    run_dir = "/n/home07/yitians/fermi/fermi-prob-prog/outputs/np_np_simple_0920"

    model = NPModelDebug()
    print('loaded model.', flush=True)

    counts = jnp.array(np.load(f"{run_dir}/counts_{args.i}.npy"), dtype=jnp.int32)
    model.data = counts
    print('loaded counts.', flush=True)

    # svi
    rng_key = jax.random.PRNGKey(424242 + 42 * args.i)
    rng_key, key = jax.random.split(rng_key)
    svi_results = model.fit_svi(
        rng_key,
        num_flows=5, hidden_dims=[512, 512],
        n_steps=20000, lr=5e-3, num_particles=32, data=counts
    )
    print('fit complete.', flush=True)
    rng_key, key = jax.random.split(rng_key)
    svi_samples = model.get_svi_samples(key, num_samples=50000)
    pickle.dump(svi_results, open(f"{run_dir}/svi_results_{args.i}_lr5e-3.p", "wb"))
    pickle.dump(svi_samples, open(f"{run_dir}/svi_samples_{args.i}_lr5e-3.p", "wb"))
    print('svi samples generated', flush=True)

    # if args.i == 0:
    #     # hmc
    #     rng_key, key = jax.random.split(rng_key)
    #     mcmc = model.run_nuts(
    #         num_chains=4, num_warmup=500, num_samples=20000, step_size=0.01,
    #         rng_key=key, use_neutra=True, data=counts
    #     )
    #     rng_key, key = jax.random.split(rng_key)
    #     hmc_samples = model.expand_samples(mcmc.get_samples())
    #     pickle.dump(hmc_samples, open(f"{run_dir}/hmc_samples_{args.i}.p", "wb"))
    #     print('  hmc samples generated', flush=True)

    print('complete!')