import os
import sys
import pickle

import numpy as np
import jax
import jax.numpy as jnp

sys.path.append("..")
from models.np_model import NPModel

os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"
print(jax.devices())


if __name__ == "__main__":

    # config
    run_dir = "../outputs/np_sim/230830"

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
    counts = jnp.array(np.load(f"{run_dir}/counts_0.npy"), dtype=jnp.int32)
    model.data = counts

    # svi
    rng_key = jax.random.PRNGKey(42)
    rng_key, key = jax.random.split(rng_key)
    svi_results = model.fit_svi(
        rng_key,
        guide='iaf', num_flows=5, hidden_dims=[512, 512],
        n_steps=20000, lr=5e-5, num_particles=16, data=counts
    )
    pickle.dump(svi_results, open(f"{run_dir}/svi_results_0.p", "wb"))
    #model.svi_results = pickle.load(open(f"{run_dir}/svi_results_0.p", "rb"))

    # hmc
    rng_key, key = jax.random.split(rng_key)
    mcmc = model.run_nuts(
        num_chains=4, num_warmup=500, num_samples=20000, step_size=0.01,
        rng_key=key, use_neutra=True, data=counts
    )
    rng_key, key = jax.random.split(rng_key)
    pickle.dump(mcmc.get_samples(), open(f"{run_dir}/hmc_samples_0.p", "wb"))