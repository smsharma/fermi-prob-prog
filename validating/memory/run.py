import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import jax.profiler

sys.path.append("../..")
from models.np_model import NPModel

os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"
print(jax.devices())


if __name__ == "__main__":

    run_dir = "/n/home07/yitians/fermi/fermi-prob-prog/outputs/np_np"

    counts = jnp.array(np.load(f"{run_dir}/counts_0.npy"), dtype=jnp.int32)
    print('loaded counts.', flush=True)

    model = NPModel(
        non_poissonian=True, l_max=0,
        dif_names=["ModelO", "ModelA"],
        bulge_hybrid=True,
        bulge_template_names=["mcdermott2022", "mcdermott2022_bbp"],
        vary_gamma=False,
        vary_disk=False,
        ps_cat="3fgl", r_outer=25, band_mask_range=2.,
        nside=128, n_exp=1, debug_model=False,
        data=counts,
    )
    print('loaded model.', flush=True)

    # svi
    rng_key = jax.random.PRNGKey(42)
    rng_key, key = jax.random.split(rng_key)
    svi_results = model.fit_svi(
        rng_key,
        guide='iaf', num_flows=5, hidden_dims=[128, 128],
        n_steps=10000, lr=5e-5, num_particles=32, vectorize_particles=False,
        data=counts
    )

    svi_results.losses.block_until_ready()

    jax.profiler.save_device_memory_profile("memory.prof")
    print('fit complete.', flush=True)