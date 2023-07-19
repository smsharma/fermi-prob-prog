import os
import sys
import pickle
from tqdm import tqdm

import numpy as np
import arviz as az
import healpy as hp

import jax
import jax.numpy as jnp

print(jax.devices())
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"

wdir = "/n/home07/yitians/fermi/fermi-prob-prog"

sys.path.append(wdir)
from models.np_model import NPModel


save_dir = f"{wdir}/outputs/poisson_sim/run_230718"

npmodel = NPModel(
    non_poissonian=True,
    l_max=2,
    vary_gamma=True,
    bulge_hybrid=True,
    ps_cat="3fgl",
    nside=128,
)


for i in [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 78, 79]:
    counts = jnp.array(np.load(f"{save_dir}/counts_{i}.npy"), dtype=jnp.int32)
    svi_results = npmodel.fit_svi(
        rng_key=jax.random.PRNGKey(4234),
        n_steps=2000,
        guide="iaf",
        lr=5e-5,
        num_particles=8,
        data=jnp.array(counts),
    )
    pickle.dump(svi_results, open(f"{save_dir}/svi_results_{i}.p", 'wb'))
    samples = npmodel.get_svi_samples(
        rng_key=jax.random.PRNGKey(42),
        num_samples=50000,
    )
    pickle.dump(samples, open(f"{save_dir}/samples_{i}.p", 'wb'))