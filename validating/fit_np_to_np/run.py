import os
import sys
import pickle
import argparse

import numpy as np
import healpy as hp
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

    data_dir = "/n/home07/yitians/fermi/fermi-prob-prog/outputs/np_np_dsk/counts"
    save_dir = "/n/home07/yitians/fermi/fermi-prob-prog/outputs/np_np_dsk/Sps-shape"
    os.makedirs(save_dir, exist_ok=True)

    # model = NPModel(
    #     non_poissonian=True, l_max=2,
    #     dif_names=["ModelO", "ModelA", "ModelF"],
    #     bulge_hybrid=True,
    #     bulge_template_names=["mcdermott2022", "mcdermott2022_bbp", "mcdermott2022_x", "macias2019", "coleman2019"],
    #     vary_gamma=True,
    #     vary_disk=True,
    #     ps_cat="3fgl", r_outer=25, band_mask_range=2.,
    #     nside=128, n_exp=1, debug_model=False,
    # )
    model = NPModelDebug(
        vary_gamma=False,
        vary_disk=True,
        use_flat_exposure=True,
    )
    print('loaded model.', flush=True)

    counts = np.zeros(hp.nside2npix(model.nside), dtype=jnp.int32)
    counts[~model.normalization_mask] = np.load(f"{data_dir}/counts_{args.i}.npy")
    counts = jnp.asarray(counts, dtype=jnp.int32)
    model.data = counts
    print('loaded counts.', flush=True)

    rng_key = jax.random.PRNGKey(42 + 4242 * args.i)

    run_mode = 'svi' # 'svi' or 'hmc' or 'hmc-neutra'

    if run_mode in ['svi', 'hmc-neutra']:
        if run_mode == 'hmc-neutra':
            n_steps = 5000
        else:
            n_steps = 5000
        # svi
        rng_key, key = jax.random.split(rng_key)
        svi_results = model.fit_svi(
            rng_key,
            guide='iaf', num_flows=5, hidden_dims=[512, 512],
            n_steps=n_steps, lr=5e-4, num_particles=8, data=counts
        )
        print('fit complete.', flush=True)
        rng_key, key = jax.random.split(rng_key)
        svi_samples = model.get_svi_samples(key, num_samples=50000)
        pickle.dump(svi_samples, open(f"{save_dir}/svi_samples_{args.i}.p", "wb"))
        pickle.dump(svi_results, open(f"{save_dir}/svi_results_{args.i}.p", "wb"))
        print('svi samples generated', flush=True)

    elif run_mode in ['hmc', 'hmc-neutra']:
        # hmc
        rng_key, key = jax.random.split(rng_key)
        mcmc = model.run_nuts(
            num_chains=4, num_warmup=1000, num_samples=2000, step_size=0.01,
            rng_key=key, use_neutra=(run_mode=='hmc-neutra'), data=model.data
        )
        rng_key, key = jax.random.split(rng_key)
        hmc_samples = model.expand_samples(mcmc.get_samples())
        pickle.dump(hmc_samples, open(f"{save_dir}/hmc_samples_{args.i}.p", "wb"))
        print('hmc samples generated', flush=True)

    print('complete!')
