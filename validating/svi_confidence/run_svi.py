import os
import sys
import argparse
import pickle

import numpy as np
import jax

sys.path.append("../..")
from models.validation_models import PoissonModel

os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"
print(jax.devices())


if __name__ == "__main__":

    # parse n_hidden, n_steps, num_particles
    parser = argparse.ArgumentParser()
    parser.add_argument("--nhidden", type=int, default=512)
    parser.add_argument("--nsteps", type=int, default=10000)
    parser.add_argument("--npar", type=int, default=64)
    args = parser.parse_args()

    base_dir = "/n/home07/yitians/fermi/fermi-prob-prog/outputs/poiss_sim"
    save_dir = f"{base_dir}/svi_230826_nhidden{args.nhidden}_nsteps{args.nsteps}_npar{args.npar}"
    os.makedirs(save_dir, exist_ok=True)
    
    n_run = 30
    model = PoissonModel()
    truth_arr = np.load(f"{base_dir}/truth_arr_230826.npy")
    model.set_truth(truth_arr)

    for i_run in range(n_run):
        rng_key = jax.random.PRNGKey(42*i_run)
        rng_key, key = jax.random.split(rng_key)
        counts = model.generate_counts(rng_key)
        
        print(f"Running index={i_run}...")
        rng_key, key = jax.random.split(rng_key)
        model.fit_svi(key, guide='iaf', num_flows=5, hidden_dims=[args.nhidden, args.nhidden],
                      n_steps=args.nsteps, lr=5e-5, num_particles=args.npar, counts=counts)
        rng_key, key = jax.random.split(rng_key)
        samples = model.get_svi_samples(50000, key)
        pickle.dump(samples, open(f"{save_dir}/samples_{i_run}.p", "wb"))
        pickle.dump(model.svi_results, open(f"{save_dir}/svi_results_{i_run}.p", "wb"))