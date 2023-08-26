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

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--index", type=int, required=True, help="index")
    args = parser.parse_args()

    # config
    save_dir = "../../outputs/poiss_sim/hmc_230823"

    model = PoissonModel()
    truth_arr = np.load(f"{save_dir}/truth_arr.npy")
    model.set_truth(truth_arr)

    # init
    rng_key = jax.random.PRNGKey(42*args.index)
    rng_key, key = jax.random.split(rng_key)
    counts = model.generate_counts(rng_key)
    
    # run
    print(f"Running index={args.index}...")
    rng_key, key = jax.random.split(rng_key)
    model.fit_svi(key, guide='iaf', num_flows=5, hidden_dims=[128, 128],
                  n_steps=2500, lr=5e-5, num_particles=8, counts=counts)
    rng_key, key = jax.random.split(rng_key)
    samples = model.get_svi_samples(50000, key)
    pickle.dump(samples, open(f"{save_dir}/samples_{args.index}.p", "wb"))