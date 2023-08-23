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
    parser.add_argument("--use_neutra", action="store_true", help="use_neutra")
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
    print(f"Running index={args.index} with use_neutra={args.use_neutra}...")
    if args.use_neutra:
        rng_key, key = jax.random.split(rng_key)
        model.fit_svi(key, guide='iaf', num_flows=5, hidden_dims=[128, 128],
                      n_steps=2500, lr=5e-5, num_particles=8, counts=counts)
    rng_key, key = jax.random.split(rng_key)
    mcmc = model.run_nuts(
        num_chains=4, num_warmup=500, num_samples=20000, step_size=0.01,
        rng_key=key, use_neutra=args.use_neutra, counts=counts
    )
    rng_key, key = jax.random.split(rng_key)
    pickle.dump(mcmc.get_samples(), open(f"{save_dir}/samples_{args.index}.p", "wb"))