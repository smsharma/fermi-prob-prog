import sys

import numpy as np
import jax

sys.path.append("../..")
from models.validation_models import PoissonModel


if __name__ == "__main__":

    save_dir = "../../outputs/poiss_sim/hmc_230823"

    n_temp = 11
    truth_arr = jax.random.normal(jax.random.PRNGKey(42), shape=(n_temp,)) * 0.5 + 3

    np.save(f"{save_dir}/truth_arr.npy", truth_arr)