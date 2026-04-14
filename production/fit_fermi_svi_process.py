"""
Fit Fermi data with SVI, saving the SVI state (guide params) at specified
steps so that the evolving posterior/guide can be visualized later.

Saves to: outputs/svi_process/
    - svi_params_step{step}.p   : guide params at each checkpoint
    - svi_losses.npy            : full loss array
    - svi_save_steps.npy        : array of checkpoint steps

Usage:
    python fit_fermi_svi_process.py [--seed 42] [--n_steps 10000] [--save_steps 0,100,500,1000,2000,5000,10000]
"""

import os
import dill as pickle
import argparse

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
import tqdm

import optax
from numpyro.infer import SVI, Trace_ELBO, autoguide
from numpyro import optim
from jax.example_libraries import stax

from fpp.models.np_model import NPModel


def run_svi_saving_states(
    svi, rng_key, num_steps, save_step_arr,
    *args, **kwargs,
):
    """Run SVI training loop, saving guide params at specified steps.

    Args:
        svi: numpyro SVI object.
        rng_key: JAX random key.
        num_steps: Total number of SVI steps.
        save_step_arr: List/array of step numbers at which to save params.
            Step 0 means the initial (untrained) state.
        *args, **kwargs: Forwarded to svi.update / svi.init (e.g. data=...).

    Returns:
        saved_params: dict mapping step -> params dict.
        losses: array of losses at each step.
        final_svi_state: the SVI state after the last step.
    """
    save_step_set = set(int(s) for s in save_step_arr)

    svi_state = svi.init(rng_key, *args, **kwargs)
    saved_params = {}

    # Save initial state (step 0) if requested
    if 0 in save_step_set:
        saved_params[0] = jax.device_get(svi.get_params(svi_state))

    betas = jnp.ones(num_steps)  # no tempering

    @jit
    def body_fn(svi_state, beta):
        svi_state, loss = svi.update(svi_state, *args, beta=beta, **kwargs)
        return svi_state, loss

    losses = []
    with tqdm.trange(1, num_steps + 1) as t:
        batch = max(num_steps // 20, 1)
        for i in t:
            beta = betas[i - 1]
            svi_state, loss = body_fn(svi_state, beta)
            losses.append(jax.device_get(loss))

            # Save checkpoint
            if i in save_step_set:
                saved_params[i] = jax.device_get(svi.get_params(svi_state))

            if i % batch == 0:
                avg_loss = sum(losses[i - batch:]) / batch
                t.set_postfix_str(
                    "init loss: {:.4f}, avg. loss [{}-{}]: {:.4f}".format(
                        losses[0], i - batch + 1, i, avg_loss
                    ),
                    refresh=False,
                )

    losses = np.array(losses)
    return saved_params, losses, svi_state


if __name__ == '__main__':

    save_step_arr = np.arange(0, 300, 10).tolist() + np.arange(300, 1000, 50).tolist() + np.arange(1000, 5000, 200).tolist()
    print(f"Will save SVI state at steps: {save_step_arr}")

    wdir = os.path.dirname(os.path.abspath(__file__))
    save_dir = f"{wdir}/../outputs/svi_process"
    os.makedirs(save_dir, exist_ok=True)

    # --- Load data and initialize model (same as fit_fermi.py) ---
    data = jnp.array(
        np.load(f"{wdir}/../data/fermi_data_573w/fermi_data_128/fermidata_counts.npy"),
        dtype=jnp.int32,
    )

    m = NPModel(
        data=data,
        psf_tag='king',
        n_exp=7,
        diffuse_names=["ModelO", "ModelA", "ModelF"],
    )

    # --- Set up SVI components (mirrors NPModel.fit_svi defaults from fit_fermi.py) ---
    guide = autoguide.AutoIAFNormal(
        m.model,
        num_flows=5,
        hidden_dims=[128, 128],
        nonlinearity=stax.Tanh,
    )
    m.guide = guide  # attach for later sampling

    optimizer = optim.optax_to_numpyro(optax.chain(
        optax.clip(1.),
        optax.adam(3e-4),
    ))

    loss = Trace_ELBO(num_particles=16, vectorize_particles=True)

    svi = SVI(m.model, guide, optimizer, loss)

    # --- Run SVI, saving states ---
    saved_params, losses, final_state = run_svi_saving_states(
        svi,
        rng_key=jax.random.PRNGKey(42),
        num_steps=5000,
        save_step_arr=save_step_arr,
        data=data,
    )

    # --- Save outputs ---
    for step, params in saved_params.items():
        fn = f"{save_dir}/svi_params_step{step}.p"
        pickle.dump(params, open(fn, 'wb'))
        print(f"Saved params at step {step} -> {fn}")

    np.save(f"{save_dir}/svi_losses.npy", losses)
    np.save(f"{save_dir}/svi_save_steps.npy", np.array(sorted(saved_params.keys())))

    print(f"Done. Losses saved to {save_dir}/svi_losses.npy")
    print(f"Save steps: {sorted(saved_params.keys())}")
