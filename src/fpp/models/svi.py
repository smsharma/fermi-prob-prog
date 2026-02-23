import jax
import jax.numpy as jnp
from jax import jit, lax
import tqdm
from numpyro.infer.svi import SVIRunResult


def beta_schedule(num_steps, start_T=10.0, end_T=1.0, kind="geometric"):
    """
    Returns betas in (0,1], where beta = 1/T.
    Geometric is robust; cosine is a smooth alternative.
    """
    assert start_T >= end_T >= 1.0
    if kind == "geometric":
        # T_t = start_T * (end_T/start_T)^(t/(num_steps-1))
        t = jnp.linspace(0, 1, num_steps)
        T = start_T * (end_T / start_T) ** t
    elif kind == "cosine":
        # T_t = end_T + 0.5*(start_T-end_T)*(1+cos(pi * (1 - t)))
        t = jnp.linspace(0, 1, num_steps)
        T = end_T + 0.5 * (start_T - end_T) * (1.0 + jnp.cos(jnp.pi * (1.0 - t)))
    else:
        raise ValueError("kind must be 'geometric' or 'cosine'")
    return 1.0 / T  # beta = 1/T

def run_svi_with_beta(
    svi,
    rng_key,
    num_steps,
    *args,
    progress_bar=True,
    stable_update=False,
    init_state=None,
    init_params=None,
    tempering_schedule="none", # "none", "geometric" or "cosine"
    **kwargs,
):

    if num_steps < 1:
        raise ValueError("num_steps must be a positive integer.")

    if tempering_schedule == "none":
        betas = jnp.ones(num_steps)
    else:
        betas = beta_schedule(num_steps, kind=tempering_schedule)

    def body_fn(svi_state, beta, _):
        if stable_update:
            svi_state, loss = svi.stable_update(
                svi_state,
                *args,
                beta=beta,
                **kwargs,
            )
        else:
            svi_state, loss = svi.update(
                svi_state,
                *args,
                beta=beta,
                **kwargs,
            )
        return svi_state, loss

    if init_state is None:
        svi_state = svi.init(rng_key, *args, init_params=init_params, **kwargs)
    else:
        svi_state = init_state
    if progress_bar:
        losses = []
        with tqdm.trange(1, num_steps + 1) as t:
            batch = max(num_steps // 20, 1)
            for i in t:
                beta = betas[i - 1]
                svi_state, loss = jit(body_fn)(svi_state, beta, None)
                losses.append(jax.device_get(loss))
                if i % batch == 0:
                    if stable_update:
                        valid_losses = [x for x in losses[i - batch :] if x == x]
                        num_valid = len(valid_losses)
                        if num_valid == 0:
                            avg_loss = float("nan")
                        else:
                            avg_loss = sum(valid_losses) / num_valid
                    else:
                        avg_loss = sum(losses[i - batch :]) / batch
                    t.set_postfix_str(
                        "init loss: {:.4f}, avg. loss [{}-{}]: {:.4f}, beta: {:.4f}".format(
                            losses[0], i - batch + 1, i, avg_loss, beta
                        ),
                        refresh=False,
                    )
        losses = jnp.stack(losses)
    else:
        svi_state, losses = lax.scan(body_fn, svi_state, None, length=num_steps)

    return SVIRunResult(svi.get_params(svi_state), svi_state, losses)