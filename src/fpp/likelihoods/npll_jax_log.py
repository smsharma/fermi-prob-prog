import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
from jax import jit, vmap
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.numpy import logaddexp

from functools import partial

from fpp.models.scd import dnds as dnds_func
from fpp.utils.utils import jnp_trapezoid


def log_trapz(logf, x, axis=-1):
    """Compute log of trapezoidal integration of f over x"""

    dx = x[1:] - x[:-1]
    logf_moved = jnp.moveaxis(logf, axis, -1)
    logf_avg = logaddexp(logf_moved[..., 1:], logf_moved[..., :-1]) - jnp.log(2.0)

    return logsumexp(a=logf_avg, b=dx, axis=-1)


# NPT = number of non-poissonian templates
# P   = number of pixels
# F   = number of f bins
# S   = number of s bins
# M   = number of counts (k_max + 1)


@partial(jit, static_argnums=(6,))
def return_x_m(f, rho_df, npt, data, s, dnds, k_max):
    """
    Args:
        f      (F,) jnp.array: f values
        rho_df (F,) jnp.array: rho * df values
        npt    (P,) jnp.array: non-poissonian template values
        data   (P,) jnp.array: data counts map
        s      (S,) jnp.array: s values
        dnds   (S,) jnp.array: dnds values
        k_max       int      : maximum number of counts to consider
    """

    m = jnp.arange(k_max + 1 , dtype=jnp.float64) # (M,)
    lgamma = jax.lax.lgamma(m + 1) # (M,)

    # (F, S, M)
    log_s_integrand = (jnp.log(dnds) - jnp.outer(f, s))[:, :, None] + jnp.log(jnp.outer(f, s))[:, :, None] * m[None, None, :] - lgamma
    f_integrand = log_trapz(log_s_integrand, s, axis=1) # (F, M)
    x_m_unnorm = jnp.exp(logsumexp(a=f_integrand, b=rho_df[:, None], axis=0)) # (M,)
    x_m = jnp.outer(npt, x_m_unnorm) # (P, M)

    x_m_sum_unnorm = jnp.sum(rho_df) * jnp_trapezoid(dnds, s) # ()
    x_m_sum = npt * x_m_sum_unnorm - x_m[:, 0] # (P,)

    return x_m, x_m_sum # (P, M), (P,)


# vmap over NPT
dnds_func_vmap = vmap(dnds_func, in_axes=(None, 0))
return_x_m_vmap = vmap(return_x_m, in_axes=(None, None, 0, None, None, 0, None))


@partial(jit, static_argnums=(6,7,))
def log_like_np(theta, pt_sum, npt, data, f, rho_df, k_max, n_pix):
    """
    Args:
        theta  (NPT, 6) jnp.array: dnds parameters for NPT templates
        pt_sum (P,)     jnp.array: summed poissonian templates
        npt    (NPT, P) jnp.array: non-poissonian templates
        data   (P,)     jnp.array: data counts map
        f      (F,)     jnp.array: f values
        rho_df (F,)     jnp.array: rho * df values
        k_max           int      : maximum number of counts to consider
        n_pix           int      : number of pixels in ROI
    """

    #===== dnds =====
    s = jnp.logspace(-1, 2, 1000) # (S,)
    dnds = dnds_func_vmap(s, theta) # (NPT, S)

    #===== x_m =====
    x_m, x_m_sum = return_x_m_vmap(f, rho_df, npt, data, s, dnds, k_max) # (NPT, P, M), (NPT, P)
    x_m = jnp.sum(x_m, axis=0) # (P, M)
    x_m_sum = jnp.sum(x_m_sum, axis=0) # (P,)

    #===== p_k =====
    f0 = -(pt_sum + x_m_sum) # (P,)
    f1 = pt_sum + x_m[:, 1] # (P,)

    pk = jnp.zeros((n_pix, k_max + 1)) # (P, M)
    pk = pk.at[:, 0].set(jnp.exp(f0))
    pk = pk.at[:, 1].set(pk[:, 0] * f1)
    for k in np.arange(2, k_max + 1):
        n = jnp.arange(k - 1)
        pk = pk.at[:, k].set(jnp.sum((k - n) / k * x_m[:, k - n] * pk[:, n], axis=1) + f1 * pk[:, k - 1] / k)

    pk_data = pk[jnp.arange(n_pix), data] # (P,)

    return jnp.log(pk_data)

