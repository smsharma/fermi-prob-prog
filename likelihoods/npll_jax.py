import sys

sys.path.append("./")

import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

from models.scd import dnds


def log_like_np(pt_sum_compressed, theta, npt_compressed, data, f_ary, df_rho_div_f_ary):
    """ Organize and combine non-Poissonian likelihoods across multiple templates
    """

    k_max = jnp.max(data) + 1
    npixROI = len(pt_sum_compressed)

    x_m_ary = jnp.zeros((npixROI, int(k_max) + 1))
    x_m_sum = jnp.zeros(npixROI)

    s_ary = jnp.logspace(-2, 2, 1000)

    for i in jnp.arange(len(theta)):
        dnds_ary = dnds(s_ary, theta[i])

        x_m_ary_out, x_m_sum_out = return_x_m(f_ary, df_rho_div_f_ary, npt_compressed[i], data, s_ary, dnds_ary)
        x_m_ary += x_m_ary_out
        x_m_sum += x_m_sum_out

    return log_like_internal(pt_sum_compressed, data, x_m_ary, x_m_sum)


def log_like_internal(pt_sum_compressed, data, x_m_ary, x_m_sum):
    """ Non-Poissonian likelihood for single template, given x_m and x_m_sum
    """

    k_max = jnp.max(data) + 1
    npixROI = len(pt_sum_compressed)

    f0_ary = -(pt_sum_compressed + x_m_sum)
    f1_ary = pt_sum_compressed + x_m_ary[:, 1]

    pk_ary = jnp.zeros((npixROI, int(k_max) + 1))

    pk_ary = pk_ary.at[:, 0].set(jnp.exp(f0_ary))
    pk_ary = pk_ary.at[:, 1].set(pk_ary[:, 0] * f1_ary)

    for k in jnp.arange(2, k_max + 1):

        n = jnp.arange(k - 1)
        pk_ary = pk_ary.at[:, k].set(jnp.sum((k - n) / k * x_m_ary[:, k - n] * pk_ary[:, n], axis=1) + f1_ary * pk_ary[:, k - 1] / k)
    
    pk_dat_ary = (pk_ary[jnp.arange(npixROI), data])

    return jnp.log(pk_dat_ary)


def return_x_m(f_ary, df_rho_div_f_ary, npt_compressed, data, s_ary, dnds_ary):
    """ Dedicated calculation of x_m and x_m_sum
    """

    k_max = jnp.max(data) + 1
    m_ary = jnp.arange(k_max + 1 , dtype=jnp.float64)
    gamma_ary = jnp.exp(jax.lax.lgamma(m_ary + 1))

    x_m_ary = df_rho_div_f_ary[:, None] * f_ary[:, None] * jnp.trapz(((dnds_ary * jnp.exp(-jnp.outer(f_ary, s_ary)))[:, :, None] * jax.lax.pow(jnp.outer(f_ary, s_ary)[:, :, None], m_ary[None, None, :]) / gamma_ary), s_ary, axis=1)
    x_m_ary = jnp.sum(x_m_ary, axis=0)

    x_m_ary = jnp.outer(npt_compressed, x_m_ary)

    # Get x_m_sum_ary array

    x_m_sum_ary = jnp.sum((df_rho_div_f_ary * f_ary)[:, None] * jnp.trapz(dnds_ary, s_ary), axis=0)
    x_m_sum_ary = jnp.sum(x_m_sum_ary, axis=0)

    x_m_sum_ary = npt_compressed * x_m_sum_ary - x_m_ary[:, 0]

    return x_m_ary, x_m_sum_ary
