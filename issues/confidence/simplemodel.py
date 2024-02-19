import sys

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
import jax.numpy as jnp

sys.path.append("../..")
from models.scd import dnds
from likelihoods.npll_jax import log_like_np

from common import *

def model(data=..., k_max=..., npixROI=..., deltapsf=False):
    # data is unmasked

    vd = truth_dict

    # general setting
    nm = mask_plane
    m = mask_plane
    data_in = data[~m]

    # poisson
    mu = jnp.zeros_like(data)[~m]

    # disk: param
    zs = numpyro.sample("zs", dist.Uniform(0.1, 2.0))
    #C = numpyro.sample("C", dist.Uniform(0.05, 8.))
    C = vd['C']
    temp_dsk = disk_template.get_template(zs=zs, C=C)

    # disk: normalization
    A_dsk = 1 / jnp.mean(temp_dsk[~nm])
    temp_dsk = A_dsk * temp_dsk
    npt_compressed = jnp.array([temp_dsk])

    # disk: scd
    theta = []
    for ips, ps in enumerate(["dsk"]):

        Sps = numpyro.sample("Sps_{}".format(ps), dist.Uniform(1e-3, 4.))
        # n1 = numpyro.sample("n1_{}".format(ps), dist.Uniform(4.0, 6.0))
        # n2 = numpyro.sample("n2_{}".format(ps), dist.Uniform(0.5, 1.99))
        # n3 = numpyro.sample("n3_{}".format(ps), dist.Uniform(-6., -5.))
        # sb1 = numpyro.sample("sb1_{}".format(ps), dist.Uniform(5., 40.0))
        # lambda_s = numpyro.sample("lambdas_{}".format(ps), dist.Uniform(0.1, 0.95))
        n1 = vd['n1_dsk']
        n2 = vd['n2_dsk']
        n3 = vd['n3_dsk']
        sb1 = vd['sb1_dsk']
        lambda_s = vd['lambdas_dsk']

        theta_tmp = jnp.array([1., n1, n2, n3, sb1, lambda_s * sb1])
        s_ary = jnp.logspace(-1., 2., 100)
        dnds_ary = dnds(s_ary, theta_tmp)
        A = Sps / jnp.mean(npt_compressed[ips][~nm] * jnp.trapz(s_ary * dnds_ary, s_ary))
        theta.append([A, n1, n2, n3, sb1, lambda_s * sb1])
    theta = jnp.array(theta)

    npt_compressed_in = jnp.array([temp_dsk[~m]])

    # psf
    if deltapsf:
        f_ary_in = f_ary_delta
        df_rho_div_f_ary_in = df_rho_div_f_ary_delta
    else:
        f_ary_in = f_ary
        df_rho_div_f_ary_in = df_rho_div_f_ary
            
    with numpyro.plate("data", size=len(mu), dim=-1):

        ll = log_like_np(theta, mu, npt_compressed_in, data_in, f_ary_in, df_rho_div_f_ary_in, k_max, npixROI)

        with handlers.mask(mask=~jnp.logical_or(jnp.isinf(ll), jnp.isnan(ll))):
            return numpyro.factor('log-likelihood', ll)