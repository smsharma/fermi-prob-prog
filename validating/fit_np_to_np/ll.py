import sys

from jax import jit
import jax.numpy as jnp
from functools import partial

sys.path.append("../..")
from likelihoods.npll_jax import log_like_np
from models.scd import dnds

def ll(m, vd, data):

    assert m.non_poissonian

    theta_pib = vd['theta_pib']
    theta_ics = vd['theta_ics']
    temp_pib = jnp.sum(theta_pib[:, None] * m.pib, 0)
    temp_ics = jnp.sum(theta_ics[:, None] * m.ics, 0)

    temps = [m.temp_iso, m.temp_bub, m.temp_psc, temp_pib, temp_ics]
    temp_labels = ["iso", "bub", "psc", "pib", "ics"]
            
    mu = jnp.zeros_like(data)

    for temp, temp_label in zip(temps, temp_labels):
        S_temp = vd["S_{}".format(temp_label)]
        if temp_label in ["pib"]:
            temp_pib_mod = jnp.zeros_like(data)
            for ii in range(len(m.Ylm_temps)):
                Alm = vd["Alm_{}".format(ii)]
                temp_pib_mod += Alm * m.Ylm_temps[ii]
            temp_pib_mod = (1. + temp_pib_mod) * temp
            A_temp = S_temp / jnp.mean(temp_pib_mod[~m.normalization_mask])
            mu += A_temp * temp_pib_mod  
        else:
            A_temp = S_temp / jnp.mean(temp[~m.normalization_mask])
            mu += A_temp * temp     
                                        
    if m.vary_gamma:
        gamma_ps = vd["gamma_ps"]
        gamma_poiss = vd["gamma_poiss"]
    else:
        gamma_ps = 1.2
        gamma_poiss = 1.2
    temp_gce_nfw_ps = m.nfw_template.get_NFW2_template(gamma=gamma_ps)
    temp_gce_nfw_poiss = m.nfw_template.get_NFW2_template(gamma=gamma_poiss)

    if m.vary_disk:
        zs = vd["zs"]
        C = vd["C"]
        temp_dsk = m.disk_template.get_template(zs=zs, C=C)
    else:
        temp_dsk = m.temp_dsk
            
    if m.bulge_hybrid:
        f_bulge_ps = vd["f_bulge_ps"]
        f_bulge_poiss = vd["f_bulge_poiss"]
        theta_bulge_poiss = vd["theta_bulge_poiss"]
        temp_bulge = jnp.sum(theta_bulge_poiss[:, None] * m.bulge_templates, 0)
    else:
        f_bulge_ps = 0.
        f_bulge_poiss = vd["f_bulge_poiss"]
        temp_bulge = m.bulge_templates[0]

    A_gce_nfw = vd['S_gce'] / jnp.mean(temp_gce_nfw_poiss[~m.normalization_mask])
    A_gce_bulge = vd['S_gce'] / jnp.mean(temp_bulge[~m.normalization_mask])
    temp_gce_poiss = (1 - f_bulge_poiss) * A_gce_nfw * temp_gce_nfw_poiss \
                        + f_bulge_poiss * A_gce_bulge * temp_bulge

    A_gce = vd['S_gce'] / jnp.mean(temp_gce_poiss[~m.normalization_mask])
    mu += A_gce * temp_gce_poiss

    theta_bulge_ps = vd["theta_bulge_ps"]
    temp_bulge = jnp.sum(theta_bulge_ps[:, None] * m.bulge_templates, 0)

    A_gce_nfw = 1 / jnp.mean(temp_gce_nfw_ps[~m.normalization_mask])
    A_gce_bulge = 1 / jnp.mean(temp_bulge[~m.normalization_mask])

    temp_gce_ps = (1 - f_bulge_ps) * A_gce_nfw * temp_gce_nfw_ps + f_bulge_ps * A_gce_bulge * temp_bulge
    npt_compressed = jnp.array([temp_gce_ps, temp_dsk])
    theta = []    
    for ips, ps in enumerate(["gce", "dsk"]):
        Sps = vd["Sps_{}".format(ps)]
        n1 = vd["n1_{}".format(ps)]
        n2 = vd["n2_{}".format(ps)]
        n3 = vd["n3_{}".format(ps)]
        sb1 = vd["sb1_{}".format(ps)]
        lambda_s = vd["lambdas_{}".format(ps)]
        theta_tmp = jnp.array([1., n1, n2, n3, sb1, lambda_s * sb1])
        s_ary = jnp.logspace(-1., 2, 100)
        dnds_ary = dnds(s_ary, theta_tmp)
        A = Sps / jnp.mean(npt_compressed[ips][~m.normalization_mask] * jnp.trapz(s_ary * dnds_ary, s_ary))
        theta.append([A, n1, n2, n3, sb1, lambda_s * sb1])
    theta = jnp.array(theta)
    
    mu_masked = mu[~m.mask_roi]
    npt_compressed_masked = npt_compressed[:, ~m.mask_roi]
    data_masked = data[~m.mask_roi]
    ll_arr = log_like_np(theta, mu_masked, npt_compressed_masked, data_masked, m.f_ary, m.df_rho_div_f_ary, m.k_max, len(mu_masked))
    return jnp.sum(ll_arr)

#@partial(jit, static_argnums=(1,))
def ll_justSps(m, vd, data):

    assert m.non_poissonian
            
    mu = jnp.zeros_like(data)

    gamma_ps = 1.2
    temp_gce_nfw_ps = m.nfw_template.get_NFW2_template(gamma=gamma_ps)

    zs = vd["zs"]
    C = vd["C"]
    temp_dsk = m.disk_template.get_template(zs=zs, C=C)

    A_gce_nfw = 1 / jnp.mean(temp_gce_nfw_ps[~m.normalization_mask])

    temp_gce_ps = A_gce_nfw * temp_gce_nfw_ps
    npt_compressed = jnp.array([temp_gce_ps, temp_dsk])
    theta = []    
    for ips, ps in enumerate(["gce", "dsk"]):
        Sps = vd["Sps_{}".format(ps)]
        n1 = vd["n1_{}".format(ps)]
        n2 = vd["n2_{}".format(ps)]
        n3 = vd["n3_{}".format(ps)]
        sb1 = vd["sb1_{}".format(ps)]
        lambda_s = vd["lambdas_{}".format(ps)]
        theta_tmp = jnp.array([1., n1, n2, n3, sb1, lambda_s * sb1])
        s_ary = jnp.logspace(-1., 2, 100)
        dnds_ary = dnds(s_ary, theta_tmp)
        A = Sps / jnp.mean(npt_compressed[ips][~m.normalization_mask] * jnp.trapz(s_ary * dnds_ary, s_ary))
        theta.append([A, n1, n2, n3, sb1, lambda_s * sb1])
    theta = jnp.array(theta)
    
    mask = m.normalization_mask

    mu_masked = mu[~mask]
    npt_compressed_masked = npt_compressed[:, ~mask]
    data_masked = data[~mask]
    ll_arr = log_like_np(theta, mu_masked, npt_compressed_masked, data_masked, m.f_ary, m.df_rho_div_f_ary, m.k_max, len(mu_masked))
    return jnp.sum(ll_arr)


def ll_justSps_nosum(m, vd, data, mask_fit=None):

    assert m.non_poissonian
            
    mu = jnp.zeros_like(data)

    gamma_ps = 1.2
    temp_gce_nfw_ps = m.nfw_template.get_NFW2_template(gamma=gamma_ps)

    zs = vd["zs"]
    C = vd["C"]
    temp_dsk = m.disk_template.get_template(zs=zs, C=C)

    A_gce_nfw = 1 / jnp.mean(temp_gce_nfw_ps[~m.normalization_mask])

    temp_gce_ps = A_gce_nfw * temp_gce_nfw_ps
    npt_compressed = jnp.array([temp_gce_ps, temp_dsk])
    theta = []    
    for ips, ps in enumerate(["gce", "dsk"]):
        Sps = vd["Sps_{}".format(ps)]
        n1 = vd["n1_{}".format(ps)]
        n2 = vd["n2_{}".format(ps)]
        n3 = vd["n3_{}".format(ps)]
        sb1 = vd["sb1_{}".format(ps)]
        lambda_s = vd["lambdas_{}".format(ps)]
        theta_tmp = jnp.array([1., n1, n2, n3, sb1, lambda_s * sb1])
        s_ary = jnp.logspace(-1., 2, 100)
        dnds_ary = dnds(s_ary, theta_tmp)
        A = Sps / jnp.mean(npt_compressed[ips][~m.normalization_mask] * jnp.trapz(s_ary * dnds_ary, s_ary))
        theta.append([A, n1, n2, n3, sb1, lambda_s * sb1])
    theta = jnp.array(theta)
    
    if mask_fit is not None:
        mask = mask_fit
    else:
        mask = m.normalization_mask

    mu_masked = mu[~mask]
    npt_compressed_masked = npt_compressed[:, ~mask]
    data_masked = data[~mask]
    ll_arr = log_like_np(theta, mu_masked, npt_compressed_masked, data_masked, m.f_ary, m.df_rho_div_f_ary, m.k_max, len(mu_masked))
    return ll_arr


def ll_dsk(m, vd, data):

    assert m.non_poissonian
            
    mu = jnp.zeros_like(data)

    zs = vd["zs"]
    C = vd["C"]
    temp_dsk = m.disk_template.get_template(zs=zs, C=C)

    npt_compressed = jnp.array([temp_dsk])
    theta = []    
    for ips, ps in enumerate(["dsk"]):
        Sps = vd["Sps_{}".format(ps)]
        n1 = vd["n1_{}".format(ps)]
        n2 = vd["n2_{}".format(ps)]
        n3 = vd["n3_{}".format(ps)]
        sb1 = vd["sb1_{}".format(ps)]
        lambda_s = vd["lambdas_{}".format(ps)]
        theta_tmp = jnp.array([1., n1, n2, n3, sb1, lambda_s * sb1])
        s_ary = jnp.logspace(-1., 2, 100)
        dnds_ary = dnds(s_ary, theta_tmp)
        A = Sps / jnp.mean(npt_compressed[ips][~m.normalization_mask] * jnp.trapz(s_ary * dnds_ary, s_ary))
        theta.append([A, n1, n2, n3, sb1, lambda_s * sb1])
    theta = jnp.array(theta)
    
    mu_masked = mu[~m.mask_roi]
    npt_compressed_masked = npt_compressed[:, ~m.mask_roi]
    data_masked = data[~m.mask_roi]
    ll_arr = log_like_np(theta, mu_masked, npt_compressed_masked, data_masked, m.f_ary, m.df_rho_div_f_ary, m.k_max, len(mu_masked))
    return jnp.sum(ll_arr)

def ll_gce(m, vd, data):

    assert m.non_poissonian
            
    mu = jnp.zeros_like(data)

    gamma_ps = 1.2
    temp_gce_nfw_ps = m.nfw_template.get_NFW2_template(gamma=gamma_ps)

    npt_compressed = jnp.array([temp_gce_nfw_ps])
    theta = []    
    for ips, ps in enumerate(["gce"]):
        Sps = vd["Sps_{}".format(ps)]
        n1 = vd["n1_{}".format(ps)]
        n2 = vd["n2_{}".format(ps)]
        n3 = vd["n3_{}".format(ps)]
        sb1 = vd["sb1_{}".format(ps)]
        lambda_s = vd["lambdas_{}".format(ps)]
        theta_tmp = jnp.array([1., n1, n2, n3, sb1, lambda_s * sb1])
        s_ary = jnp.logspace(0., 2, 100)
        dnds_ary = dnds(s_ary, theta_tmp)
        A = Sps / jnp.mean(npt_compressed[ips][~m.normalization_mask] * jnp.trapz(s_ary * dnds_ary, s_ary))
        theta.append([A, n1, n2, n3, sb1, lambda_s * sb1])
    theta = jnp.array(theta)
    
    mu_masked = mu[~m.mask_roi]
    npt_compressed_masked = npt_compressed[:, ~m.mask_roi]
    data_masked = data[~m.mask_roi]
    ll_arr = log_like_np(theta, mu_masked, npt_compressed_masked, data_masked, m.f_ary, m.df_rho_div_f_ary, m.k_max, len(mu_masked))
    return jnp.sum(ll_arr)