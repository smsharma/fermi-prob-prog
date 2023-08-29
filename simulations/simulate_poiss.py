import jax.numpy as jnp

def simulate_mu(model, var_dict):
    """
    theta_{pib, ics}
    S_{iso, bub, psc, pib, ics}
    S_dsk zs C
    S_gce gamma_poiss f_bulge_poiss theta_bulge_poiss
    """
    
    mu = jnp.zeros_like(model.data, dtype=float)
    
    #===== rigid templates =====
    theta_pib = var_dict['theta_pib']
    temp_pib = jnp.sum(theta_pib[:, None] * model.pib, 0)
    theta_ics = var_dict['theta_ics']
    temp_ics = jnp.sum(theta_ics[:, None] * model.ics, 0)
    
    temps = [model.temp_iso, model.temp_bub, model.temp_psc, temp_pib, temp_ics]
    temp_labels = ['iso', 'bub', 'psc', 'pib', 'ics']
    for temp, temp_label in zip(temps, temp_labels):
        S_temp = var_dict[f'S_{temp_label}']
        A_temp = S_temp / jnp.mean(temp[~model.normalization_mask])
        mu += A_temp * temp     
    
    #===== disk =====
    S_dsk = var_dict['S_dsk']
    zs = var_dict['zs']
    C = var_dict['C']
    temp_dsk = model.disk_template.get_template(zs=zs, C=C)
    mu += S_dsk * temp_dsk
    
    #===== gce = nfw + bulge =====
    S_gce = var_dict['S_gce']
    
    gamma_poiss = var_dict['gamma_poiss']
    temp_gce_nfw_poiss = model.nfw_template.get_NFW2_template(gamma=gamma_poiss)

    f_bulge_poiss = var_dict['f_bulge_poiss']
    theta_bulge_poiss = var_dict['theta_bulge_poiss']
    temp_bulge = jnp.sum(theta_bulge_poiss[:, None] * model.bulge_templates, 0)
    
    A_gce_nfw = S_gce / jnp.mean(temp_gce_nfw_poiss[~model.normalization_mask])
    A_gce_bulge = S_gce / jnp.mean(temp_bulge[~model.normalization_mask])
    temp_gce_poiss = (1 - f_bulge_poiss) * A_gce_nfw * temp_gce_nfw_poiss \
                        + f_bulge_poiss * A_gce_bulge * temp_bulge
    
    A_gce = S_gce / jnp.mean(temp_gce_poiss[~model.normalization_mask])
    mu += A_gce * temp_gce_poiss
    
    return mu